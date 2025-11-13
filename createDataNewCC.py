import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps

# ================= 配置区域 =================
VIEW_NAME = "CC_val" 
# VIEW_NAME = "MLO" # 跑 MLO 时取消注释这行，并修改下面的路径

# --- 路径配置 ---
# 1. 原始图像 (必须提供，用于裁剪正样本)
ORIGINAL_IMAGES_DIR = Path('/kaggle/input/breast/data/cc_view/images/validation') 

# 2. 真实的 Ground Truth 标签路径 (YOLO格式)
GT_LABELS_DIR = Path('/kaggle/input/breast/data/cc_view/labels/validation')

# 3. YOLO detect 运行结果的路径 (用于裁剪负样本)
DETECT_RUN_DIR = Path('/kaggle/input/siamesedata/siameseD/val/cc')

# 4. 最终输出的根目录
OUTPUT_ROOT = Path('/kaggle/working/siamese_data_from_paper_strategy')

# --- 参数配置 ---
# 论文中使用的 patch 尺寸
PATCH_SIZE = (64, 64) # [cite: 199] 
# IoU 阈值
NEG_IOU_THRESHOLD = 0.1 # 论文中使用 < 10^-4 [cite: 198]，但 < 0.1 效果类似

# ==========================================

# --- 创建输出目录 ---
CURRENT_OUTPUT_DIR = OUTPUT_ROOT / VIEW_NAME
POSITIVE_PATCH_DIR = CURRENT_OUTPUT_DIR / 'positive_patches'
NEGATIVE_PATCH_DIR = CURRENT_OUTPUT_DIR / 'negative_patches'
POSITIVE_PATCH_DIR.mkdir(parents=True, exist_ok=True)
NEGATIVE_PATCH_DIR.mkdir(parents=True, exist_ok=True)

# --- 辅助函数 ---
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2; y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2; y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def yolo_to_pixel(yolo_box, img_width, img_height):
    """将 YOLO 相对坐标 [x_c, y_c, w, h] 转为 PIL crop 用的像素坐标 [x1, y1, x2, y2]"""
    x_c, y_c, w, h = yolo_box
    x1 = (x_c - w / 2) * img_width
    y1 = (y_c - h / 2) * img_height
    x2 = (x_c + w / 2) * img_width
    y2 = (y_c + h / 2) * img_height
    return [int(x1), int(y1), int(x2), int(y2)]

def compute_iou_numpy(box1, boxes2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    inter_rect_x1 = np.maximum(b1_x1, b2_x1); inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2); inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    inter_area = np.maximum(0, inter_rect_x2 - inter_rect_x1) * np.maximum(0, inter_rect_y2 - inter_rect_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-6)
    return iou

# =================================================================
print(f"--- 策略 1: 正在从 YOLO 检测结果中提取 负样本 (Negative Patches) ---")
# =================================================================

pred_labels_dir = DETECT_RUN_DIR / 'labels'
crops_dir = DETECT_RUN_DIR / 'crops'
cnt_neg = 0

# 1. 递归查找所有 YOLO detect 保存的 crop 图片
all_crop_files = {}
print("Finding all detected crop files...")
for root, dirs, files in os.walk(crops_dir):
    for f in files:
        if f.endswith('.jpg'):
            all_crop_files[f] = os.path.join(root, f)
print(f"Found {len(all_crop_files)} crop files.")

# 2. 遍历所有 *预测* 标签
pred_files = list(pred_labels_dir.glob('*.txt'))
for pred_path in tqdm(pred_files, desc=f"Processing {VIEW_NAME} Negatives"):
    
    parts = pred_path.stem.rpartition('_')
    original_filename = parts[0] # P001_L_CC
    crop_index = parts[2]        # 0
    
    gt_path = GT_LABELS_DIR / f"{original_filename}.txt"
    
    # 读取 Pred Box
    with open(pred_path, 'r') as f:
        line = f.read().strip()
        if not line: continue
        vals = list(map(float, line.split()))
    pred_cls_id = int(vals[0])
    pred_box_xyxy = xywh2xyxy(np.array(vals[1:5]))

    max_iou = 0.0 # 默认为 0
    if gt_path.exists():
        gt_data_list = []
        with open(gt_path, 'r') as f:
            for line in f:
                gt_data_list.append(list(map(float, line.strip().split()))[1:]) # 只需坐标
        
        if gt_data_list:
            gt_boxes_xyxy = xywh2xyxy(np.array(gt_data_list))
            ious = compute_iou_numpy(pred_box_xyxy, gt_boxes_xyxy)
            max_iou = np.max(ious)

    # --- 负样本决策 ---
    # 如果与 *所有* GT 的 IoU 都很低，它就是 False Positive [cite: 197]
    if max_iou < NEG_IOU_THRESHOLD: 
        crop_filename = f"{pred_path.stem}.jpg"
        if crop_filename in all_crop_files:
            src_crop_path = all_crop_files[crop_filename]
            
            # 打开, resize, 保存
            try:
                with Image.open(src_crop_path) as img:
                    img_resized = img.resize(PATCH_SIZE, Image.LANCZOS)
                    # 确保是灰度图 (L) 或 RGB (如果需要)
                    img_resized = img_resized.convert("L") 
                    
                    new_filename = f"{original_filename}_idx{crop_index}_gtNone_pred{pred_cls_id}.jpg"
                    dest_path = NEGATIVE_PATCH_DIR / new_filename
                    img_resized.save(dest_path)
                    cnt_neg += 1
            except Exception as e:
                print(f"Warning: Failed to process neg sample {src_crop_path}: {e}")

# =================================================================
print(f"--- 策略 2: 正在从 Ground Truth 中提取 正样本 (Positive Patches) ---")
# =================================================================

gt_files = list(GT_LABELS_DIR.glob('*.txt'))
cnt_pos = 0
for gt_path in tqdm(gt_files, desc=f"Processing {VIEW_NAME} Positives"):
    
    original_filename = gt_path.stem # P001_L_CC
    
    # 找到对应的原始图像
    # 尝试 .jpg, .png 等常见格式
    img_path = None
    for ext in ['.jpg', '.png', '.jpeg']:
        potential_path = ORIGINAL_IMAGES_DIR / f"{original_filename}{ext}"
        if potential_path.exists():
            img_path = potential_path
            break
            
    if not img_path:
        print(f"Warning: No original image found for GT {gt_path.name}")
        continue
        
    try:
        with Image.open(img_path) as img:
            img_width, img_height = img.size
            
            # 打开 GT 标签并遍历 *所有* 病灶
            with open(gt_path, 'r') as f:
                for j, line in enumerate(f):
                    vals = line.strip().split()
                    if not vals: continue
                    
                    gt_cls_id = int(vals[0])
                    yolo_box = np.array(vals[1:], dtype=float)
                    
                    # 将 GT YOLO 坐标转为像素坐标
                    pixel_box = yolo_to_pixel(yolo_box, img_width, img_height)
                    
                    # 裁剪图像 
                    cropped_img = img.crop(pixel_box)
                    
                    # 调整大小 
                    cropped_img_resized = cropped_img.resize(PATCH_SIZE, Image.LANCZOS)
                    # 确保是灰度图 (L) 或 RGB (如果需要)
                    cropped_img_resized = cropped_img_resized.convert("L") 
                    
                    # 按论文生成 K 个样本 [cite: 190, 191]
                    # 为简单起见，我们先只保存这个完美的 GT 裁剪 (K=1)
                    # 论文中的 K=5 和 IoU > 0.5 [cite: 190, 191] 是数据增强，
                    # 可以在 Data Loader 阶段实时 (on-the-fly) 完成。
                    
                    new_filename = f"{original_filename}_gtidx{j}_cls{gt_cls_id}.jpg"
                    dest_path = POSITIVE_PATCH_DIR / new_filename
                    cropped_img_resized.save(dest_path)
                    cnt_pos += 1
                    
    except Exception as e:
        print(f"Warning: Failed to process pos sample {img_path.name}: {e}")


print(f"--- [{VIEW_NAME}] Finished! ---")
print(f"Total Positive Patches (from GT): {cnt_pos}")
print(f"Total Negative Patches (from YOLO FP): {cnt_neg}")
print(f"Data saved in: {CURRENT_OUTPUT_DIR}")
