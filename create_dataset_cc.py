import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
VIEW_NAME = "CC"
DETECT_RUN_DIR = Path('/kaggle/input/siamesedata1/cc_candidates/cc_candidates')
GT_LABELS_DIR = Path('/kaggle/input/breast/data/cc_view/labels/train')
OUTPUT_ROOT = Path('/kaggle/working/siamese_data_cleaned_multitask') # 用个新名字以示区别
# ==========================================

CURRENT_OUTPUT_DIR = OUTPUT_ROOT / VIEW_NAME
(CURRENT_OUTPUT_DIR / 'positive').mkdir(parents=True, exist_ok=True)
(CURRENT_OUTPUT_DIR / 'negative').mkdir(parents=True, exist_ok=True)

def xywh2xyxy(x):
    # convert [x_c, y_c, w, h] -> [x1, y1, x2, y2]
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def compute_iou_numpy(box1, boxes2):
    # cal box1 and box2 IOU
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(0, inter_rect_x2 - inter_rect_x1) * np.maximum(0, inter_rect_y2 - inter_rect_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-6)
    return iou

print(f"Starting data cleaning for view: {VIEW_NAME}...")
pred_labels_dir = DETECT_RUN_DIR / 'labels'
crops_dir = DETECT_RUN_DIR / 'crops'

# 使用 os.walk 递归查找所有 crop 图片
all_crop_files = {}
for root, dirs, files in os.walk(crops_dir):
    for f in files:
        if f.endswith('.jpg'):
            all_crop_files[f] = os.path.join(root, f)

cnt_pos = 0
cnt_neg = 0

pred_files = list(pred_labels_dir.glob('*.txt'))
for pred_path in tqdm(pred_files, desc=f"Processing {VIEW_NAME}"):
    parts = pred_path.stem.rpartition('_')
    original_filename = parts[0]
    crop_index = parts[2]
    
    gt_path = GT_LABELS_DIR / f"{original_filename}.txt"
    if not gt_path.exists(): continue

    gt_boxes = []
    with open(gt_path, 'r') as f:
        for line in f:
            # GT 也需要读取类别，用于更精确的匹配（可选，这里暂时只用位置匹配）
            gt_boxes.append(list(map(float, line.strip().split()))[1:]) 
    if not gt_boxes: continue
    gt_boxes_xyxy = xywh2xyxy(np.array(gt_boxes))

    with open(pred_path, 'r') as f:
        line = f.read().strip()
        if not line: continue
        vals = list(map(float, line.split()))
        
        # --- 关键修改：读取预测类别 ---
        pred_cls_id = int(vals[0]) # 第一个数是类别 ID
        pred_box = np.array(vals[1:5])

    pred_box_xyxy = xywh2xyxy(pred_box)
    ious = compute_iou_numpy(pred_box_xyxy, gt_boxes_xyxy)
    max_iou = np.max(ious)

    crop_filename = f"{pred_path.stem}.jpg"
    if crop_filename in all_crop_files:
        src_crop = all_crop_files[crop_filename]
        
        # --- 关键修改：在保存的文件名中加入类别信息 ---
        # 新文件名格式: {原始文件名}_cls{类别ID}_{索引}.jpg
        new_filename = f"{original_filename}_cls{pred_cls_id}_{crop_index}.jpg"
        
        if max_iou > 0.5:
            shutil.copy(src_crop, CURRENT_OUTPUT_DIR / 'positive' / new_filename)
            cnt_pos += 1
        elif max_iou < 0.1:
            # 负样本也保留类别信息，因为它被模型预测为了某个类
            shutil.copy(src_crop, CURRENT_OUTPUT_DIR / 'negative' / new_filename)
            cnt_neg += 1

print(f"[{VIEW_NAME}] Finished! Positive: {cnt_pos}, Negative: {cnt_neg}")
# suitable for pairing
'''
import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


VIEW_NAME = "CC" 

# yolo detect dir
DETECT_RUN_DIR = Path('/kaggle/input/siamesedata1/cc_candidates/cc_candidates')

# ori gt dir
GT_LABELS_DIR = Path('/kaggle/input/breast/data/cc_view/labels/train')

# save dir
OUTPUT_ROOT = Path('/kaggle/working/siamese_data_cleaned')
# =========================================================

# Define the output directory for the current view
CURRENT_OUTPUT_DIR = OUTPUT_ROOT / VIEW_NAME
(CURRENT_OUTPUT_DIR / 'positive').mkdir(parents=True, exist_ok=True)
(CURRENT_OUTPUT_DIR / 'negative').mkdir(parents=True, exist_ok=True)

def xywh2xyxy(x):
    # convert [x_c, y_c, w, h] -> [x1, y1, x2, y2]
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def compute_iou_numpy(box1, boxes2):
    # cal box1 and box2 IOU
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(0, inter_rect_x2 - inter_rect_x1) * np.maximum(0, inter_rect_y2 - inter_rect_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-6)
    return iou

print(f"Starting data cleaning for view: {VIEW_NAME}...")
pred_labels_dir = DETECT_RUN_DIR / 'labels'
crops_dir = DETECT_RUN_DIR / 'crops'

# get class name -> directory
class_names = [d.name for d in crops_dir.iterdir() if d.is_dir()]

cnt_pos = 0
cnt_neg = 0


# loop
pred_files = list(pred_labels_dir.glob('*.txt'))
for pred_path in tqdm(pred_files, desc=f"Processing {VIEW_NAME}"):
    
    # Parse filenames: P001_L_CC_0.txt -> ori file name P001_L_CC, index 0
    # rpartition('_')  split from right-hand side，suitable for {filename}_{index}.txt 
    parts = pred_path.stem.rpartition('_')
    original_filename = parts[0]
    crop_index = parts[2]
    
    # find gt labels
    gt_path = GT_LABELS_DIR / f"{original_filename}.txt"
    if not gt_path.exists(): continue

    # read gt box
    gt_boxes = []
    with open(gt_path, 'r') as f:
        for line in f:
            gt_boxes.append(list(map(float, line.strip().split()))[1:]) # ignore class only take coordinate ******** need changes 
    if not gt_boxes: continue
    gt_boxes_xyxy = xywh2xyxy(np.array(gt_boxes))

    # current prediction bounding box
    with open(pred_path, 'r') as f:
        line = f.read().strip()
        if not line: continue
        vals = list(map(float, line.split()))
        pred_box = np.array(vals[1:5]) # [x_c, y_c, w, h]
        
    pred_box_xyxy = xywh2xyxy(pred_box)

    # cal Max IOU
    ious = compute_iou_numpy(pred_box_xyxy, gt_boxes_xyxy)
    max_iou = np.max(ious)

    # crop
    crop_found = False
    for cls_name in class_names:
        potential_crop = crops_dir / cls_name / f"{pred_path.stem}.jpg"
        if potential_crop.exists():
            src_crop = potential_crop
            crop_found = True
            break
    if not crop_found: continue

    # final split decision
    # Only keep high-quality positive samples (IoU > 0.5) and clean negative samples.
    if max_iou > 0.5:
        shutil.copy(src_crop, CURRENT_OUTPUT_DIR / 'positive' / f"{original_filename}_{crop_index}.jpg")
        cnt_pos += 1
    elif max_iou < 0.1:
        shutil.copy(src_crop, CURRENT_OUTPUT_DIR / 'negative' / f"{original_filename}_{crop_index}.jpg")
        cnt_neg += 1

print(f"[{VIEW_NAME}] Finished! Positive: {cnt_pos}, Negative: {cnt_neg}")
'''
