import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 定义输入及其对应的输出目录
# 格式: ("视角名称", "原始图片目录", "原始标签目录")
TASKS = [
    (
        "CC", 
        Path('/kaggle/input/breast/data/cc_view/images/train'), 
        Path('/kaggle/input/breast/data/cc_view/labels/train')
    ),
    (
        "MLO", 
        # 请务必确认 MLO 的路径是否正确！
        Path('/kaggle/input/breast/data/mlo_view/images/train'), 
        Path('/kaggle/input/breast/data/mlo_view/labels/train')
    )
]

# 你的 Siamese 训练数据根目录
OUTPUT_ROOT = Path('/kaggle/working/siamese_data_cleaned')
# ==========================================

def crop_gt_regions(image_path, label_path, output_dir):
    # 1. 读取图像
    # 使用 cv2 读取，它可以处理各种常见格式
    img = cv2.imread(str(image_path))
    if img is None:
        # print(f"Warning: Could not read image {image_path}")
        return 0
    h_img, w_img = img.shape[:2]

    # 2. 读取 GT 标签
    if not label_path.exists(): return 0
    
    count = 0
    with open(label_path, 'r') as f:
        for i, line in enumerate(f):
            # YOLO 格式: class x_center y_center width height (归一化)
            parts = list(map(float, line.strip().split()))
            if len(parts) < 5: continue
            
            cls_id, xc, yc, w, h = parts[0], parts[1], parts[2], parts[3], parts[4]
            
            # 3. 将归一化坐标转换为绝对像素坐标
            x1 = int((xc - w / 2) * w_img)
            y1 = int((yc - h / 2) * h_img)
            x2 = int((xc + w / 2) * w_img)
            y2 = int((yc + h / 2) * h_img)
            
            # 确保坐标不超出图像边界
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            
            # 如果框无效（例如宽度或高度为0），跳过
            if x2 <= x1 or y2 <= y1: continue

            # 4. 裁剪图像
            crop = img[y1:y2, x1:x2]
            
            # 5. 保存裁剪结果
            # 命名格式: {原始文件名}_GT_{索引}.jpg
            # 加上 _GT_ 是为了和 YOLO 生成的区分开，方便你以后辨认
            # 但它依然符合 {ID}_{SIDE}_{VIEW}_... 的通用格式，Dataloader 能识别
            save_name = f"{image_path.stem}_GT_{i}.jpg"
            save_path = output_dir / save_name
            cv2.imwrite(str(save_path), crop)
            count += 1
            
    return count

# === 主循环 ===
for view_name, img_dir, label_dir in TASKS:
    print(f"Starting GT cropping for view: {view_name}...")
    
    # 确保输出目录存在 (直接放入 positive 文件夹)
    pos_output_dir = OUTPUT_ROOT / view_name / 'positive'
    pos_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not img_dir.exists():
        print(f"Error: Image directory not found: {img_dir}")
        continue

    total_crops = 0
    # 遍历所有图片文件
    # 这里假设图片格式可能是 png, jpg, jpeg，根据你实际情况调整
    image_files = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
    
    for img_path in tqdm(image_files, desc=f"Processing {view_name}"):
        # 找到对应的标签文件 (假设文件名相同，只是后缀不同)
        label_path = label_dir / f"{img_path.stem}.txt"
        
        if label_path.exists():
            num = crop_gt_regions(img_path, label_path, pos_output_dir)
            total_crops += num
            
    print(f"[{view_name}] Finished! Generated {total_crops} GT positive crops.")
