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
            gt_boxes.append(list(map(float, line.strip().split()))[1:]) # 忽略类别，只取坐标
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
