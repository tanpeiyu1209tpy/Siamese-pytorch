import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# format: ("view_name","ori imges dir", "ori labels dir")
TASKS = [
    (
        "CC", 
        Path('/kaggle/input/breast/data/cc_view/images/train'), 
        Path('/kaggle/input/breast/data/cc_view/labels/train')
    ),
    (
        "MLO", 
        Path('/kaggle/input/breast/data/mlo_view/images/train'), 
        Path('/kaggle/input/breast/data/mlo_view/labels/train')
    )
]

# save path
OUTPUT_ROOT = Path('/kaggle/working/siamese_data_cleaned')

def crop_gt_regions(image_path, label_path, output_dir):

    img = cv2.imread(str(image_path))
    if img is None:
        # print(f"Warning: Could not read image {image_path}")
        return 0
    h_img, w_img = img.shape[:2]

    # read gt
    if not label_path.exists(): return 0
    
    count = 0
    with open(label_path, 'r') as f:
        for i, line in enumerate(f):
            # YOLO format: class x_center y_center width height 
            parts = list(map(float, line.strip().split()))
            if len(parts) < 5: continue
            
            cls_id, xc, yc, w, h = parts[0], parts[1], parts[2], parts[3], parts[4]
            
            # Convert normalized coordinates to absolute coor
            x1 = int((xc - w / 2) * w_img)
            y1 = int((yc - h / 2) * h_img)
            x2 = int((xc + w / 2) * w_img)
            y2 = int((yc + h / 2) * h_img)
            
            # make sure coordinate valid
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            
            # skip if box invalid 
            if x2 <= x1 or y2 <= y1: continue

            crop = img[y1:y2, x1:x2]
            
            # save crop img
            # name_format: {ori name}_GT_{index}.jpg
            
            
            
            #save_name = f"{image_path.stem}_GT_{i}.jpg"
            save_name = f"{image_path.stem}_GT_{int(cls_id)}_{i}.jpg"
            
            
            save_path = output_dir / save_name
            cv2.imwrite(str(save_path), crop)
            count += 1
            
    return count


for view_name, img_dir, label_dir in TASKS:
    print(f"Starting GT cropping for view: {view_name}...")
    
    # make sure put in pos folder
    pos_output_dir = OUTPUT_ROOT / view_name / 'positive'
    pos_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not img_dir.exists():
        print(f"Error: Image directory not found: {img_dir}")
        continue

    total_crops = 0

    image_files = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
    
    for img_path in tqdm(image_files, desc=f"Processing {view_name}"):
        # find label path
        label_path = label_dir / f"{img_path.stem}.txt"
        
        if label_path.exists():
            num = crop_gt_regions(img_path, label_path, pos_output_dir)
            total_crops += num
            
    print(f"[{view_name}] Finished! Generated {total_crops} GT positive crops.")
