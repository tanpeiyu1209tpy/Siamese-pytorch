import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import math
import re

# ============================
# Parse patch filename
# ============================
def parse_patch_name(name):
    """
    imageid_CC_pred3_yolo10.png
    return: image_id, pred_idx, yolo_idx
    """
    name = name.replace(".png", "")

    m = re.match(r"(.+?)_[A-Z]+_pred(\d+)_yolo(\d+)", name)
    if not m:
        return None
    return (
        m.group(1),      # image_id
        int(m.group(2)), # pred index
        int(m.group(3))  # yolo line index
    )


# ============================
# IoU function
# ============================
def bbox_iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter_area + 1e-16

    return inter_area / union


# ============================
# Compute AP
# ============================
def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


# ============================
# MAIN EVAL
# ============================
def evaluate(gt_dir, yolo_pred_dir, siamese_csv):

    df = pd.read_csv(siamese_csv)

    preds_by_class = {0:[], 1:[]}

    print("🔍 Extracting Siamese + YOLO fused predictions...")

    for _, row in df.iterrows():

        for patch_name, pred_class in [
            (row["CC_patch"], row["cc_pred_class"]),
            (row["MLO_patch"], row["mlo_pred_class"])
        ]:

            pred_class = int(pred_class)
            if pred_class not in [0,1]:
                continue

            parsed = parse_patch_name(patch_name)
            if parsed is None:
                continue

            image_id, pred_idx, yolo_idx = parsed

            txt_path = os.path.join(yolo_pred_dir, f"{image_id}.txt")
            if not os.path.exists(txt_path):
                continue

            lines = open(txt_path).read().strip().split("\n")
            if yolo_idx >= len(lines):
                continue

            parts = lines[yolo_idx].split()
            _, xc, yc, w, h, conf = parts

            # Siamese distance
            dist = float(row["distance"])

            # final fused score
            match_score = math.exp(-dist)
            final_score = float(conf) * match_score

            preds_by_class[pred_class].append(
                (image_id, final_score, [float(xc), float(yc), float(w), float(h)])
            )

    # ============================
    # Load GT
    # ============================
    gt_by_class = {0: defaultdict(list), 1: defaultdict(list)}

    for fname in os.listdir(gt_dir):
        if fname.endswith(".txt"):
            img_id = fname.replace(".txt", "")
            lines = open(os.path.join(gt_dir, fname)).read().strip().split("\n")
            for line in lines:
                parts = line.split()
                cls = int(parts[0])
                if cls in [0,1]:
                    bbox = list(map(float, parts[1:5]))
                    gt_by_class[cls][img_id].append({"bbox": bbox, "used": False})

    # ============================
    # Evaluate
    # ============================
    print("\n==============================")
    print("   FINAL EVALUATION (YOLO Style)")
    print("==============================\n")
    print(f"{'Class':<15}{'Instances':<12}{'P':<8}{'R':<8}{'mAP50':<10}{'mAP50-95'}")

    for cls_id in [0,1]:

        preds = sorted(preds_by_class[cls_id], key=lambda x: x[1], reverse=True)
        gts = gt_by_class[cls_id]
        total_gt = sum(len(v) for v in gts.values())

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        # matching
        for i, (img_id, score, box_pred) in enumerate(preds):

            if img_id not in gts or len(gts[img_id]) == 0:
                fp[i] = 1
                continue

            ious = np.array([
                bbox_iou(box_pred, g["bbox"]) for g in gts[img_id]
            ])
            best = np.argmax(ious)

            if ious[best] >= 0.5 and not gts[img_id][best]["used"]:
                tp[i] = 1
                gts[img_id][best]["used"] = True
            else:
                fp[i] = 1

        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)

        recall = tp_c / (total_gt + 1e-16)
        precision = tp_c / (tp_c + fp_c + 1e-16)

        ap50 = compute_ap(recall, precision)

        # AP50-95
        aps = []
        for thr in np.arange(0.5, 0.96, 0.05):

            # reset GT
            g_clone = defaultdict(list)
            for img, v in gts.items():
                g_clone[img] = [ {"bbox": g["bbox"], "used": False} for g in v ]

            tp2 = np.zeros(len(preds))
            fp2 = np.zeros(len(preds))

            for i, (img_id, score, box_pred) in enumerate(preds):
                if img_id not in g_clone:
                    fp2[i] = 1
                    continue

                ious = np.array([
                    bbox_iou(box_pred, g["bbox"]) for g in g_clone[img_id]
                ])
                best = np.argmax(ious)

                if ious[best] >= thr and not g_clone[img_id][best]["used"]:
                    tp2[i] = 1
                    g_clone[img_id][best]["used"] = True
                else:
                    fp2[i] = 1

            r = np.cumsum(tp2) / (total_gt + 1e-16)
            p = tp2.cumsum() / (tp2.cumsum() + fp2.cumsum() + 1e-16)
            aps.append(compute_ap(r, p))

        ap5095 = np.mean(aps)

        cname = "Mass" if cls_id == 0 else "Suspicious_Calcification"

        print(f"{cname:<15}{total_gt:<12}{precision[-1]:<8.3f}{recall[-1]:<8.3f}{ap50:<10.3f}{ap5095:.3f}")
