import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from math import inf
import math

# --------------------------------------------------------
# IoU function (YOLO format)
# --------------------------------------------------------
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


# --------------------------------------------------------
# Compute AP from sorted predictions
# --------------------------------------------------------
def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


# --------------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------------
def evaluate(gt_dir, yolo_pred_dir, siamese_csv):

    # ------------------------------------------
    # Load Siamese detections (all positive)
    # ------------------------------------------
    df_sia = pd.read_csv(siamese_csv)

    # preds_by_class = {0:[], 1:[]}
    preds_by_class = {0: [], 1: []}

    for _, row in df_sia.iterrows():
        # parse patch filename → image_id + index
        for view_patch, pred_class in [
            (row["CC_patch"], row["cc_pred_class"]),
            (row["MLO_patch"], row["mlo_pred_class"])
        ]:
            pred_class = int(pred_class)
            if pred_class not in [0, 1]:
                continue

            if "_pred" not in view_patch:
                continue

            base = view_patch.replace(".png", "")
            prefix, idx_str = base.rsplit("_pred", 1)
            image_id = prefix.split("_")[0]
            idx = int(idx_str)

            # load YOLO pred txt
            txt_path = os.path.join(yolo_pred_dir, f"{image_id}.txt")
            if not os.path.exists(txt_path):
                continue

            lines = open(txt_path).read().strip().split("\n")
            if idx >= len(lines):
                continue

            parts = lines[idx].split()
            _, xc, yc, w, h, conf = parts
            
            yolo_conf = float(conf)
            distance = float(row["distance"])   # ⭐从 Siamese CSV 拿 distance
            
            match_score = math.exp(-distance)   # ⭐越 similar score 越高
            final_score = yolo_conf * match_score   # ⭐融合 Siamese + YOLO
            
            preds_by_class[pred_class].append(
                (image_id, final_score, [float(xc), float(yc), float(w), float(h)])
            )

    # ------------------------------------------
    # Load GT
    # ------------------------------------------
    gt_by_class = {0: defaultdict(list), 1: defaultdict(list)}

    for fname in os.listdir(gt_dir):
        if not fname.endswith(".txt"):
            continue
        img_id = fname.replace(".txt", "")

        lines = open(os.path.join(gt_dir, fname)).read().strip().split("\n")
        for line in lines:
            parts = line.split()
            cls = int(parts[0])
            if cls not in [0, 1]:
                continue
            bbox = list(map(float, parts[1:5]))
            gt_by_class[cls][img_id].append({"bbox": bbox, "used": False})

    # ------------------------------------------
    # Evaluate for each class
    # ------------------------------------------
    result_rows = []

    for cls_id in [0, 1]:
        preds = sorted(preds_by_class[cls_id], key=lambda x: x[1], reverse=True)
        gts = gt_by_class[cls_id]

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        total_gt = sum(len(v) for v in gts.values())

        # match
        for i, (img_id, conf, pb) in enumerate(preds):
            if img_id not in gts or len(gts[img_id]) == 0:
                fp[i] = 1
                continue

            ious = np.array([bbox_iou(pb, g["bbox"]) for g in gts[img_id]])
            best = np.argmax(ious)
            best_iou = ious[best]

            if best_iou >= 0.5 and not gts[img_id][best]["used"]:
                tp[i] = 1
                gts[img_id][best]["used"] = True
            else:
                fp[i] = 1

        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        recall = tp_c / (total_gt + 1e-16)
        precision = tp_c / (tp_c + fp_c + 1e-16)

        ap50 = compute_ap(recall, precision)

        # compute AP50:95
        aps = []
        for thr in np.arange(0.5, 0.96, 0.05):
            # reset used flags
            from copy import deepcopy
            g_clone = deepcopy(gts_by_class:=gt_by_class[cls_id])
            preds_re = preds.copy()

            tp2 = np.zeros(len(preds_re))
            fp2 = np.zeros(len(preds_re))

            for i, (img_id, conf, pb) in enumerate(preds_re):
                if img_id not in g_clone or len(g_clone[img_id]) == 0:
                    fp2[i] = 1
                    continue
                ious = np.array([bbox_iou(pb, g["bbox"]) for g in g_clone[img_id]])
                best = np.argmax(ious)
                best_iou = ious[best]
                if best_iou >= thr and not g_clone[img_id][best]["used"]:
                    tp2[i] = 1
                    g_clone[img_id][best]["used"] = True
                else:
                    fp2[i] = 1

            r = np.cumsum(tp2) / (total_gt + 1e-16)
            p = np.cumsum(tp2) / (np.cumsum(tp2) + np.cumsum(fp2) + 1e-16)
            aps.append(compute_ap(r, p))

        ap5095 = np.mean(aps)

        P = precision[-1] if len(precision) > 0 else 0
        R = recall[-1] if len(recall) > 0 else 0

        result_rows.append([cls_id, total_gt, P, R, ap50, ap5095])

    # ------------------------------------------
    # Print YOLO-style result
    # ------------------------------------------
    print("\n==============================")
    print("   FINAL EVALUATION (YOLO Style)")
    print("==============================\n")
    print(f"{'Class':<15}{'Instances':<12}{'P':<8}{'R':<8}{'mAP50':<10}{'mAP50-95'}")

    for cls_id, total, P, R, ap50, ap5095 in result_rows:
        cname = "Mass" if cls_id == 0 else "Suspicious_Calcification"
        print(f"{cname:<15}{total:<12}{P:<8.3f}{R:<8.3f}{ap50:<10.3f}{ap5095:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Siamese + YOLO")

    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Path to ground truth YOLO labels folder (txt)")

    parser.add_argument("--yolo_pred_dir", type=str, required=True,
                        help="YOLO predicted label folder (txt with conf)")

    parser.add_argument("--siamese_csv", type=str, required=True,
                        help="siamese_full_results.csv path")

    args = parser.parse_args()

    evaluate(
        gt_dir=args.gt_dir,
        yolo_pred_dir=args.yolo_pred_dir,
        siamese_csv=args.siamese_csv
    )
