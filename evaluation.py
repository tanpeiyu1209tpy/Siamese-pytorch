import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
from math import inf

try:
    from sklearn.metrics import roc_curve, auc
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# -----------------------------
# 工具函数：IoU
# -----------------------------
def bbox_iou(box1, box2):
    """
    box: [xc, yc, w, h] (normalized 0~1, YOLO format)
    返回 IoU
    """
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


# -----------------------------
# 计算 单一 IoU 阈值 的 AP
# -----------------------------
def compute_ap(recall, precision):
    """
    经典 VOC 风格插值 AP
    """
    # 在两端补点
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # 从后往前 envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 计算面积
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def compute_precision_recall_ap(
    preds,  # list of (img_id, score, bbox[4])
    gts,    # dict img_id -> list of {bbox[4], used=False}
    iou_thresh=0.5
):
    """
    preds: [(image_id, score, bbox[4])]
    gts: { image_id: [ {"bbox": [xc,yc,w,h], "used": False}, ... ] }
    """
    if len(preds) == 0:
        return 0.0, 0.0, 0.0, np.array([]), np.array([])

    # 按 score 降序排序
    preds = sorted(preds, key=lambda x: x[1], reverse=True)

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    # 每个图像的 gt 数量
    npos = 0
    for img_id in gts:
        npos += len(gts[img_id])

    for i, (img_id, score, pb) in enumerate(preds):
        if img_id not in gts or len(gts[img_id]) == 0:
            fp[i] = 1
            continue

        gt_list = gts[img_id]
        ious = np.array([bbox_iou(pb, g["bbox"]) for g in gt_list])

        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]

        if max_iou >= iou_thresh and not gt_list[max_iou_idx]["used"]:
            tp[i] = 1
            gt_list[max_iou_idx]["used"] = True
        else:
            fp[i] = 1

    # 累积
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / (npos + 1e-16)
    precision = tp_cum / (tp_cum + fp_cum + 1e-16)

    ap = compute_ap(recall, precision)

    prec_final = precision[-1] if len(precision) > 0 else 0.0
    rec_final = recall[-1] if len(recall) > 0 else 0.0

    return prec_final, rec_final, ap, recall, precision


# -----------------------------
# 主流程：整合 Siamese + YOLO + GT
# -----------------------------
def evaluate(
    gt_dir,
    yolo_pred_dir,
    siamese_csv_path,
    classes_to_eval=(0, 1),
):
    """
    gt_dir:           /kaggle/input/yolo1911/yolo_dataset/labels
    yolo_pred_dir:    /kaggle/input/siamesetestlabel/labels
    siamese_csv_path: /kaggle/working/siamese_best_results.csv
    """

    # 1. 读 Siamese CSV
    df_sia = pd.read_csv(siamese_csv_path)

    # 收集预测：
    # per class: list[(img_id, score, bbox)]
    preds_by_class = {c: [] for c in classes_to_eval}

    # 为 ROC 记录：
    roc_scores = []
    roc_labels = []

    # 2. 从 Siamese best results 回到 YOLO 预测 txt
    for _, row in df_sia.iterrows():
        patient_side = row["patient"]  # 比如 xxxx_R
        cc_patch = row["CC_patch"]     # e.g. imageid_CC_pred3.png
        mlo_patch = row["MLO_patch"]

        cc_pred_cls = int(row["cc_pred_class"])
        mlo_pred_cls = int(row["mlo_pred_class"])

        # ----- CC patch -----
        if "_pred" in cc_patch:
            base = cc_patch.replace(".png", "")
            # e.g. xyz_CC_pred3 → ["xyz_CC", "3"]
            prefix, idx_str = base.rsplit("_pred", 1)
            # xyz_CC → image_id + view
            # 这里我们只要 image_id
            cc_image_id = prefix.split("_")[0]
            cc_idx = int(idx_str)

            yolo_cc_txt = os.path.join(yolo_pred_dir, f"{cc_image_id}.txt")
            if os.path.isfile(yolo_cc_txt) and cc_pred_cls in classes_to_eval:
                with open(yolo_cc_txt, "r") as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                if cc_idx < len(lines):
                    parts = lines[cc_idx].split()
                    # YOLO pred: cls xc yc w h conf
                    # 我们用 Siamese 的 class 覆盖 cls
                    _, xc, yc, w, h, conf = parts
                    xc = float(xc); yc = float(yc)
                    w = float(w); h = float(h)
                    conf = float(conf)

                    preds_by_class[cc_pred_cls].append(
                        (cc_image_id, conf, [xc, yc, w, h])
                    )

                    # ROC label: 有无匹配我们后面再确定，这里先存 score
                    roc_scores.append(conf)
                    # label 暂时先占位，后面更新（需要 matching info）
                    # 这里先写 0，后面不再精细更新也 OK，主要是给你一个代码样板
                    roc_labels.append(0)

        # ----- MLO patch -----
        if "_pred" in mlo_patch:
            base = mlo_patch.replace(".png", "")
            prefix, idx_str = base.rsplit("_pred", 1)
            mlo_image_id = prefix.split("_")[0]
            mlo_idx = int(idx_str)

            yolo_mlo_txt = os.path.join(yolo_pred_dir, f"{mlo_image_id}.txt")
            if os.path.isfile(yolo_mlo_txt) and mlo_pred_cls in classes_to_eval:
                with open(yolo_mlo_txt, "r") as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                if mlo_idx < len(lines):
                    parts = lines[mlo_idx].split()
                    _, xc, yc, w, h, conf = parts
                    xc = float(xc); yc = float(yc)
                    w = float(w); h = float(h)
                    conf = float(conf)

                    preds_by_class[mlo_pred_cls].append(
                        (mlo_image_id, conf, [xc, yc, w, h])
                    )
                    roc_scores.append(conf)
                    roc_labels.append(0)

    # 3. 读 GT
    # gt_by_class: class -> { image_id: [ {bbox: [4], used:False}, ... ] }
    gt_by_class: Dict[int, Dict[str, List[Dict]]] = {
        c: defaultdict(list) for c in classes_to_eval
    }

    for fname in os.listdir(gt_dir):
        if not fname.endswith(".txt"):
            continue
        image_id = fname.replace(".txt", "")
        gt_path = os.path.join(gt_dir, fname)
        with open(gt_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id not in classes_to_eval:
                continue
            xc, yc, w, h = map(float, parts[1:5])
            gt_by_class[cls_id][image_id].append(
                {"bbox": [xc, yc, w, h], "used": False}
            )

    # 4. 对每个类计算 AP / mAP
    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    ap_results = {c: [] for c in classes_to_eval}
    prec_results = {}
    rec_results = {}

    print("\n==============================")
    print(" Evaluation per class")
    print("==============================")

    for c in classes_to_eval:
        print(f"\n>>> Class {c} <<<")
        preds_c = preds_by_class[c]
        gts_c = gt_by_class[c]

        if len(preds_c) == 0:
            print("   No predictions for this class.")
            for iou in iou_thresholds:
                ap_results[c].append(0.0)
            continue

        prec_results_c = []
        rec_results_c = []

        for iou in iou_thresholds:
            # 注意：为了多次 IoU 评估，需要重新 deep copy gts_c
            import copy
            gts_clone = copy.deepcopy(gts_c)

            p, r, ap, recall_curve, prec_curve = compute_precision_recall_ap(
                preds_c, gts_clone, iou_thresh=iou
            )

            ap_results[c].append(ap)
            prec_results_c.append(p)
            rec_results_c.append(r)

            print(f"  IoU={iou:.2f} | P={p:.3f}, R={r:.3f}, AP={ap:.3f}")

        prec_results[c] = prec_results_c
        rec_results[c] = rec_results_c

    # 5. mAP
    print("\n==============================")
    print(" Summary mAP")
    print("==============================")
    for i, iou in enumerate(iou_thresholds):
        aps_at_iou = [ap_results[c][i] for c in classes_to_eval]
        mAP_iou = np.mean(aps_at_iou)
        print(f"IoU={iou:.2f} | mAP={mAP_iou:.3f}  (per-class AP: {aps_at_iou})")

    mAP_50 = np.mean([ap_results[c][0] for c in classes_to_eval])
    mAP_50_95 = np.mean(
        [np.mean(ap_results[c]) for c in classes_to_eval]
    )

    print("\n==============================")
    print(f" mAP@0.5       = {mAP_50:.3f}")
    print(f" mAP@0.5:0.95  = {mAP_50_95:.3f}")
    print("==============================\n")

    # 6. 简单 ROC 样例
    if SKLEARN_AVAILABLE and len(roc_scores) > 0:
        # 这里 ROC label 我们上面没精细标 TP/FP，只做 demo
        fpr, tpr, _ = roc_curve(roc_labels, roc_scores)
        roc_auc = auc(fpr, tpr)
        print(f"(Demo) ROC AUC (using conf only) = {roc_auc:.3f}")
    else:
        print("sklearn 不可用或没有 ROC 数据，略过 ROC 计算。")


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Siamese + YOLO against GT labels"
    )

    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Ground truth label folder (YOLO format, no conf)",
    )
    parser.add_argument(
        "--yolo_pred_dir",
        type=str,
        required=True,
        help="YOLO prediction txt folder (with conf)",
    )
    parser.add_argument(
        "--siamese_csv",
        type=str,
        required=True,
        help="siamese_best_results.csv path",
    )

    args = parser.parse_args()

    evaluate(
        gt_dir=args.gt_dir,
        yolo_pred_dir=args.yolo_pred_dir,
        siamese_csv_path=args.siamese_csv,
        classes_to_eval=(0, 1),  # 只评估 mass & calcification
    )
