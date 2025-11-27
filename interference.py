import torch
import cv2
import os
import numpy as np
from nets.cmcnet import CMCNet
from torchvision import transforms
import pandas as pd
import argparse

transform = transforms.Compose([
    transforms.ToTensor()
])


def classify_single_patch(model, img_path, view, device):
    """对单个 patch 做分类，返回 predicted class"""
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        dist, cc_logits, mlo_logits = model((img_tensor, img_tensor))

    if view == "CC":
        pred = torch.argmax(cc_logits, dim=1).item()
    else:
        pred = torch.argmax(mlo_logits, dim=1).item()

    return pred


def run_full_inference(model_path, cc_dir, mlo_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🔍 Loading CMCNet model...")
    model = CMCNet(input_channels=3, num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    rows = []

    patients = sorted(os.listdir(cc_dir))
    print(f"📌 Found {len(patients)} patient-sides")

    for p in patients:
        print(f"➡ Processing {p} ...")

        # ---------------------------
        # CC patches (single patch classification)
        # ---------------------------
        cc_path = os.path.join(cc_dir, p)
        for cc_f in sorted(os.listdir(cc_path)):

            if "_pred" not in cc_f:
                continue

            img_path = os.path.join(cc_path, cc_f)

            pred_class = classify_single_patch(model, img_path, "CC", device)
            if pred_class not in (0, 1):   # skip negative 2
                continue

            # 解析 YOLO index
            idx = int(cc_f.split("_pred")[-1].split(".")[0])

            rows.append({
                "patient": p,
                "view": "CC",
                "patch": cc_f,
                "pred_class": pred_class,
                "yolo_index": idx
            })

        # ---------------------------
        # MLO patches
        # ---------------------------
        mlo_path = os.path.join(mlo_dir, p)
        for mlo_f in sorted(os.listdir(mlo_path)):

            if "_pred" not in mlo_f:
                continue

            img_path = os.path.join(mlo_path, mlo_f)

            pred_class = classify_single_patch(model, img_path, "MLO", device)
            if pred_class not in (0, 1):   # skip negative
                continue

            idx = int(mlo_f.split("_pred")[-1].split(".")[0])

            rows.append({
                "patient": p,
                "view": "MLO",
                "patch": mlo_f,
                "pred_class": pred_class,
                "yolo_index": idx
            })

    df = pd.DataFrame(rows)
    df.to_csv("siamese_filtered_patches.csv", index=False)
    print("💾 Saved siamese_filtered_patches.csv")

    return df


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMCNet patch-level inference")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cc_dir", type=str, required=True)
    parser.add_argument("--mlo_dir", type=str, required=True)

    args = parser.parse_args()

    print("🚀 Starting inference...\n")

    run_full_inference(
        model_path=args.model_path,
        cc_dir=args.cc_dir,
        mlo_dir=args.mlo_dir
    )
