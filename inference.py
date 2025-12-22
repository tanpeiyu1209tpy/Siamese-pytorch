import torch
import cv2
import os
import numpy as np
from nets.cmcnet import CMCNet
from torchvision import transforms
import pandas as pd
import argparse
import re

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 解析 patch 文件名
def parse_patch_filename(name):
    name = name.replace(".png", "")
    m = re.match(r"(.+?)_([A-Z]+)_pred(\d+)_yolo(\d+)", name)
    if not m:
        return None
    return (
        m.group(1),            # image_id
        m.group(2),            # view (CC/MLO)
        int(m.group(3)),       # pred index
        int(m.group(4))        # yolo index
    )


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

        cc_list = sorted(os.listdir(os.path.join(cc_dir, p)))
        mlo_list = sorted(os.listdir(os.path.join(mlo_dir, p)))

        best_pair = None  # ⭐ 最佳 pair
        best_dist = float("inf")

        # ----------------------------------------
        # Step 0 — compute all distances and pick global best
        # ----------------------------------------
        for cc_f in cc_list:
            cc_info = parse_patch_filename(cc_f)
            if cc_info is None:
                continue
            cc_imgid, cc_view, cc_pred_idx, cc_yolo_idx = cc_info

            cc_img = cv2.imread(os.path.join(cc_dir, p, cc_f))
            cc_tensor = transform(cc_img).unsqueeze(0).to(device)

            for mlo_f in mlo_list:
                mlo_info = parse_patch_filename(mlo_f)
                if mlo_info is None:
                    continue
                mlo_imgid, mlo_view, mlo_pred_idx, mlo_yolo_idx = mlo_info

                mlo_img = cv2.imread(os.path.join(mlo_dir, p, mlo_f))
                mlo_tensor = transform(mlo_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    dist, cc_logits, mlo_logits = model((cc_tensor, mlo_tensor))

                cc_pred = torch.argmax(cc_logits, dim=1).item()
                mlo_pred = torch.argmax(mlo_logits, dim=1).item()

                # only positive
                if cc_pred in (0,1) and mlo_pred in (0,1):

                    # ⭐ pick global best (minimum distance)
                    if dist.item() < best_dist:
                        best_dist = dist.item()
                        best_pair = {
                            "patient": p,
                            "CC_patch": cc_f,
                            "MLO_patch": mlo_f,
                            "distance": dist.item(),
                            "cc_pred_class": cc_pred,
                            "mlo_pred_class": mlo_pred,
                            "cc_image_id": cc_imgid,
                            "mlo_image_id": mlo_imgid,
                            "cc_yolo_idx": cc_yolo_idx,
                            "mlo_yolo_idx": mlo_yolo_idx
                        }

        # ----------------------------------------
        # Step 1 — save only ONE best pair for this patient
        # ----------------------------------------
        if best_pair is not None:
            rows.append(best_pair)

    df = pd.DataFrame(rows)
    df.to_csv("siamese_full_results.csv", index=False)
    print("💾 Saved siamese_full_results.csv")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMCNet Siamese Inference")
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
