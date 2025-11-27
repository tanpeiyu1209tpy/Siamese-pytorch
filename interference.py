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


def run_full_inference(model_path, cc_dir, mlo_dir, threshold=4.0):
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

        cc_patches = sorted(os.listdir(os.path.join(cc_dir, p)))
        mlo_patches = sorted(os.listdir(os.path.join(mlo_dir, p)))

        best_pair = None
        best_dist = 999

        for cc_f in cc_patches:
            cc_img = cv2.imread(os.path.join(cc_dir, p, cc_f))
            cc_tensor = transform(cc_img).unsqueeze(0).to(device)

            for mlo_f in mlo_patches:
                mlo_img = cv2.imread(os.path.join(mlo_dir, p, mlo_f))
                mlo_tensor = transform(mlo_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    dist, cc_logits, mlo_logits = model((cc_tensor, mlo_tensor))

                dist = dist.item()
                cc_pred = torch.argmax(cc_logits, dim=1).item()
                mlo_pred = torch.argmax(mlo_logits, dim=1).item()
                
                if cc_pred == 2 or mlo_pred == 2:
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_pair = (cc_f, mlo_f, dist, cc_pred, mlo_pred)

        rows.append({
            "patient": p,
            "CC_patch": best_pair[0],
            "MLO_patch": best_pair[1],
            "distance": best_pair[2],
            "cc_pred_class": best_pair[3],
            "mlo_pred_class": best_pair[4]
        })

    df = pd.DataFrame(rows)
    df.to_csv("siamese_best_results.csv", index=False)
    print("💾 Saved siamese_best_results.csv")

    return df


# =========================================================
# Main entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMCNet Siamese Inference")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to CMCNet .pth model")

    parser.add_argument("--cc_dir", type=str, required=True,
                        help="Directory containing CC patch folders")

    parser.add_argument("--mlo_dir", type=str, required=True,
                        help="Directory containing MLO patch folders")

    parser.add_argument("--threshold", type=float, default=4.0,
                        help="Matching threshold")

    args = parser.parse_args()

    print("🚀 Starting inference...\n")

    run_full_inference(
        model_path=args.model_path,
        cc_dir=args.cc_dir,
        mlo_dir=args.mlo_dir,
        threshold=args.threshold
    )
