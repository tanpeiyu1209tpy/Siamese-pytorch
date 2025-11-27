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

        cc_patches = sorted(os.listdir(os.path.join(cc_dir, p)))
        mlo_patches = sorted(os.listdir(os.path.join(mlo_dir, p)))

        # ⭐ 保留所有 positive patch
        for cc_f in cc_patches:
            cc_img = cv2.imread(os.path.join(cc_dir, p, cc_f))
            cc_tensor = transform(cc_img).unsqueeze(0).to(device)

            for mlo_f in mlo_patches:
                mlo_img = cv2.imread(os.path.join(mlo_dir, p, mlo_f))
                mlo_tensor = transform(mlo_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    dist, cc_logits, mlo_logits = model((cc_tensor, mlo_tensor))

                cc_pred = torch.argmax(cc_logits, dim=1).item()
                mlo_pred = torch.argmax(mlo_logits, dim=1).item()

                # ⭐ 保留 positive（0:mass,1:cal）
                if cc_pred in (0,1) or mlo_pred in (0,1):
                    rows.append({
                        "patient": p,
                        "CC_patch": cc_f,
                        "MLO_patch": mlo_f,
                        "distance": dist.item(),
                        "cc_pred_class": cc_pred,
                        "mlo_pred_class": mlo_pred
                    })

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
