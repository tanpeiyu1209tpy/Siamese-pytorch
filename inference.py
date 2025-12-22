import torch
import cv2
import os
import numpy as np
from nets.cmcnet import CMCNet
from torchvision import transforms
import pandas as pd
import argparse
import re
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Parse patch filename
# --------------------------------------------------
def parse_patch_filename(name):
    """
    Example:
    012e0595adba5173b6e60a97f9e84b6e_L_MLO_3.png
    """
    name = name.replace(".png", "").replace(".jpg", "")
    parts = name.split("_")
    if len(parts) < 4:
        return None

    image_id = parts[0]
    side     = parts[1]        # L / R
    view     = parts[2]        # CC / MLO
    idx      = int(parts[3])   # crop index

    return image_id, side, view, idx


# --------------------------------------------------
# Inference with soft probabilistic fusion
# --------------------------------------------------
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

        best_pair = None
        best_score = -1.0   # 🔧 CHANGED: maximize fusion score

        for cc_f in cc_list:
            cc_info = parse_patch_filename(cc_f)
            if cc_info is None:
                continue
            _, _, _, cc_idx = cc_info

            cc_img = cv2.imread(os.path.join(cc_dir, p, cc_f))
            cc_tensor = transform(cc_img).unsqueeze(0).to(device)

            for mlo_f in mlo_list:
                mlo_info = parse_patch_filename(mlo_f)
                if mlo_info is None:
                    continue
                _, _, _, mlo_idx = mlo_info

                mlo_img = cv2.imread(os.path.join(mlo_dir, p, mlo_f))
                mlo_tensor = transform(mlo_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    dist, cc_logits, mlo_logits = model((cc_tensor, mlo_tensor))

                # --------------------------------------------------
                # 1. Distance → match probability
                # --------------------------------------------------
                distance = dist.item()
                match_prob = torch.exp(-dist).item()   # 🔧 CHANGED

                # --------------------------------------------------
                # 2. Classification probability (lesion)
                # assume:
                #   class 0,1 = lesion
                #   class 2   = background
                # --------------------------------------------------
                cc_prob = F.softmax(cc_logits, dim=1)[0]
                mlo_prob = F.softmax(mlo_logits, dim=1)[0]

                cc_lesion_prob = (cc_prob[0] + cc_prob[1]).item()
                mlo_lesion_prob = (mlo_prob[0] + mlo_prob[1]).item()

                # --------------------------------------------------
                # 3. Soft probabilistic fusion score
                # --------------------------------------------------
                fusion_score = (
                    match_prob *
                    np.sqrt(cc_lesion_prob * mlo_lesion_prob)
                )  # 🔧 CHANGED

                # --------------------------------------------------
                # 4. Pick global best by fusion score
                # --------------------------------------------------
                if fusion_score > best_score:
                    best_score = fusion_score
                    best_pair = {
                        "patient": p,
                        "CC_patch": cc_f,
                        "MLO_patch": mlo_f,
                        "distance": distance,
                        "match_prob": match_prob,
                        "fusion_score": fusion_score,
                        "cc_lesion_prob": cc_lesion_prob,
                        "mlo_lesion_prob": mlo_lesion_prob,
                        "cc_idx": cc_idx,
                        "mlo_idx": mlo_idx
                    }

        if best_pair is not None:
            rows.append(best_pair)

    df = pd.DataFrame(rows)
    df.to_csv("siamese_full_results.csv", index=False)
    print("💾 Saved siamese_full_results.csv")

    return df


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMCNet Siamese Inference (Soft Fusion)")
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
