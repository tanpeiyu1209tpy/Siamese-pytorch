'''
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

                # ⭐ 保留 positive（0:Mass, 1:Cal）🚫 不要 class 2
                if cc_pred in (0, 1) and mlo_pred in (0, 1):
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
'''

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
    transforms.ToTensor()
])

def parse_patch_filename(patch_name):
    """
    解析 patch 文件名:
    形如:  imageid_CC_pred3_yolo10.png
    返回: image_id, view, pred_idx, yolo_idx
    """
    base = patch_name.replace(".png", "")

    # 用 regex 拆分
    m = re.match(r"(.+?)_([A-Z]+)_pred(\d+)_yolo(\d+)", base)
    if not m:
        return None

    image_id = m.group(1)
    view = m.group(2)
    pred_idx = int(m.group(3))
    yolo_idx = int(m.group(4))

    return image_id, view, pred_idx, yolo_idx

'''
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

        for cc_f in cc_patches:
            cc_info = parse_patch_filename(cc_f)
            if cc_info is None:
                continue
            cc_image_id, cc_view, cc_pred_idx, cc_yolo_idx = cc_info

            cc_img = cv2.imread(os.path.join(cc_dir, p, cc_f))
            cc_tensor = transform(cc_img).unsqueeze(0).to(device)

            for mlo_f in mlo_patches:
                mlo_info = parse_patch_filename(mlo_f)
                if mlo_info is None:
                    continue
                mlo_image_id, mlo_view, mlo_pred_idx, mlo_yolo_idx = mlo_info

                mlo_img = cv2.imread(os.path.join(mlo_dir, p, mlo_f))
                mlo_tensor = transform(mlo_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    dist, cc_logits, mlo_logits = model((cc_tensor, mlo_tensor))

                cc_pred = torch.argmax(cc_logits, dim=1).item()
                mlo_pred = torch.argmax(mlo_logits, dim=1).item()

                # ⭐ 过滤掉 class 2
                if cc_pred in (0, 1) and mlo_pred in (0, 1):
                    rows.append({
                        "patient": p,

                        # patch 文件名
                        "CC_patch": cc_f,
                        "MLO_patch": mlo_f,

                        # Siamese 距离
                        "distance": dist.item(),

                        # Siamese 分类结果
                        "cc_pred_class": cc_pred,
                        "mlo_pred_class": mlo_pred,

                        # ⭐ 关键：记录 YOLO 对应哪一行预测
                        "cc_yolo_idx": cc_yolo_idx,
                        "mlo_yolo_idx": mlo_yolo_idx,

                        # ⭐ 关键：记录 image_id 方便 eval
                        "cc_image_id": cc_image_id,
                        "mlo_image_id": mlo_image_id
                    })

    df = pd.DataFrame(rows)
    df.to_csv("siamese_full_results.csv", index=False)
    print("💾 Saved siamese_full_results.csv")

    return df
'''

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

        # 先算所有 pair 距离
        all_pairs = []   # (cc_f, mlo_f, distance, cc_pred, mlo_pred)

        for cc_f in cc_list:
            cc_img = cv2.imread(os.path.join(cc_dir, p, cc_f))
            cc_tensor = transform(cc_img).unsqueeze(0).to(device)

            for mlo_f in mlo_list:
                mlo_img = cv2.imread(os.path.join(mlo_dir, p, mlo_f))
                mlo_tensor = transform(mlo_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    dist, cc_logits, mlo_logits = model((cc_tensor, mlo_tensor))

                cc_pred = torch.argmax(cc_logits, dim=1).item()
                mlo_pred = torch.argmax(mlo_logits, dim=1).item()

                # 过滤 class 2
                if cc_pred in (0,1) and mlo_pred in (0,1):
                    all_pairs.append({
                        "cc": cc_f,
                        "mlo": mlo_f,
                        "dist": dist.item(),
                        "cc_pred": cc_pred,
                        "mlo_pred": mlo_pred
                    })

        # -------------------------------------
        # ⭐ Step 1 — per-CC best match
        # -------------------------------------
        best_pairs = []
        cc_groups = {}

        for d in all_pairs:
            cc = d["cc"]
            if cc not in cc_groups or d["dist"] < cc_groups[cc]["dist"]:
                cc_groups[cc] = d

        best_pairs.extend(cc_groups.values())

        # -------------------------------------
        # ⭐ Step 2 — per-MLO best match
        # -------------------------------------
        mlo_groups = {}

        for d in all_pairs:
            mlo = d["mlo"]
            if mlo not in mlo_groups or d["dist"] < mlo_groups[mlo]["dist"]:
                mlo_groups[mlo] = d

        best_pairs.extend(mlo_groups.values())

        # -------------------------------------
        # ⭐ Step 3 — remove duplicate pairs
        # -------------------------------------
        seen = set()
        final = []
        for d in best_pairs:
            key = (d["cc"], d["mlo"])
            if key not in seen:
                seen.add(key)
                final.append(d)

        # -------------------------------------
        # ⭐ Step 4 — write to CSV
        # -------------------------------------
        for d in final:
            rows.append({
                "patient": p,
                "CC_patch": d["cc"],
                "MLO_patch": d["mlo"],
                "distance": d["dist"],
                "cc_pred_class": d["cc_pred"],
                "mlo_pred_class": d["mlo_pred"]
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
