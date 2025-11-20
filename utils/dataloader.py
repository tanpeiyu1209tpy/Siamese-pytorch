import os
import random
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ===============================================================
# Parse filename: <pid>_<side>_<view>_(pos|neg)<idx>.png
# ===============================================================
def parse_filename(fname):
    pattern = r"^(.*?)_([LR])_(CC|MLO)_(pos|neg)(\d+)\.png$"
    m = re.match(pattern, fname)
    if m:
        return {
            "patient_id": m.group(1),
            "side": m.group(2),
            "view": m.group(3),
            "patch_type": m.group(4),   # pos / neg
            "idx": int(m.group(5))
        }
    return None


# ===============================================================
# NEW CMCNet Dataset – Correct multi-task sampling
# ===============================================================
class SiameseDataset(Dataset):
    def __init__(self, root_dir, input_size=(64, 64), random_flag=True):
        print("📌 Using NEW CMCNet SiameseDataset !!!")  # debug 标记
        self.root_dir = root_dir
        self.random_flag = random_flag
        self.input_size = input_size

        # Folder-based labels
        self.class_map = {"Mass": 0, "Calcification": 1, "Negative": 2}

        # Data storage
        # patient_id → CC_pos, MLO_pos, CC_neg, MLO_neg
        self.data = {}

        # -----------------------------
        # 1. Scan all folders
        # -----------------------------
        for cls_name in ["Mass", "Calcification", "Negative"]:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue

            for fname in os.listdir(cls_dir):
                parsed = parse_filename(fname)
                if parsed is None:
                    continue

                pid = f"{parsed['patient_id']}_{parsed['side']}"
                if pid not in self.data:
                    self.data[pid] = {
                        "CC_pos": [], "MLO_pos": [],
                        "CC_neg": [], "MLO_neg": []
                    }

                fpath = os.path.join(cls_dir, fname)

                # ---------------------------
                # Decide positive / negative
                # ---------------------------
                is_positive = (cls_name != "Negative")

                # ---------------------------
                # Store by view
                # ---------------------------
                if parsed["view"] == "CC":
                    if is_positive:
                        self.data[pid]["CC_pos"].append(
                            (fpath, self.class_map[cls_name])
                        )
                    else:
                        self.data[pid]["CC_neg"].append((fpath, 2))
                else:
                    if is_positive:
                        self.data[pid]["MLO_pos"].append(
                            (fpath, self.class_map[cls_name])
                        )
                    else:
                        self.data[pid]["MLO_neg"].append((fpath, 2))

        # -----------------------------
        # 2. Keep only patients with both views
        # -----------------------------
        self.valid_ids = [
            pid for pid, v in self.data.items()
            if len(v["CC_pos"]) > 0 and len(v["MLO_pos"]) > 0
        ]

        print(f"✔ Loaded {len(self.valid_ids)} valid patient-sides with CC+MLO.")

        # transform
        self.to_tensor = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.valid_ids)

    # ======================================================
    # Return 1 positive pair + 1 negative pair
    # ======================================================
    def __getitem__(self, idx):
        pid = self.valid_ids[idx]
        v = self.data[pid]

        # -------------------------
        # POSITIVE pair (same lesion = both positive)
        # -------------------------
        cc_pos_path, cc_pos_cls = random.choice(v["CC_pos"])
        mlo_pos_path, mlo_pos_cls = random.choice(v["MLO_pos"])

        match_pos = 1.0

        # Patch-level classification labels
        cc_pos_lbl = cc_pos_cls
        mlo_pos_lbl = mlo_pos_cls

        # -------------------------
        # NEGATIVE pair (正 × 负 或 负 × 正)
        # -------------------------
        cc_neg_pool = v["CC_neg"] + v["CC_pos"]
        mlo_neg_pool = v["MLO_neg"] + v["MLO_pos"]

        if random.random() < 0.5:
            # CC negative + MLO positive
            cc_neg_path, cc_neg_lbl = random.choice(cc_neg_pool)
            mlo_neg_path, mlo_neg_lbl = random.choice(v["MLO_pos"])
        else:
            # CC positive + MLO negative
            cc_neg_path, cc_neg_lbl = random.choice(v["CC_pos"])
            mlo_neg_path, mlo_neg_lbl = random.choice(mlo_neg_pool)

        match_neg = 0.0

        # -------------------------
        # load images
        # -------------------------
        cc_imgs = [
            self.load_image(cc_pos_path),
            self.load_image(cc_neg_path)
        ]

        mlo_imgs = [
            self.load_image(mlo_pos_path),
            self.load_image(mlo_neg_path)
        ]

        match_labels = torch.tensor([match_pos, match_neg])
        cc_cls = torch.tensor([cc_pos_lbl, cc_neg_lbl])
        mlo_cls = torch.tensor([mlo_pos_lbl, mlo_neg_lbl])

        return (
            torch.stack(cc_imgs),
            torch.stack(mlo_imgs)
        ), (
            match_labels.float(),
            cc_cls.long(),
            mlo_cls.long()
        )

    def load_image(self, path):
        return self.to_tensor(Image.open(path).convert("RGB"))


# ===============================================================
# Correct collate function
# ===============================================================
def siamese_collate(batch):
    cc_imgs, mlo_imgs = [], []
    match, cc_lbls, mlo_lbls = [], [], []

    for (cc, mlo), (m, c1, c2) in batch:
        cc_imgs.append(cc)
        mlo_imgs.append(mlo)
        match.append(m)
        cc_lbls.append(c1)
        mlo_lbls.append(c2)

    return (
        torch.cat(cc_imgs),
        torch.cat(mlo_imgs)
    ), (
        torch.cat(match),
        torch.cat(cc_lbls),
        torch.cat(mlo_lbls)
    )
