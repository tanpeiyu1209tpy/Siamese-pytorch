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
    pattern = r"^(.*?)_([LR])_(CC|MLO)_(pos|neg)_?(\d+)\.png$"
    m = re.match(pattern, fname)
    if m:
        return {
            "patient_id": m.group(1),
            "side": m.group(2),
            "view": m.group(3),
            "patch_type": m.group(4),
            "idx": int(m.group(5))
        }
    return None


# ===============================================================
# ⭐ Final Version: CMCNet Siamese Dataset ⭐
# ===============================================================
class SiameseDataset(Dataset):
    def __init__(self, root_dir, input_size=(64, 64), random_flag=True):
        print("📌 Loading FINAL CMCNet SiameseDataset")

        self.root_dir = root_dir
        self.random_flag = random_flag
        self.input_size = input_size

        # Class mapping
        self.class_map = {"Mass": 0, "Calcification": 1, "Negative": 2}

        # pid → CC/MLO positive & negative patches
        self.data = {}

        # -------------------------------------------------
        # Scan folders
        # -------------------------------------------------
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
                        "CC_neg": [], "MLO_neg": [],
                        "cls": self.class_map[cls_name]
                    }

                fpath = os.path.join(cls_dir, fname)
                label = self.class_map[cls_name]

                # Store into correct bucket
                if parsed["view"] == "CC":
                    if parsed["patch_type"] == "pos":
                        self.data[pid]["CC_pos"].append((fpath, label, parsed["idx"]))
                    else:
                        self.data[pid]["CC_neg"].append((fpath, 2))
                else:
                    if parsed["patch_type"] == "pos":
                        self.data[pid]["MLO_pos"].append((fpath, label, parsed["idx"]))
                    else:
                        self.data[pid]["MLO_neg"].append((fpath, 2))

        # -------------------------------------------------
        # Patients with valid CC & MLO (positive or negative)
        # -------------------------------------------------
        self.valid_ids = [
            pid for pid, v in self.data.items()
            if (len(v["CC_pos"]) > 0 and len(v["MLO_pos"]) > 0)
        ]

        print(f"✔ Valid patient-sides: {len(self.valid_ids)}")

        # -------------------------------------------------
        # Transforms
        # -------------------------------------------------
        if random_flag:
            self.to_tensor = transforms.Compose([
                transforms.RandomRotation(25),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        else:
            self.to_tensor = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.valid_ids)

    # ======================================================
    # ⭐ Final sampling strategy (论文级)：1 pos + 1 neg
    # ======================================================
    def __getitem__(self, idx):
        pid = self.valid_ids[idx]
        v = self.data[pid]

        # ======================================================
        # ⭐ 1. Positive pair (index aligned)
        # ======================================================
        # Extract CC_pos and MLO_pos sorted by idx
        CC_pos_sorted = sorted(v["CC_pos"], key=lambda x: x[2])
        MLO_pos_sorted = sorted(v["MLO_pos"], key=lambda x: x[2])

        # Choose the same index (0–4)
        i = random.randint(0, min(len(CC_pos_sorted), len(MLO_pos_sorted)) - 1)

        cc_pos_path, cc_pos_lbl, _ = CC_pos_sorted[i]
        mlo_pos_path, mlo_pos_lbl, _ = MLO_pos_sorted[i]

        match_pos = 1.0  # same lesion

        # ======================================================
        # ⭐ 2. Negative pair (strict & stable)
        # ======================================================
        cc_neg_pool = v["CC_neg"] + [(p, lbl) for (p, lbl, _) in v["CC_pos"]]
        mlo_neg_pool = v["MLO_neg"] + [(p, lbl) for (p, lbl, _) in v["MLO_pos"]]

        case = random.random()
        if case < 0.33:
            # pos ↔ neg
            cc_neg_path, cc_neg_lbl = random.choice(v["CC_pos"])
            mlo_neg_path, mlo_neg_lbl = random.choice(v["MLO_neg"])
        elif case < 0.66:
            # neg ↔ pos
            cc_neg_path, cc_neg_lbl = random.choice(v["CC_neg"])
            mlo_neg_path, mlo_neg_lbl = random.choice(v["MLO_pos"])
        else:
            # neg ↔ neg
            cc_neg_path, cc_neg_lbl = random.choice(v["CC_neg"])
            mlo_neg_path, mlo_neg_lbl = random.choice(v["MLO_neg"])

        match_neg = 0.0

        # ======================================================
        # Load images
        # ======================================================
        cc_imgs = [
            self.load_image(cc_pos_path),
            self.load_image(cc_neg_path)
        ]
        mlo_imgs = [
            self.load_image(mlo_pos_path),
            self.load_image(mlo_neg_path)
        ]

        return (
            torch.stack(cc_imgs),
            torch.stack(mlo_imgs)
        ), (
            torch.tensor([match_pos, match_neg]).float(),
            torch.tensor([cc_pos_lbl, cc_neg_lbl]).long(),
            torch.tensor([mlo_pos_lbl, mlo_neg_lbl]).long(),
        )

    def load_image(self, path):
        return self.to_tensor(Image.open(path).convert("RGB"))



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
