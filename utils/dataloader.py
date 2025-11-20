import os
import random
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ===============================================================
# 解析档名： <patient>_<side>_<view>_(pos/neg)<idx>.png
# ===============================================================
def parse_filename(fname):
    pattern = r"^(.*?)_([LR])_(CC|MLO)_(pos|neg)(\d+)\.png$"
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
# Siamese Dataset（完全修复）
# ===============================================================
class SiameseDataset(Dataset):
    def __init__(self, root_dir, input_size=(64, 64), random_flag=True):
        self.root_dir = root_dir
        self.random_flag = random_flag
        self.input_size = input_size

        # Only Mass / Calcification are real positive classes
        self.class_map = {"Mass": 0, "Calcification": 1}
        NEG_CLASS = 2

        # patient → 正样本类别
        patient_cls_map = {}

        # ----------------------------------------------------
        # Step 1: Collect positive classes (Mass / Calcification)
        # ----------------------------------------------------
        for cls_name in ["Mass", "Calcification"]:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue

            cls_id = self.class_map[cls_name]

            for fname in os.listdir(cls_dir):
                parsed = parse_filename(fname)
                if parsed is None:
                    continue

                pid = f"{parsed['patient_id']}_{parsed['side']}"
                patient_cls_map[pid] = cls_id   # 正确 patient-level class

        # ----------------------------------------------------
        # Step 2: Load ALL patches (Mass/Calcification/Negative)
        # ----------------------------------------------------
        self.data = {}

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
                    self.data[pid] = {"CC": [], "MLO": [], "cls": None}

                fpath = os.path.join(cls_dir, fname)

                self.data[pid][parsed["view"]].append({
                    "path": fpath,
                    "patch_type": parsed["patch_type"],
                    "folder": cls_name
                })

        # ----------------------------------------------------
        # Step 3: Assign patient class ONLY from positive patches
        # ----------------------------------------------------
        self.valid_ids = []
        for pid, entry in self.data.items():
            if pid not in patient_cls_map:
                continue   # skip purely negative patients

            if len(entry["CC"]) == 0 or len(entry["MLO"]) == 0:
                continue   # need both CC + MLO

            entry["cls"] = patient_cls_map[pid]
            self.valid_ids.append(pid)

        if len(self.valid_ids) == 0:
            raise ValueError("❌ No valid CC+MLO patients with positive class found!")

        print(f"Loaded {len(self.valid_ids)} valid patient-sides with CC+MLO.")

        # group patients by positive class
        self.cls_to_patient = {0: [], 1: []}
        for pid in self.valid_ids:
            cls_id = self.data[pid]["cls"]
            self.cls_to_patient[cls_id].append(pid)

        self.to_tensor = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.valid_ids)

    # --------------------------------------------------------------
    # Produce Positive pair + Negative pair
    # --------------------------------------------------------------
    def __getitem__(self, idx):
        pid = self.valid_ids[idx]
        cls_id = self.data[pid]["cls"]

        cc_list = self.data[pid]["CC"]
        mlo_list = self.data[pid]["MLO"]

        # Positive pair
        cc_pos = random.choice(cc_list)
        mlo_pos = random.choice(mlo_list)

        # Negative pair (same class, different patient)
        candidates = [p for p in self.cls_to_patient[cls_id] if p != pid]
        neg_pid = random.choice(candidates)

        cc_neg = random.choice(self.data[neg_pid]["CC"])
        mlo_neg = random.choice(self.data[neg_pid]["MLO"])

        cc_imgs = [
            self.load_image(cc_pos["path"]),
            self.load_image(cc_neg["path"])
        ]
        mlo_imgs = [
            self.load_image(mlo_pos["path"]),
            self.load_image(mlo_neg["path"])
        ]

        match_labels = torch.tensor([1.0, 0.0])
        cc_labels = torch.tensor([cls_id, cls_id])
        mlo_labels = torch.tensor([cls_id, cls_id])

        return (torch.stack(cc_imgs), torch.stack(mlo_imgs)), (
            match_labels, cc_labels, mlo_labels
        )

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.to_tensor(img)


# ===============================================================
# Collate
# ===============================================================
def siamese_collate(batch):
    cc_imgs = []
    mlo_imgs = []
    match_labels = []
    cc_labels = []
    mlo_labels = []

    for (cc, mlo), (m, c1, c2) in batch:
        cc_imgs.append(cc)
        mlo_imgs.append(mlo)
        match_labels.append(m)
        cc_labels.append(c1)
        mlo_labels.append(c2)

    return (torch.cat(cc_imgs, 0), torch.cat(mlo_imgs, 0)), (
        torch.cat(match_labels),
        torch.cat(cc_labels),
        torch.cat(mlo_labels)
    )
