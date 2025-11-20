import os
import random
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ===============================================================
# Parse filename: <patient>_<side>_<view>_(pos|neg)<idx>.png
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
# SiameseDataset — FINAL FIXED VERSION
# ===============================================================
class SiameseDataset(Dataset):
    def __init__(self, root_dir, input_size=(64,64), random_flag=True):
        self.root_dir = root_dir
        self.random_flag = random_flag
        self.input_size = input_size

        # ❗ class_map 只用于 POSITIVE 类别
        self.class_map = {"Mass": 0, "Calcification": 1}

        # Patient data container
        self.data = {}   # patient → {CC:[], MLO:[], cls:0/1, neg_CC:[], neg_MLO:[]}
        self.cls_to_patient = {0: [], 1: []}

        # ======================================================
        # Step 1 — 先读 Mass & Calcification（决定病人真实类别）
        # ======================================================
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

                # 初始化 patient 结构
                if pid not in self.data:
                    self.data[pid] = {
                        "CC": [], "MLO": [],
                        "neg_CC": [], "neg_MLO": [],
                        "cls": cls_id
                    }

                fpath = os.path.join(cls_dir, fname)
                self.data[pid][parsed["view"]].append({
                    "path": fpath,
                    "cls": cls_id,
                    "patch_type": "pos"
                })

        # ======================================================
        # Step 2 — 再读 Negative folder（不改变病人 class）
        # ======================================================
        neg_dir = os.path.join(root_dir, "Negative")
        if os.path.exists(neg_dir):
            for fname in os.listdir(neg_dir):
                parsed = parse_filename(fname)
                if parsed is None:
                    continue

                pid = f"{parsed['patient_id']}_{parsed['side']}"

                # 如果此病人根本不是 Mass/Calc → 丢掉
                if pid not in self.data:
                    continue

                fpath = os.path.join(neg_dir, fname)

                if parsed["view"] == "CC":
                    self.data[pid]["neg_CC"].append(fpath)
                else:
                    self.data[pid]["neg_MLO"].append(fpath)

        # ======================================================
        # Step 3 — 保留有 CC 和 MLO 的病人
        # ======================================================
        self.valid_ids = [
            pid for pid, v in self.data.items()
            if len(v["CC"]) > 0 and len(v["MLO"]) > 0
        ]

        for pid in self.valid_ids:
            self.cls_to_patient[self.data[pid]["cls"]].append(pid)

        print(f"Loaded {len(self.valid_ids)} valid patient-sides with CC+MLO.")

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
    # Produce: Positive pair + Negative pair
    # ======================================================
    def __getitem__(self, idx):
        pid = self.valid_ids[idx]
        cls_id = self.data[pid]["cls"]

        CC_pos_list = self.data[pid]["CC"]
        MLO_pos_list = self.data[pid]["MLO"]

        # -------------------------
        # Positive pair
        # -------------------------
        cc_pos = random.choice(CC_pos_list)
        mlo_pos = random.choice(MLO_pos_list)
        match_pos = 1.0

        # -------------------------
        # Negative pair — same class, different patient
        # -------------------------
        candidates = [p for p in self.cls_to_patient[cls_id] if p != pid]
        neg_pid = random.choice(candidates)

        # 从对方病人抽 CC/MLO
        cc_neg_candidates = self.data[neg_pid]["CC"] + \
                            [{"path": n, "cls": cls_id} for n in self.data[neg_pid]["neg_CC"]]

        mlo_neg_candidates = self.data[neg_pid]["MLO"] + \
                             [{"path": n, "cls": cls_id} for n in self.data[neg_pid]["neg_MLO"]]

        cc_neg = random.choice(cc_neg_candidates)
        mlo_neg = random.choice(mlo_neg_candidates)

        match_neg = 0.0

        # -------------------------
        # Load images
        # -------------------------
        cc_imgs = [
            self.load_image(cc_pos["path"]),
            self.load_image(cc_neg["path"])
        ]
        mlo_imgs = [
            self.load_image(mlo_pos["path"]),
            self.load_image(mlo_neg["path"])
        ]

        match_labels = torch.tensor([match_pos, match_neg])
        cc_cls = torch.tensor([cls_id, cls_id])
        mlo_cls = torch.tensor([cls_id, cls_id])

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
# collate_fn
# ===============================================================
def siamese_collate(batch):
    cc_imgs, mlo_imgs = [], []
    m_labels, cc_labels, mlo_labels = [], [], []

    for (cc, mlo), (m, cc_c, mlo_c) in batch:
        cc_imgs.append(cc)
        mlo_imgs.append(mlo)
        m_labels.append(m)
        cc_labels.append(cc_c)
        mlo_labels.append(mlo_c)

    return (
        torch.cat(cc_imgs),
        torch.cat(mlo_imgs)
    ), (
        torch.cat(m_labels),
        torch.cat(cc_labels),
        torch.cat(mlo_labels)
    )
