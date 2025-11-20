import os
import random
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ===============================================================
# 解析档名：
#   003700abcd_L_CC_pos0.png
#   <patient>_<side>_<view>_<pos/neg><idx>.png
# ===============================================================
def parse_filename(fname):
    pattern = r"^(.*?)_([LR])_(CC|MLO)_(pos|neg)(\d+)\.png$"
    m = re.match(pattern, fname)
    if m:
        return {
            "patient_id": m.group(1),
            "side": m.group(2),
            "view": m.group(3),      # CC or MLO
            "patch_type": m.group(4),  # pos / neg
            "idx": int(m.group(5))
        }
    return None


# ===============================================================
# 主要 Dataset
# ===============================================================
class SiameseDataset(Dataset):
    def __init__(self, root_dir, input_size=(64,64), random_flag=True):
        self.root_dir = root_dir
        self.random_flag = random_flag
        self.input_size = input_size

        # 3 类
        self.class_map = {"Mass": 0, "Calcification": 1, "Negative": 2}

        # self.data[patient_side] = {"CC":[...], "MLO":[...], "cls": class_id}
        self.data = {}
        self.cls_to_patient = {0: [], 1: [], 2: []}

        # ----------------------------------------------------
        # Step 1: load all images
        # ----------------------------------------------------
        for cls_name in ["Mass", "Calcification", "Negative"]:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue

            cls_id = self.class_map[cls_name]

            for fname in os.listdir(cls_dir):
                parsed = parse_filename(fname)
                if parsed is None:
                    continue

                patient_side = f"{parsed['patient_id']}_{parsed['side']}"

                if patient_side not in self.data:
                    self.data[patient_side] = {
                        "CC": [], "MLO": [], "cls": cls_id
                    }

                fpath = os.path.join(cls_dir, fname)
                self.data[patient_side][parsed["view"]].append({
                    "path": fpath,
                    "cls": cls_id,
                    "patch_type": parsed["patch_type"]
                })

        # ----------------------------------------------------
        # Step 2: keep only cases with CC + MLO
        # ----------------------------------------------------
        self.valid_ids = [
            pid for pid, v in self.data.items()
            if len(v["CC"]) > 0 and len(v["MLO"]) > 0
        ]

        if len(self.valid_ids) == 0:
            raise ValueError("❌ No valid CC+MLO pairs found!")

        # class grouping
        for pid in self.valid_ids:
            cls_id = self.data[pid]["cls"]
            self.cls_to_patient[cls_id].append(pid)

        print(f"Loaded {len(self.valid_ids)} valid patient-sides with CC+MLO.")

        # Transform
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.valid_ids)

    # --------------------------------------------------------
    # 产生 positive + negative pair
    # --------------------------------------------------------
    def __getitem__(self, idx):
        pid = self.valid_ids[idx]
        cls_id = self.data[pid]["cls"]

        cc_list = self.data[pid]["CC"]
        mlo_list = self.data[pid]["MLO"]

        # ---------------------------
        # Positive Pair
        # ---------------------------
        cc_pos = random.choice(cc_list)
        mlo_pos = random.choice(mlo_list)
        label_pos = 1.0

        # ---------------------------
        # Negative Pair（同类、不同 patient_side）
        # ---------------------------
        candidates = [p for p in self.cls_to_patient[cls_id] if p != pid]

        if len(candidates) == 0:
            candidates = [p for p in self.cls_to_patient[2] if p != pid]

        neg_pid = random.choice(candidates)

        # ❗ Negative CC 必须从 CC 选，Negative MLO 必须从 MLO 选
        cc_neg = random.choice(self.data[neg_pid]["CC"])
        mlo_neg = random.choice(self.data[neg_pid]["MLO"])
        label_neg = 0.0

        # ---------------------------
        # Load images
        # ---------------------------
        cc_imgs = [
            self.load_image(cc_pos["path"]),
            self.load_image(cc_neg["path"]),   # 修复!!
        ]
        mlo_imgs = [
            self.load_image(mlo_pos["path"]),
            self.load_image(mlo_neg["path"]),  # 修复!!
        ]

        match_labels = torch.tensor([label_pos, label_neg], dtype=torch.float32)

        cc_cls_labels = torch.tensor([cls_id, cc_neg["cls"]], dtype=torch.long)
        mlo_cls_labels = torch.tensor([cls_id, mlo_neg["cls"]], dtype=torch.long)

        return (
            torch.stack(cc_imgs),
            torch.stack(mlo_imgs)
        ), (
            match_labels,
            cc_cls_labels,
            mlo_cls_labels
        )

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.to_tensor(img)



# ===============================================================
# collate_fn
# ===============================================================
def siamese_collate(batch):
    cc_imgs, mlo_imgs = [], []
    match_labels, cc_cls_labels, mlo_cls_labels = [], [], []

    for (cc, mlo), (m, cc_c, mlo_c) in batch:
        cc_imgs.append(cc)
        mlo_imgs.append(mlo)
        match_labels.append(m)
        cc_cls_labels.append(cc_c)
        mlo_cls_labels.append(mlo_c)

    cc_imgs = torch.cat(cc_imgs, dim=0)
    mlo_imgs = torch.cat(mlo_imgs, dim=0)

    return (cc_imgs, mlo_imgs), (
        torch.cat(match_labels),
        torch.cat(cc_cls_labels),
        torch.cat(mlo_cls_labels)
    )
