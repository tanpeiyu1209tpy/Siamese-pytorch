import os
import random
import re
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ===============================================================
# 解析文件名: <pid>_<side>_<view>_(pos|neg)<idx>.png
# 或       : <pid>_<side>_<view>_(pos|neg)_<idx>.png
# ===============================================================
def parse_filename(fname):
    pattern = r"^(.*?)_([LR])_(CC|MLO)_(pos|neg)_?(\d+)\.png$"
    m = re.match(pattern, fname)
    if m:
        return {
            "patient_id": m.group(1),
            "side": m.group(2),
            "view": m.group(3),          # CC / MLO
            "patch_type": m.group(4),    # pos / neg
            "idx": int(m.group(5))       # 0,1,2,...
        }
    return None


# ===============================================================
# CMCNet SiameseDataset — 论文版 (K=5)
# ===============================================================
class SiameseDataset(Dataset):
    def __init__(self, root_dir, input_size=(64, 64),
                 random_flag=True, k_per_lesion=5):
        print("📌 Using PAPER-STYLE CMCNet SiameseDataset !!!")

        self.root_dir = root_dir
        self.random_flag = random_flag
        self.input_size = input_size
        self.k_per_lesion = k_per_lesion

        # 文件夹 → 类别 id（和你训练代码一致）
        self.class_map = {"Mass": 0, "Calcification": 1, "Negative": 2}

        # pid → meta
        # meta 结构：
        # {
        #   "CC_pos_lesions": {lesion_id: [(path, cls), ...], ...}
        #   "MLO_pos_lesions": {...}
        #   "CC_neg": [(path, 2), ...]
        #   "MLO_neg": [(path, 2), ...]
        # }
        self.meta = {}

        # -------------------------------------------------
        # 1. 扫描所有 patch
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
                view = parsed["view"]
                patch_type = parsed["patch_type"]
                idx = parsed["idx"]

                if pid not in self.meta:
                    self.meta[pid] = {
                        "CC_pos_lesions": defaultdict(list),
                        "MLO_pos_lesions": defaultdict(list),
                        "CC_neg": [],
                        "MLO_neg": []
                    }

                fpath = os.path.join(cls_dir, fname)

                # 正样本: 来自 Mass / Calcification + pos
                is_pos = (cls_name != "Negative" and patch_type == "pos")
                # 负样本: Negative 文件夹 或 patch_type == 'neg'
                is_neg = (cls_name == "Negative" or patch_type == "neg")

                if is_pos:
                    lesion_id = idx // self.k_per_lesion  # K=5 → lesion 分组
                    cls_id = self.class_map[cls_name]

                    if view == "CC":
                        self.meta[pid]["CC_pos_lesions"][lesion_id].append(
                            (fpath, cls_id)
                        )
                    else:  # MLO
                        self.meta[pid]["MLO_pos_lesions"][lesion_id].append(
                            (fpath, cls_id)
                        )

                elif is_neg:
                    # 所有 negative patch 的 cls 统一用 2 (Negative)
                    if view == "CC":
                        self.meta[pid]["CC_neg"].append((fpath, 2))
                    else:
                        self.meta[pid]["MLO_neg"].append((fpath, 2))

        # -------------------------------------------------
        # 2. 只保留 “同时有 CC 正样本 & MLO 正样本” 的病人
        #    （和论文一样：matching 只对有 mass 的 pair 做正样本）
        # -------------------------------------------------
        self.valid_ids = []
        self.patient_info = {}

        for pid, v in self.meta.items():
            cc_lesions = set(v["CC_pos_lesions"].keys())
            mlo_lesions = set(v["MLO_pos_lesions"].keys())
            common_lesions = sorted(list(cc_lesions & mlo_lesions))

            if len(common_lesions) == 0:
                # 没有共同 lesion，就不用于 matching 训练
                continue

            self.valid_ids.append(pid)
            self.patient_info[pid] = {
                "CC_pos_lesions": v["CC_pos_lesions"],
                "MLO_pos_lesions": v["MLO_pos_lesions"],
                "CC_neg": v["CC_neg"],
                "MLO_neg": v["MLO_neg"],
                "common_lesions": common_lesions
            }

        print(f"✔ Loaded {len(self.valid_ids)} valid patient-sides with CC+MLO.")

        # -------------------------------------------------
        # 3. transform（训练：随机增强 / 验证：只 resize+norm）
        # -------------------------------------------------
        if self.random_flag:
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
    # 返回：1 个正 pair + 1 个负 pair
    # ======================================================
    def __getitem__(self, idx):
        pid = self.valid_ids[idx]
        info = self.patient_info[pid]

        cc_pos_lesions = info["CC_pos_lesions"]
        mlo_pos_lesions = info["MLO_pos_lesions"]
        cc_neg_list = info["CC_neg"]
        mlo_neg_list = info["MLO_neg"]
        common_lesions = info["common_lesions"]

        # -------------------------
        # ① POSITIVE pair (same lesion)
        # -------------------------
        lesion_id = random.choice(common_lesions)
        cc_pos_path, cc_pos_lbl = random.choice(cc_pos_lesions[lesion_id])
        mlo_pos_path, mlo_pos_lbl = random.choice(mlo_pos_lesions[lesion_id])
        match_pos = 1.0

        # -------------------------
        # ② NEGATIVE pair
        #    - 首选：一个 positive + 一个 negative
        #    - 兜底：两个不同 lesion 的 positive 也算不匹配
        # -------------------------
        use_cc_neg = len(cc_neg_list) > 0
        use_mlo_neg = len(mlo_neg_list) > 0

        if random.random() < 0.5 and use_cc_neg:
            # CC negative + MLO positive
            cc_neg_path, cc_neg_lbl = random.choice(cc_neg_list)
            # MLO positive 可以来自任意 lesion（包括同一个）
            lid = random.choice(common_lesions)
            mlo_neg_path, mlo_neg_lbl = random.choice(mlo_pos_lesions[lid])

        elif use_mlo_neg:
            # CC positive + MLO negative
            lid = random.choice(common_lesions)
            cc_neg_path, cc_neg_lbl = random.choice(cc_pos_lesions[lid])
            mlo_neg_path, mlo_neg_lbl = random.choice(mlo_neg_list)

        else:
            # 兜底：都没有 negative patch，就用两个不同 lesion 的 positive
            if len(common_lesions) > 1:
                lid1, lid2 = random.sample(common_lesions, 2)
            else:
                lid1 = lid2 = common_lesions[0]

            cc_neg_path, cc_neg_lbl = random.choice(cc_pos_lesions[lid1])
            mlo_neg_path, mlo_neg_lbl = random.choice(mlo_pos_lesions[lid2])

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

        # match_label: [1, 0]
        match_labels = torch.tensor([match_pos, match_neg], dtype=torch.float32)
        # classification labels: 0/1/2, 保持你的 num_classes=3 设定
        cc_cls = torch.tensor([cc_pos_lbl, cc_neg_lbl], dtype=torch.long)
        mlo_cls = torch.tensor([mlo_pos_lbl, mlo_neg_lbl], dtype=torch.long)

        return (
            torch.stack(cc_imgs),   # (2, C, H, W)
            torch.stack(mlo_imgs)   # (2, C, H, W)
        ), (
            match_labels,           # (2,)
            cc_cls,                 # (2,)
            mlo_cls                 # (2,)
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
