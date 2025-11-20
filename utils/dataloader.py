import os
import random
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ------------------------------------------------
#  解析你当前命名格式的函数
#  e.g. 0037abcd_L_CC_pos0.png
# ------------------------------------------------
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


# ------------------------------------------------
#  Siamese Dataset（最终版）
# ------------------------------------------------
class SiameseDataset(Dataset):
    def __init__(self, root_dir, input_size=(64,64), random_flag=True):
        """
        root_dir/
            Mass/
            Calcification/
            Negative/

        每个目录里面都是形如：
            patient_side_view_posK.png
        """
        self.root_dir = root_dir
        self.random_flag = random_flag
        self.input_size = input_size

        # label 映射 Mass=0, Calcification=1, Negative=2
        self.class_map = {
            "Mass": 0,
            "Calcification": 1,
            "Negative": 2
        }

        # ------------------------------------------------
        # 1. 读取所有图像 → 解析命名 → 按 patient_side 分组
        # ------------------------------------------------
        self.data = {}  # key = patient_side, value = {CC:[], MLO:[]}

        for cls_name in ["Mass", "Calcification", "Negative"]:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue

            for fname in os.listdir(cls_dir):
                parsed = parse_filename(fname)
                if parsed is None:
                    continue

                patient_side = f"{parsed['patient_id']}_{parsed['side']}"

                if patient_side not in self.data:
                    self.data[patient_side] = {"CC": [], "MLO": []}

                fullpath = os.path.join(cls_dir, fname)
                self.data[patient_side][parsed["view"]].append({
                    "path": fullpath,
                    "cls": self.class_map[cls_name],
                    "patch_type": parsed["patch_type"]
                })

        # 过滤掉没有 CC 或 MLO 的
        self.valid_ids = [
            p for p in self.data.keys()
            if len(self.data[p]["CC"]) > 0 and len(self.data[p]["MLO"]) > 0
        ]

        if len(self.valid_ids) == 0:
            raise ValueError("❌ No valid CC-MLO pairs found!")

        print(f"Loaded {len(self.valid_ids)} valid patient-sides with CC+MLO.")

        # transforms
        self.to_tensor = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        # 1. 选一个 patient_side
        pid = self.valid_ids[idx]

        cc_list = self.data[pid]["CC"]
        mlo_list = self.data[pid]["MLO"]

        # ------------------------------------------------
        # 2. 构建 positive pair
        # ------------------------------------------------
        cc_pos = random.choice(cc_list)
        mlo_pos = random.choice(mlo_list)

        match_pos = 1.0  # label for pair
        cc_pos_cls = cc_pos["cls"]
        mlo_pos_cls = mlo_pos["cls"]

        # ------------------------------------------------
        # 3. 构建 negative pair
        #   - 使用当前 patient 的 CC
        #   - 使用随机另一病人的 MLO
        # ------------------------------------------------
        anchor = cc_pos
        neg_pid = random.choice(self.valid_ids)
        while neg_pid == pid:
            neg_pid = random.choice(self.valid_ids)

        mlo_neg = random.choice(self.data[neg_pid]["MLO"])
        match_neg = 0.0
        anchor_cls = anchor["cls"]
        mlo_neg_cls = mlo_neg["cls"]  # negative class maybe 2

        # ------------------------------------------------
        # 4. 读取图像
        # ------------------------------------------------
        cc_imgs = [
            self.load_image(cc_pos["path"]),
            self.load_image(anchor["path"])
        ]
        mlo_imgs = [
            self.load_image(mlo_pos["path"]),
            self.load_image(mlo_neg["path"])
        ]

        # labels
        match_labels = torch.tensor([match_pos, match_neg], dtype=torch.float32)
        cc_cls_labels = torch.tensor([cc_pos_cls, anchor_cls], dtype=torch.long)
        mlo_cls_labels = torch.tensor([mlo_pos_cls, mlo_neg_cls], dtype=torch.long)

        return (
            torch.stack(cc_imgs),        # shape (2, 3, H, W)
            torch.stack(mlo_imgs)        # shape (2, 3, H, W)
        ), (
            match_labels,
            cc_cls_labels,
            mlo_cls_labels
        )

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.to_tensor(img)


# ------------------------------------------------
# Collate for DataLoader
# ------------------------------------------------
def siamese_collate(batch):
    cc_imgs = []
    mlo_imgs = []
    match_labels = []
    cc_cls_labels = []
    mlo_cls_labels = []

    for (cc,mlo), (m,cc_c,mlo_c) in batch:
        cc_imgs.append(cc)
        mlo_imgs.append(mlo)
        match_labels.append(m)
        cc_cls_labels.append(cc_c)
        mlo_cls_labels.append(mlo_c)

    # concat on batch dimension
    cc_imgs = torch.cat(cc_imgs, dim=0)     # (B*2, 3, H, W)
    mlo_imgs = torch.cat(mlo_imgs, dim=0)
    match_labels = torch.cat(match_labels, dim=0)
    cc_cls_labels = torch.cat(cc_cls_labels, dim=0)
    mlo_cls_labels = torch.cat(mlo_cls_labels, dim=0)

    return (cc_imgs, mlo_imgs), (match_labels, cc_cls_labels, mlo_cls_labels)
