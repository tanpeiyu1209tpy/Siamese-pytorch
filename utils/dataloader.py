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
            "patch_type": m.group(4),   # pos / neg
            "idx": int(m.group(5))
        }
    return None

# ===============================================================
# NEW CMCNet Dataset – FINAL FIX
# ===============================================================
class SiameseDatasetTrain(Dataset):
    def __init__(self, root_dir, input_size=(128,128), K=5):
        print("📌 Using SiameseDatasetTrain (YOLO-FP intra-patient matching)")

        self.K = K
        self.class_map = {
            "Mass": 0,
            "Suspicious_Calcification": 1,
            "Negative": 2   # YOLO FP
        }

        self.data = {}

        # -------- scan folders --------
        for cls in ["Mass", "Suspicious_Calcification", "Negative"]:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue

            for fname in os.listdir(cls_dir):
                info = parse_filename(fname)
                if info is None:
                    continue

                pid = f"{info['patient_id']}_{info['side']}"
                self.data.setdefault(pid, {
                    "CC_pos": [], "MLO_pos": [],
                    "CC_fp": [],  "MLO_fp": []
                })

                path = os.path.join(cls_dir, fname)

                if cls == "Negative":
                    if info["view"] == "CC":
                        self.data[pid]["CC_fp"].append((path, 2))
                    else:
                        self.data[pid]["MLO_fp"].append((path, 2))
                else:
                    lbl = self.class_map[cls]
                    if info["view"] == "CC":
                        self.data[pid]["CC_pos"].append((path, lbl))
                    else:
                        self.data[pid]["MLO_pos"].append((path, lbl))

        # -------- valid ids --------
        self.valid_ids = [
            pid for pid, v in self.data.items()
            if len(v["CC_pos"]) > 0 and len(v["MLO_pos"]) > 0
            and (len(v["CC_fp"]) > 0 or len(v["MLO_fp"]) > 0)
        ]

        print(f"✔ Loaded {len(self.valid_ids)} training patient-sides")

        self.tf = transforms.Compose([
            transforms.RandomRotation(25),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485]*3, [0.229]*3)
        ])

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        pid = self.valid_ids[idx]
        v = self.data[pid]

        cc_imgs, mlo_imgs = [], []
        match, cc_lbl, mlo_lbl = [], [], []

        # -------- Positive: lesion ↔ lesion --------
        for _ in range(self.K):
            cc_p, cc_l = random.choice(v["CC_pos"])
            mlo_p, mlo_l = random.choice(v["MLO_pos"])

            cc_imgs.append(self.tf(Image.open(cc_p).convert("RGB")))
            mlo_imgs.append(self.tf(Image.open(mlo_p).convert("RGB")))

            match.append(1.0)
            cc_lbl.append(cc_l)
            mlo_lbl.append(mlo_l)

        # -------- Negative: lesion ↔ YOLO FP --------
        for _ in range(self.K):
            if len(v["MLO_fp"]) > 0:
                cc_p, cc_l = random.choice(v["CC_pos"])
                mlo_p, mlo_l = random.choice(v["MLO_fp"])
            else:
                cc_p, cc_l = random.choice(v["CC_fp"])
                mlo_p, mlo_l = random.choice(v["MLO_pos"])

            cc_imgs.append(self.tf(Image.open(cc_p).convert("RGB")))
            mlo_imgs.append(self.tf(Image.open(mlo_p).convert("RGB")))

            match.append(0.0)
            cc_lbl.append(cc_l)
            mlo_lbl.append(mlo_l)

        return (
            torch.stack(cc_imgs),
            torch.stack(mlo_imgs)
        ), (
            torch.tensor(match),
            torch.tensor(cc_lbl),
            torch.tensor(mlo_lbl)
        )


class SiameseDatasetVal(Dataset):
    def __init__(self, root_dir, input_size=(128,128), K=5):
        print("📌 Using SiameseDatasetVal (fixed YOLO-FP matching)")
        self.K = K
        self.class_map = {"Mass":0,"Suspicious_Calcification":1,"Negative":2}
        self.data = {}

        for cls in ["Mass","Suspicious_Calcification","Negative"]:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue

            for fname in sorted(os.listdir(cls_dir)):
                info = parse_filename(fname)
                if info is None:
                    continue

                pid = f"{info['patient_id']}_{info['side']}"
                self.data.setdefault(pid,{
                    "CC_pos":[], "MLO_pos":[],
                    "CC_fp":[],  "MLO_fp":[]
                })

                path = os.path.join(cls_dir, fname)

                if cls=="Negative":
                    if info["view"]=="CC":
                        self.data[pid]["CC_fp"].append((path,2))
                    else:
                        self.data[pid]["MLO_fp"].append((path,2))
                else:
                    lbl=self.class_map[cls]
                    if info["view"]=="CC":
                        self.data[pid]["CC_pos"].append((path,lbl))
                    else:
                        self.data[pid]["MLO_pos"].append((path,lbl))

        self.valid_ids=[
            pid for pid,v in self.data.items()
            if len(v["CC_pos"])>=K and len(v["MLO_pos"])>=K
            and (len(v["CC_fp"])>0 or len(v["MLO_fp"])>0)
        ]

        print(f"✔ Loaded {len(self.valid_ids)} validation patient-sides")

        self.tf = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485]*3,[0.229]*3)
        ])

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        pid=self.valid_ids[idx]
        v=self.data[pid]

        cc_imgs,mlo_imgs=[],[]
        match,cc_lbl,mlo_lbl=[],[],[]

        for i in range(self.K):
            cc_p,cc_l=v["CC_pos"][i]
            mlo_p,mlo_l=v["MLO_pos"][i]

            cc_imgs.append(self.tf(Image.open(cc_p).convert("RGB")))
            mlo_imgs.append(self.tf(Image.open(mlo_p).convert("RGB")))
            match.append(1.0)
            cc_lbl.append(cc_l)
            mlo_lbl.append(mlo_l)

        for i in range(self.K):
            if len(v["MLO_fp"])>0:
                cc_p,cc_l=v["CC_pos"][i]
                mlo_p,mlo_l=v["MLO_fp"][i%len(v["MLO_fp"])]
            else:
                cc_p,cc_l=v["CC_fp"][i%len(v["CC_fp"])]
                mlo_p,mlo_l=v["MLO_pos"][i]

            cc_imgs.append(self.tf(Image.open(cc_p).convert("RGB")))
            mlo_imgs.append(self.tf(Image.open(mlo_p).convert("RGB")))
            match.append(0.0)
            cc_lbl.append(cc_l)
            mlo_lbl.append(mlo_l)

        return (
            torch.stack(cc_imgs),
            torch.stack(mlo_imgs)
        ), (
            torch.tensor(match),
            torch.tensor(cc_lbl),
            torch.tensor(mlo_lbl)
        )


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
'''
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
# SiameseDatasetTrain
# ===============================================================
class SiameseDatasetTrain(Dataset):
    def __init__(self, root_dir, input_size=(128, 128), K=5):
        print("📌 Using SiameseDatasetTrain (cross-pid matching)")

        self.root_dir = root_dir
        self.input_size = input_size
        self.K = K

        self.class_map = {"Mass": 0, "Suspicious_Calcification": 1, "Negative": 2}
        self.data = {}

        # ------------------------------
        # Scan folders
        # ------------------------------
        for cls_name in ["Mass", "Suspicious_Calcification", "Negative"]:
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

                if parsed["view"] == "CC":
                    if cls_name == "Negative":
                        self.data[pid]["CC_neg"].append((fpath, 2))
                    else:
                        self.data[pid]["CC_pos"].append((fpath, self.class_map[cls_name]))
                else:
                    if cls_name == "Negative":
                        self.data[pid]["MLO_neg"].append((fpath, 2))
                    else:
                        self.data[pid]["MLO_pos"].append((fpath, self.class_map[cls_name]))

        # ------------------------------
        # valid lesion pids (Mass / Calc only)
        # ------------------------------
        self.valid_ids = [
            pid for pid, v in self.data.items()
            if len(v["CC_pos"]) > 0 and len(v["MLO_pos"]) > 0
        ]

        # ------------------------------
        # build lesion pools
        # ------------------------------
        self.mass_pids = []
        self.calc_pids = []

        for pid in self.valid_ids:
            lbl = self.data[pid]["CC_pos"][0][1]
            if lbl == self.class_map["Mass"]:
                self.mass_pids.append(pid)
            elif lbl == self.class_map["Suspicious_Calcification"]:
                self.calc_pids.append(pid)

        print(f"✔ Loaded {len(self.valid_ids)} training patient-sides")
        print("Mass pids:", len(self.mass_pids))
        print("Calc pids:", len(self.calc_pids))

        # augmentation (train only)
        self.to_tensor = transforms.Compose([
            transforms.RandomRotation(25),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        pid = self.valid_ids[idx]
        v = self.data[pid]

        cc_imgs, mlo_imgs = [], []
        match_labels, cc_labels, mlo_labels = [], [], []

        # -------------------------
        # Positive matching (same pid)
        # -------------------------
        for _ in range(self.K):
            cc_path, cc_lbl = random.choice(v["CC_pos"])
            mlo_path, mlo_lbl = random.choice(v["MLO_pos"])

            cc_imgs.append(self.load_image(cc_path))
            mlo_imgs.append(self.load_image(mlo_path))
            match_labels.append(1.0)
            cc_labels.append(cc_lbl)
            mlo_labels.append(mlo_lbl)

        # -------------------------
        # Negative matching (different pid, same lesion type)
        # -------------------------
        pid_label = v["CC_pos"][0][1]

        if pid_label == self.class_map["Mass"]:
            neg_pool = [p for p in self.mass_pids if p != pid]
        else:
            neg_pool = [p for p in self.calc_pids if p != pid]

        for _ in range(self.K):
            neg_pid = random.choice(neg_pool)
            neg_v = self.data[neg_pid]

            cc_path, cc_lbl = random.choice(v["CC_pos"])
            mlo_path, mlo_lbl = random.choice(neg_v["MLO_pos"])

            cc_imgs.append(self.load_image(cc_path))
            mlo_imgs.append(self.load_image(mlo_path))
            match_labels.append(0.0)
            cc_labels.append(cc_lbl)
            mlo_labels.append(mlo_lbl)

        return (
            torch.stack(cc_imgs),
            torch.stack(mlo_imgs)
        ), (
            torch.tensor(match_labels).float(),
            torch.tensor(cc_labels).long(),
            torch.tensor(mlo_labels).long()
        )

    def load_image(self, path):
        return self.to_tensor(Image.open(path).convert("RGB"))


# ===============================================================
# SiameseDatasetVal (fixed pairs)
# ===============================================================
class SiameseDatasetVal(Dataset):
    def __init__(self, root_dir, input_size=(128, 128), K=5):
        print("📌 Using SiameseDatasetVal (fixed cross-pid pairs)")

        self.root_dir = root_dir
        self.input_size = input_size
        self.K = K

        self.class_map = {"Mass": 0, "Suspicious_Calcification": 1, "Negative": 2}
        self.data = {}

        # scan folders
        for cls_name in ["Mass", "Suspicious_Calcification", "Negative"]:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue

            for fname in sorted(os.listdir(cls_dir)):
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

                if parsed["view"] == "CC":
                    if cls_name == "Negative":
                        self.data[pid]["CC_neg"].append((fpath, 2))
                    else:
                        self.data[pid]["CC_pos"].append((fpath, self.class_map[cls_name]))
                else:
                    if cls_name == "Negative":
                        self.data[pid]["MLO_neg"].append((fpath, 2))
                    else:
                        self.data[pid]["MLO_pos"].append((fpath, self.class_map[cls_name]))

        self.valid_ids = [
            pid for pid, v in self.data.items()
            if len(v["CC_pos"]) >= K and len(v["MLO_pos"]) >= K
        ]

        self.mass_pids, self.calc_pids = [], []
        for pid in self.valid_ids:
            lbl = self.data[pid]["CC_pos"][0][1]
            if lbl == self.class_map["Mass"]:
                self.mass_pids.append(pid)
            elif lbl == self.class_map["Suspicious_Calcification"]:
                self.calc_pids.append(pid)

        print(f"✔ Loaded {len(self.valid_ids)} validation patient-sides")
        print("Mass pids:", len(self.mass_pids))
        print("Calc pids:", len(self.calc_pids))


        self.to_tensor = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        pid = self.valid_ids[idx]
        v = self.data[pid]

        cc_imgs, mlo_imgs = [], []
        match_labels, cc_labels, mlo_labels = [], [], []

        # positive
        for i in range(self.K):
            cc_path, cc_lbl = v["CC_pos"][i]
            mlo_path, mlo_lbl = v["MLO_pos"][i]

            cc_imgs.append(self.load_image(cc_path))
            mlo_imgs.append(self.load_image(mlo_path))
            match_labels.append(1.0)
            cc_labels.append(cc_lbl)
            mlo_labels.append(mlo_lbl)

        # negative (fixed cross pid)
        pid_label = v["CC_pos"][0][1]
        pool = self.mass_pids if pid_label == self.class_map["Mass"] else self.calc_pids
        neg_pid = pool[(idx + 1) % len(pool)]
        neg_v = self.data[neg_pid]

        for i in range(self.K):
            cc_path, cc_lbl = v["CC_pos"][i]
            mlo_path, mlo_lbl = neg_v["MLO_pos"][i]

            cc_imgs.append(self.load_image(cc_path))
            mlo_imgs.append(self.load_image(mlo_path))
            match_labels.append(0.0)
            cc_labels.append(cc_lbl)
            mlo_labels.append(mlo_lbl)

        return (
            torch.stack(cc_imgs),
            torch.stack(mlo_imgs)
        ), (
            torch.tensor(match_labels).float(),
            torch.tensor(cc_labels).long(),
            torch.tensor(mlo_labels).long()
        )

    def load_image(self, path):
        return self.to_tensor(Image.open(path).convert("RGB"))


# ===============================================================
# Shared collate function (train & val)
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
'''
