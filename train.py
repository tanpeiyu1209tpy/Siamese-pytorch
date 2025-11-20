# ==========================================================
# train_cmcnet.py  — Full Training Pipeline (Revised)
# ==========================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from PIL import Image
from torchvision import transforms

from nets.cmcnet import CMCNet
from utils.dataloader import SiameseDataset, siamese_collate


# ------------------------------------------------------
# Contrastive Loss (论文一致)
# ------------------------------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=5.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance, label):
        label = label.float()
        pos_loss = label * distance.pow(2)
        neg_loss = (1 - label) * torch.clamp(self.margin - distance, min=0).pow(2)
        return torch.mean(pos_loss + neg_loss)


# ------------------------------------------------------
# 单图分类用 Dataset（验证用）
#   root_dir/
#       Mass/*.png
#       Calcification/*.png
#       Negative/*.png
# ------------------------------------------------------
class SingleImageDataset(Dataset):
    def __init__(self, root_dir, input_size=(64, 64)):
        self.paths = []
        self.input_size = input_size
        self.class_map = {
            "Mass": 0,
            "Calcification": 1,
            "Negative": 2
        }

        for cls_name in ["Mass", "Calcification", "Negative"]:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            cls_id = self.class_map[cls_name]
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                fpath = os.path.join(cls_dir, fname)
                self.paths.append((fpath, cls_id))

        if len(self.paths) == 0:
            raise ValueError(f"❌ SingleImageDataset: no images found under {root_dir}")

        self.to_tensor = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        print(f"[INFO] SingleImageDataset loaded {len(self.paths)} images from {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.to_tensor(img)
        return img, label


# ------------------------------------------------------
# Train One Epoch (pair-based, multi-task)
# ------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, contrastive_loss, ce_loss,
                    epoch, total_epoch, loss_weights):

    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epoch} [Train]")

    total_loss = 0.0
    match_loss_sum = 0.0
    cls_loss_sum = 0.0

    for (cc, mlo), (match_label, cc_label, mlo_label) in pbar:
        cc, mlo = cc.to(device), mlo.to(device)
        match_label = match_label.to(device)
        cc_label = cc_label.to(device)
        mlo_label = mlo_label.to(device)

        optimizer.zero_grad()

        dist, cc_logits, mlo_logits = model((cc, mlo))

        loss_m   = contrastive_loss(dist, match_label)
        loss_cc  = ce_loss(cc_logits, cc_label)
        loss_mlo = ce_loss(mlo_logits, mlo_label)

        loss = (
            loss_weights["gamma"] * loss_m +
            loss_weights["alpha"] * loss_cc +
            loss_weights["beta"]  * loss_mlo
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        match_loss_sum += loss_m.item()
        cls_loss_sum += (loss_cc.item() + loss_mlo.item()) / 2.0

        pbar.set_postfix({
            "loss":   total_loss / (pbar.n + 1),
            "match":  match_loss_sum / (pbar.n + 1),
            "cls":    cls_loss_sum / (pbar.n + 1),
        })

    return total_loss / len(loader)


# ------------------------------------------------------
# Validate One Epoch (pair-based，只看 match + 总 loss)
#   分类的 acc 我们用 SingleImageDataset 单独测
# ------------------------------------------------------
def validate_pairs(model, loader, device, contrastive_loss, ce_loss,
                   epoch, total_epoch, loss_weights, margin=5):

    model.eval()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epoch} [Val-Pairs]")

    total_loss = 0.0
    match_correct = 0
    total_samples = 0

    threshold = margin / 2.0

    with torch.no_grad():
        for (cc, mlo), (match_label, cc_label, mlo_label) in pbar:
            cc, mlo = cc.to(device), mlo.to(device)
            match_label = match_label.to(device)
            cc_label = cc_label.to(device)
            mlo_label = mlo_label.to(device)

            dist, cc_logits, mlo_logits = model((cc, mlo))

            loss_m   = contrastive_loss(dist, match_label)
            loss_cc  = ce_loss(cc_logits, cc_label)
            loss_mlo = ce_loss(mlo_logits, mlo_label)
            loss = (
                loss_weights["gamma"] * loss_m +
                loss_weights["alpha"] * loss_cc +
                loss_weights["beta"]  * loss_mlo
            )

            total_loss += loss.item()

            # match accuracy（用 distance）
            pred_match = (dist < threshold).long()
            match_correct += (pred_match == match_label).sum().item()
            total_samples += cc.size(0)

            pbar.set_postfix({
                "val_loss": total_loss / (pbar.n + 1),
            })

    match_acc = match_correct / max(total_samples, 1)

    return total_loss / len(loader), match_acc


# ------------------------------------------------------
# Validate classification（单图，不用 pair）
#   用 SingleImageDataset，分别看 CC_HEAD 和 MLO_HEAD 的 acc
# ------------------------------------------------------
def validate_classification(model, loader, device):
    model.eval()

    cc_correct = 0
    mlo_correct = 0
    total = 0

    pbar = tqdm(loader, desc="[Val-Cls]")

    with torch.no_grad():
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 我们把同一张 img 送入 CC/MLO 两个分支
            # 只是为了评估两个 head 的分类效果
            _, logits_cc, logits_mlo = model((imgs, imgs))

            cc_pred = torch.argmax(logits_cc, dim=1)
            mlo_pred = torch.argmax(logits_mlo, dim=1)

            cc_correct += (cc_pred == labels).sum().item()
            mlo_correct += (mlo_pred == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "CC_acc": cc_correct / max(total, 1),
                "MLO_acc": mlo_correct / max(total, 1)
            })

    cc_acc = cc_correct / max(total, 1)
    mlo_acc = mlo_correct / max(total, 1)
    return cc_acc, mlo_acc


# ------------------------------------------------------
# Main
# ------------------------------------------------------
if __name__ == "__main__":

    # ==================================================
    # A) 路径
    # ==================================================
    train_dir = "/kaggle/input/s-dataset/siamese/train_data_siamese"
    val_dir   = "/kaggle/input/s-dataset/siamese/val_data_siamese"

    input_size = (64, 64)
    num_classes = 3  # Mass / Calc / Negative

    epochs = 50
    batch_size = 4
    lr = 1e-4
    margin = 5

    save_dir = "cmcnet_logs"
    os.makedirs(save_dir, exist_ok=True)

    # ==================================================
    # B) Device
    # ==================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================================================
    # C) Dataset & Loader
    # ==================================================
    print("\n[INFO] Loading dataset...")
    # 训练 & pair-based 验证
    train_dataset = SiameseDataset(train_dir, input_size, random_flag=True)
    val_pair_dataset = SiameseDataset(val_dir, input_size, random_flag=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=siamese_collate, drop_last=True
    )
    val_pair_loader = DataLoader(
        val_pair_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=siamese_collate, drop_last=False
    )

    # 单图分类验证
    val_cls_dataset = SingleImageDataset(val_dir, input_size)
    val_cls_loader = DataLoader(
        val_cls_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, drop_last=False
    )

    # ==================================================
    # D) Model
    # ==================================================
    model = CMCNet(input_channels=3, num_classes=num_classes, pretrained=True)
    model.to(device)

    # ==================================================
    # E) Loss
    # ==================================================
    ce_loss = nn.CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss(margin=margin)

    # loss 权重（分类+匹配）
    loss_weights = {
        "alpha": 1.0,   # CC classification
        "beta":  1.0,   # MLO classification
        "gamma": 1.0    # Matching (contrastive)
    }

    # ==================================================
    # F) Optimizer
    # ==================================================
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ==================================================
    # Training Loop
    # ==================================================
    best_val_loss = 1e9

    for epoch in range(1, epochs + 1):

        # ---- Train (pairs) ----
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            contrastive_loss, ce_loss,
            epoch, epochs, loss_weights
        )

        # ---- Val: pair-based (match + multitask loss) ----
        val_loss, match_acc = validate_pairs(
            model, val_pair_loader, device,
            contrastive_loss, ce_loss,
            epoch, epochs, loss_weights, margin
        )

        # ---- Val: classification on single images ----
        cc_acc, mlo_acc = validate_classification(
            model, val_cls_loader, device
        )

        print("\n------------------------------")
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Match Acc  : {match_acc:.4f}")
        print(f"CC Acc     : {cc_acc:.4f}")
        print(f"MLO Acc    : {mlo_acc:.4f}")
        print("------------------------------\n")

        # Save Best Model（按 val_loss）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "best_model.pth"))
            print("✔ Saved best_model.pth")

        # 每个 epoch 单独存一份
        torch.save(model.state_dict(),
                   os.path.join(save_dir, f"epoch_{epoch}.pth"))
