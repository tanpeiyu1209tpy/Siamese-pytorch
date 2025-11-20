# ==========================================================
# train_cmcnet.py — Final Version (Pair Training + Dual Validation)
# ==========================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.cmcnet import CMCNet
from utils.dataloader import (
    SiameseDataset, 
    siamese_collate,
    SingleImageDataset
)


# ------------------------------------------------------
# Contrastive Loss
# ------------------------------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=5.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance, label):
        """
        distance: (N,) L2 distance between embeddings
        label:    (N,) 1 = positive pair, 0 = negative pair
        """
        label = label.float()
        pos_loss = label * distance.pow(2)
        neg_loss = (1 - label) * torch.clamp(self.margin - distance, min=0).pow(2)
        return torch.mean(pos_loss + neg_loss)


# ------------------------------------------------------
# Train One Epoch (pair-based)
# ------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, contrastive_loss, ce_loss,
                    epoch, total_epoch, loss_weights):

    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epoch} [Train]")

    total_loss = 0.0

    for (cc, mlo), (match_label, cc_label, mlo_label) in pbar:

        cc, mlo = cc.to(device), mlo.to(device)
        match_label = match_label.to(device)
        cc_label = cc_label.to(device)
        mlo_label = mlo_label.to(device)

        optimizer.zero_grad()

        # forward
        dist, cc_logits, mlo_logits = model((cc, mlo))

        # multi-task loss
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

        pbar.set_postfix({
            "loss": total_loss / (pbar.n + 1),
            "match": loss_m.item(),
            "cls": (loss_cc.item() + loss_mlo.item()) / 2.0,
        })

    return total_loss / len(loader)


# ------------------------------------------------------
# Validation (pair-based matching)
# ------------------------------------------------------
def validate_matching(model, loader, device, contrastive_loss, margin=5):

    model.eval()
    total_loss = 0.0
    match_correct = 0
    total = 0

    threshold = margin / 2.0

    with torch.no_grad():
        for (cc, mlo), (match_label, _, _) in loader:

            cc, mlo = cc.to(device), mlo.to(device)
            match_label = match_label.to(device)

            dist, _, _ = model((cc, mlo))
            loss = contrastive_loss(dist, match_label)
            total_loss += loss.item()

            pred_match = (dist < threshold).long()
            match_correct += (pred_match == match_label).sum().item()
            total += cc.size(0)

    avg_loss = total_loss / len(loader)
    acc = match_correct / total if total > 0 else 0.0

    return avg_loss, acc


# ------------------------------------------------------
# Validation (single-image classification, CC & MLO 分开统计)
# ------------------------------------------------------
def validate_classification(model, loader, device):
    """
    loader: SingleImageDataset 的 DataLoader
            __getitem__ 返回 (img, label, view)
            view ∈ {"CC", "MLO"}
    """
    model.eval()

    cc_correct = 0
    cc_total = 0
    mlo_correct = 0
    mlo_total = 0

    with torch.no_grad():
        for imgs, labels, views in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 在一个 batch 里用 mask 拆出 CC / MLO
            cc_mask = [v == "CC" for v in views]
            mlo_mask = [v == "MLO" for v in views]

            # ----- CC branch -----
            if any(cc_mask):
                cc_imgs = imgs[cc_mask]
                cc_labels = labels[cc_mask]

                # CC 图像通过 CC 分支
                _, cc_logits, _ = model((cc_imgs, cc_imgs))
                preds = torch.argmax(cc_logits, dim=1)

                cc_correct += (preds == cc_labels).sum().item()
                cc_total += cc_labels.size(0)

            # ----- MLO branch -----
            if any(mlo_mask):
                mlo_imgs = imgs[mlo_mask]
                mlo_labels = labels[mlo_mask]

                # MLO 图像通过 MLO 分支
                _, _, mlo_logits = model((mlo_imgs, mlo_imgs))
                preds = torch.argmax(mlo_logits, dim=1)

                mlo_correct += (preds == mlo_labels).sum().item()
                mlo_total += mlo_labels.size(0)

    cc_acc = cc_correct / cc_total if cc_total > 0 else 0.0
    mlo_acc = mlo_correct / mlo_total if mlo_total > 0 else 0.0

    return cc_acc, mlo_acc


# ------------------------------------------------------
# Main Training Loop
# ------------------------------------------------------
if __name__ == "__main__":

    # 路径
    train_dir = "/kaggle/input/s-dataset/siamese/train_data_siamese"
    val_dir   = "/kaggle/input/s-dataset/siamese/val_data_siamese"

    input_size = (64, 64)
    num_classes = 3       # Mass / Calcification / Negative

    epochs = 50
    batch_size = 4
    lr = 1e-4
    margin = 5

    save_dir = "cmcnet_logs"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =======================
    # Load Datasets
    # =======================
    print("\n[INFO] Loading dataset...")

    train_dataset = SiameseDataset(train_dir, input_size, random_flag=True)
    val_pair_dataset = SiameseDataset(val_dir, input_size, random_flag=False)
    val_cls_dataset = SingleImageDataset(val_dir, input_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=siamese_collate, drop_last=True
    )
    val_pair_loader = DataLoader(
        val_pair_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=siamese_collate
    )
    val_cls_loader = DataLoader(
        val_cls_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4
    )

    # =======================
    # Model & Loss
    # =======================
    model = CMCNet(input_channels=3, num_classes=num_classes, pretrained=True)
    model.to(device)

    ce_loss = nn.CrossEntropyLoss()
    contrastive = ContrastiveLoss(margin)

    # loss 权重（三个任务）
    loss_weights = {
        "alpha": 1.0,   # CC 分类
        "beta":  1.0,   # MLO 分类
        "gamma": 1.0    # Matching
    }

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # =======================
    # Training
    # =======================
    best_val = 1e9

    for epoch in range(1, epochs + 1):

        # 1) 训练一轮（pair-based）
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            contrastive, ce_loss,
            epoch, epochs, loss_weights
        )

        # 2) 验证 matching（pair-based）
        val_match_loss, match_acc = validate_matching(
            model, val_pair_loader, device, contrastive
        )

        # 3) 验证分类（单图，CC/MLO 分开）
        cc_acc, mlo_acc = validate_classification(
            model, val_cls_loader, device
        )

        print("\n------------------------------")
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss      : {train_loss:.4f}")
        print(f"Match Val Loss  : {val_match_loss:.4f}")
        print(f"Match Accuracy  : {match_acc:.4f}")
        print(f"CC  Accuracy    : {cc_acc:.4f}")
        print(f"MLO Accuracy    : {mlo_acc:.4f}")
        print("------------------------------\n")

        # Save best (按 matching 的 val loss)
        if val_match_loss < best_val:
            best_val = val_match_loss
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_model.pth")
            )
            print("✔ Saved best_model.pth")

        # 每个 epoch 存一份
        torch.save(
            model.state_dict(),
            os.path.join(save_dir, f"epoch_{epoch}.pth")
        )
