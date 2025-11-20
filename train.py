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
    SingleImageDataset      # <── 新增
)

# ------------------------------------------------------
# Contrastive Loss
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
# Train One Epoch (pair-based)
# ------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, contrastive_loss, ce_loss,
                    epoch, total_epoch, loss_weights):

    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epoch} [Train]")

    total_loss = 0
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
        pbar.set_postfix({
            "loss": total_loss / (pbar.n + 1),
            "match": loss_m.item(),
            "cls": (loss_cc.item() + loss_mlo.item()) / 2
        })

    return total_loss / len(loader)


# ------------------------------------------------------
# Validation (pair-based)
# ------------------------------------------------------
def validate_matching(model, loader, device, contrastive_loss, margin=5):

    model.eval()
    total_loss = 0
    match_correct = 0
    total = 0

    threshold = margin / 2

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

    return total_loss / len(loader), match_correct / total


# ------------------------------------------------------
# Validation (classification-based, single image)
# ------------------------------------------------------
def validate_classification(model, loader, device):

    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 只用 CC 分支评估分类
            _, cc_logits, _ = model((imgs, imgs))  
            preds = torch.argmax(cc_logits, dim=1)

            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return correct / total



# ------------------------------------------------------
# Main Training Loop
# ------------------------------------------------------
if __name__ == "__main__":

    train_dir = "/kaggle/input/s-dataset/siamese/train_data_siamese"
    val_dir   = "/kaggle/input/s-dataset/siamese/val_data_siamese"

    input_size = (64, 64)
    num_classes = 3
    epochs = 50
    batch_size = 4
    lr = 1e-4
    margin = 5

    save_dir = "cmcnet_logs"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =======================
    # Load Dataset
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

    loss_weights = {"alpha":1.0,"beta":1.0,"gamma":1.0}

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # =======================
    # Training
    # =======================
    best_val = 1e9

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            contrastive, ce_loss,
            epoch, epochs, loss_weights
        )

        val_match_loss, match_acc = validate_matching(
            model, val_pair_loader, device, contrastive
        )

        cls_acc = validate_classification(
            model, val_cls_loader, device
        )

        print("\n------------------------------")
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss       : {train_loss:.4f}")
        print(f"Match Val Loss   : {val_match_loss:.4f}")
        print(f"Match Accuracy   : {match_acc:.4f}")
        print(f"Class Accuracy   : {cls_acc:.4f}")
        print("------------------------------\n")

        # Save best
        if val_match_loss < best_val:
            best_val = val_match_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("✔ Saved best_model.pth")

        torch.save(model.state_dict(),
                   os.path.join(save_dir, f"epoch_{epoch}.pth"))
