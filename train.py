# ==========================================================
# train_cmcnet.py — Compatible with your new dataloader.py
# (Joint Training: Matching + CC Classification + MLO Classification)
# ==========================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.cmcnet import CMCNet
from utils.dataloader import SiameseDataset, siamese_collate


# ------------------------------------------------------
# Contrastive Loss
# ------------------------------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.5):
        super().__init__()
        self.margin = margin

    def forward(self, dist, label):
        label = label.float()
        pos_loss = label * dist.pow(2)                    # match = 1
        neg_loss = (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return torch.mean(pos_loss + neg_loss)


# ------------------------------------------------------
# Train One Epoch
# ------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device,
                    contrastive_loss, ce_loss, weights,
                    epoch, total_epoch):

    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epoch} [Train]")
    total_loss = 0

    for (cc, mlo), (match_label, cc_label, mlo_label) in pbar:

        cc, mlo = cc.to(device), mlo.to(device)
        match_label = match_label.to(device)
        cc_label = cc_label.to(device)
        mlo_label = mlo_label.to(device)

        optimizer.zero_grad()

        # Forward
        dist, cc_logits, mlo_logits = model((cc, mlo))

        loss_m = contrastive_loss(dist, match_label)
        loss_cc = ce_loss(cc_logits, cc_label)
        loss_mlo = ce_loss(mlo_logits, mlo_label)

        loss = (weights["gamma"] * loss_m +
                weights["alpha"] * loss_cc +
                weights["beta"] * loss_mlo)

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
# Validation — Full Multi-task (matching + 2-way classification)
# ------------------------------------------------------
def validate_joint(model, loader, device,
                   contrastive_loss, ce_loss, weights,
                   margin=5):

    model.eval()

    total_loss = 0
    match_correct = 0
    total_samples = 0

    cc_correct = 0
    mlo_correct = 0

    threshold = margin / 2.0

    with torch.no_grad():
        for (cc, mlo), (match_label, cc_label, mlo_label) in loader:

            cc, mlo = cc.to(device), mlo.to(device)
            match_label = match_label.to(device)
            cc_label = cc_label.to(device)
            mlo_label = mlo_label.to(device)

            dist, cc_logits, mlo_logits = model((cc, mlo))

            loss_m = contrastive_loss(dist, match_label)
            loss_cc = ce_loss(cc_logits, cc_label)
            loss_mlo = ce_loss(mlo_logits, mlo_label)

            loss = (weights["gamma"] * loss_m +
                    weights["alpha"] * loss_cc +
                    weights["beta"] * loss_mlo)
            total_loss += loss.item()

            # Accuracy: Matching
            pred_match = (dist < threshold).long()
            match_correct += (pred_match == match_label).sum().item()

            # Accuracy: CC classification
            cc_pred = torch.argmax(cc_logits, dim=1)
            mlo_pred = torch.argmax(mlo_logits, dim=1)

            cc_correct += (cc_pred == cc_label).sum().item()
            mlo_correct += (mlo_pred == mlo_label).sum().item()

            total_samples += cc.size(0)

    return (
        total_loss / len(loader),
        match_correct / total_samples,
        cc_correct / total_samples,
        mlo_correct / total_samples
    )


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

    print("\n[INFO] Loading dataset...")

    # Use your dataloader exactly as you wrote it
    train_dataset = SiameseDataset(train_dir, input_size, random_flag=True)
    val_dataset   = SiameseDataset(val_dir, input_size, random_flag=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=siamese_collate,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=siamese_collate
    )

    # Model
    model = CMCNet(input_channels=3, num_classes=num_classes, pretrained=True)
    model.to(device)

    ce_loss = nn.CrossEntropyLoss()
    contrastive = ContrastiveLoss(margin)

    weights = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = 1e9

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            contrastive, ce_loss, weights,
            epoch, epochs
        )

        val_loss, match_acc, cc_acc, mlo_acc = validate_joint(
            model, val_loader, device,
            contrastive, ce_loss, weights
        )

        print("\n------------------------------")
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss      : {train_loss:.4f}")
        print(f"Val Total Loss  : {val_loss:.4f}")
        print(f"Match Accuracy  : {match_acc:.4f}")
        print(f"CC Accuracy     : {cc_acc:.4f}")
        print(f"MLO Accuracy    : {mlo_acc:.4f}")
        print("------------------------------\n")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("✔ Saved best_model.pth")

        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pth"))
