# ==========================================================
# train_cmcnet.py  — Full Training Pipeline (Final Version)
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
# Train One Epoch
# ------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, contrastive_loss, ce_loss,
                    epoch, total_epoch, loss_weights):

    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epoch} [Train]")

    total_loss = 0
    match_loss_sum = 0
    cls_loss_sum = 0

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
        cls_loss_sum += (loss_cc.item() + loss_mlo.item()) / 2

        pbar.set_postfix({
            "loss": total_loss / (pbar.n + 1),
            "match": match_loss_sum / (pbar.n + 1),
            "cls": cls_loss_sum / (pbar.n + 1)
        })

    return total_loss / len(loader)


# ------------------------------------------------------
# Validate One Epoch
# ------------------------------------------------------
def validate_one_epoch(model, loader, device, contrastive_loss, ce_loss,
                       epoch, total_epoch, margin=5):

    model.eval()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epoch} [Val]")

    total_loss = 0
    match_correct = 0
    cc_correct = 0
    mlo_correct = 0
    total_samples = 0

    threshold = margin / 2

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
            loss = loss_m + loss_cc + loss_mlo

            total_loss += loss.item()

            # match accuracy
            pred_match = (dist < threshold).long()
            match_correct += (pred_match == match_label).sum().item()

            # classification accuracy
            cc_correct += (torch.argmax(cc_logits, 1) == cc_label).sum().item()
            mlo_correct += (torch.argmax(mlo_logits, 1) == mlo_label).sum().item()

            total_samples += cc.size(0)

            pbar.set_postfix({"val_loss": total_loss / (pbar.n + 1)})

    match_acc = match_correct / total_samples
    cc_acc = cc_correct / total_samples
    mlo_acc = mlo_correct / total_samples

    return total_loss / len(loader), match_acc, cc_acc, mlo_acc


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
    batch_size = 32
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
    train_dataset = SiameseDataset(train_dir, input_size)
    val_dataset   = SiameseDataset(val_dir, input_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=siamese_collate, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=siamese_collate, drop_last=False
    )

    # ==================================================
    # D) Model
    # ==================================================
    model = CMCNet(input_channels=3, num_classes=num_classes, pretrained=True)
    model.to(device)

    # ==================================================
    # E) Balanced Loss (class imbalance fix)
    # ==================================================
    pos_count = train_dataset.total_positive
    neg_count = train_dataset.total_negative
    ratio = neg_count / max(pos_count, 1)

    pos_weight = max(1.0, ratio)

    print(f"\n[INFO] Positive Samples: {pos_count}")
    print(f"[INFO] Negative Samples: {neg_count}")
    print(f"[INFO] Class Weight (pos): {pos_weight:.2f}")

    ce_loss = nn.CrossEntropyLoss(
        weight=torch.tensor([pos_weight, pos_weight, 1.0], device=device)
    )

    # ==================================================
    # F) Optimizer & weighted multi-loss
    # ==================================================
    contrastive_loss = ContrastiveLoss(margin=margin)

    loss_weights = {
        "alpha": 1.0,   # CC classification
        "beta":  1.0,   # MLO classification
        "gamma": 1.0    # Matching (contrastive)
    }

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ==================================================
    # Training Loop
    # ==================================================
    best_val_loss = 1e9

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            contrastive_loss, ce_loss,
            epoch, epochs, loss_weights
        )

        val_loss, match_acc, cc_acc, mlo_acc = validate_one_epoch(
            model, val_loader, device,
            contrastive_loss, ce_loss,
            epoch, epochs, margin
        )

        print("\n------------------------------")
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Match Acc  : {match_acc:.4f}")
        print(f"CC Acc     : {cc_acc:.4f}")
        print(f"MLO Acc    : {mlo_acc:.4f}")
        print("------------------------------\n")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "best_model.pth"))
            print("✔ Saved best_model.pth")

        # save every epoch
        torch.save(model.state_dict(),
                   os.path.join(save_dir, f"epoch_{epoch}.pth"))
