# ==========================================================
# train_cmcnet.py — Final Clean Version (Correct CMCNet Training)
# Multi-task: Matching + CC Classification + MLO Classification
# ==========================================================
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from nets.cmcnet import CMCNet
from utils.dataloader import SiameseDatasetTrain, siamese_collate,SiameseDatasetVal
from torch.optim.lr_scheduler import StepLR


# ------------------------------------------------------
# Contrastive Loss (Correct Version)
# ------------------------------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=5.0):
        super().__init__()
        self.margin = margin

    def forward(self, dist, label):
        label = label.float()
        pos_loss = label * dist.pow(2)
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

        # Forward pass
        dist, cc_logits, mlo_logits = model((cc, mlo))

        # Losses
        loss_m = contrastive_loss(dist, match_label)
        loss_cc = ce_loss(cc_logits, cc_label)
        loss_mlo = ce_loss(mlo_logits, mlo_label)

        loss = (weights["gamma"] * loss_m +
                weights["alpha"] * loss_cc +
                weights["beta"] * loss_mlo)

        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({
            "loss": total_loss / (pbar.n + 1),
            "match": loss_m.item(),
            "cls": (loss_cc.item() + loss_mlo.item()) / 2
        })

    return total_loss / len(loader)


# ------------------------------------------------------
# Validation Function
# ------------------------------------------------------
def validate_joint(model, loader, device,
                   contrastive_loss, ce_loss, weights,
                   margin):

    model.eval()

    total_loss = 0
    total_match_correct = 0
    total_cc_correct = 0
    total_mlo_correct = 0
    total_pairs = 0

    threshold = margin / 2.0

    with torch.no_grad():

        for (cc_batch, mlo_batch), (match_label, cc_label, mlo_label) in loader:

            # cc_batch shape: [B*5, C, H, W]  ← 每个 patient 五个正负对
            cc_batch = cc_batch.to(device)
            mlo_batch = mlo_batch.to(device)
            match_label = match_label.to(device)
            cc_label = cc_label.to(device)
            mlo_label = mlo_label.to(device)

            # forward all 10 samples (5 pos, 5 neg)
            dist, cc_logits, mlo_logits = model((cc_batch, mlo_batch))

            # compute loss
            loss_m = contrastive_loss(dist, match_label)
            loss_cc = ce_loss(cc_logits, cc_label)
            loss_mlo = ce_loss(mlo_logits, mlo_label)

            loss = (weights["gamma"] * loss_m +
                    weights["alpha"] * loss_cc +
                    weights["beta"] * loss_mlo)

            num_pairs = match_label.size(0)
            total_loss += loss.item() * num_pairs
            total_pairs += num_pairs

            # -------------------------
            # Compute accuracy
            # -------------------------
            pred_match = (dist < threshold).long()
            total_match_correct += (pred_match == match_label).sum().item()

            cc_pred = torch.argmax(cc_logits, dim=1)
            mlo_pred = torch.argmax(mlo_logits, dim=1)

            total_cc_correct += (cc_pred == cc_label).sum().item()
            total_mlo_correct += (mlo_pred == mlo_label).sum().item()

    return (
        total_loss / total_pairs,
        total_match_correct / total_pairs,
        total_cc_correct / total_pairs,
        total_mlo_correct / total_pairs,
    )

def plot_history(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # --- 1. Loss Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], 'r', label='Training Loss')
    plt.plot(epochs, history["val_loss"], 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

    # --- 2. Accuracy Plot ---
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["match_acc"], 'g', label='Match Accuracy (Contrastive)')
    plt.plot(epochs, history["cc_acc"], 'm', label='CC Class. Accuracy')
    plt.plot(epochs, history["mlo_acc"], 'c', label='MLO Class. Accuracy')
    plt.title('Validation Accuracies for Three Tasks')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_plot.png"))
    plt.close()

    print(f"\n[INFO] Plots saved to {save_dir}/")

def get_args():
    parser = argparse.ArgumentParser("CMCNet Siamese Training")

    # ---------------- Paths ----------------
    parser.add_argument("--train-dir", type=str, required=True,
                        help="Path to siamese training dataset")
    parser.add_argument("--val-dir", type=str, required=True,
                        help="Path to siamese validation dataset")
    parser.add_argument("--save-dir", type=str, default="cmcnet_logs")

    # ---------------- Model ----------------
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet pretrained backbone")

    # ---------------- Training ----------------
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=4)

    # ---------------- Siamese ----------------
    parser.add_argument("--K", type=int, default=5,
                        help="Number of positive / negative pairs per patient")
    parser.add_argument("--margin", type=float, default=5.0)

    # ---------------- Loss Weights ----------------
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="CC classification loss weight")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="MLO classification loss weight")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="Contrastive loss weight")

    return parser.parse_args()

# ------------------------------------------------------
# Main Training Loop
# ------------------------------------------------------
if __name__ == "__main__":

    args = get_args()

    train_dir = args.train_dir
    val_dir   = args.val_dir
    save_dir  = args.save_dir

    input_size = (args.input_size, args.input_size)
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    margin = args.margin


    save_dir = "cmcnet_logs"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    # --- 1. 计算分类权重 (用于加权交叉熵) ---
    # 假设的类别数量：Mass: 2610, Calcification: 570, Negative: 3180
    class_counts = torch.tensor([2610.0, 570.0, 3180.0]) 
    total_samples = class_counts.sum()

    # 计算类别权重 (Inverse Frequency Weighting: 样本数越少，权重越高)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = class_weights.to(device)


    
    print("\n[INFO] Loading dataset...")

    # NEW DATASET (CMCNet version)
    train_dataset = SiameseDatasetTrain(train_dir, input_size=input_size, K=args.K)
    val_dataset   = SiameseDatasetVal(val_dir, input_size=input_size, K=args.K)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=siamese_collate,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=siamese_collate
    )

    # Model

    #resume_path = "cmcnet_logs/best_model.pth"
    resume_path = None
    model = CMCNet(input_channels=3, num_classes=args.num_classes, pretrained=args.pretrained)
    model.to(device)
    
    
    # -------------------------------------------------
    # Resume Training from Best Model (if exists)
    # -------------------------------------------------
    if resume_path is not None and os.path.isfile(resume_path):
        print(f"🔄 Loading pretrained model from: {resume_path}")
        state_dict = torch.load(resume_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("⚠ No resume model loaded")

    

    # Loss functions
    ce_loss = nn.CrossEntropyLoss(weight=class_weights) # 使用加权交叉熵
    contrastive = ContrastiveLoss(margin)
    
    #ce_loss = nn.CrossEntropyLoss()
    #contrastive = ContrastiveLoss(margin)

    weights = {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma}
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    #scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    best_val = 1e9

    history = {
        "train_loss": [],
        "val_loss": [],
        "match_acc": [],
        "cc_acc": [],
        "mlo_acc": []
    }

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
    for epoch in range(1, epochs + 1):

        if epoch == 1:
            optimizer.param_groups[0]['lr'] = 0.001
        elif epoch == 40:
            optimizer.param_groups[0]['lr'] = 0.0005
        elif epoch == 80:
            optimizer.param_groups[0]['lr'] = 0.0001
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[LR] Current learning rate: {current_lr}")

        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            contrastive, ce_loss, weights,
            epoch, epochs
        )

        val_loss, match_acc, cc_acc, mlo_acc = validate_joint(
            model, val_loader, device,
            contrastive, ce_loss, weights,
            margin=margin
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["match_acc"].append(match_acc)
        history["cc_acc"].append(cc_acc)
        history["mlo_acc"].append(mlo_acc)

        print("\n------------------------------")
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss      : {train_loss:.4f}")
        print(f"Val Total Loss  : {val_loss:.4f}")
        print(f"Match Accuracy  : {match_acc:.4f}")
        print(f"CC Accuracy     : {cc_acc:.4f}")
        print(f"MLO Accuracy    : {mlo_acc:.4f}")
        print("------------------------------\n")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("✔ Saved best_model.pth")

        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pth"))
        plot_history(history, save_dir)
        #scheduler.step()
        #print(f"[LR] Current learning rate: {scheduler.get_last_lr()[0]}")
