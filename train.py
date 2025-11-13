# ==========================================================
# File: train.py
# ==========================================================
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.cmcnet import CMCNet
from utils.dataloader import SiameseDataset, dataset_collate
from utils.callbacks import LossHistory
from utils.utils import get_lr_scheduler, set_optimizer_lr, show_config


# -------------------- Contrastive Loss --------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label = label.float()
        loss_pos = label * torch.pow(distance, 2)
        loss_neg = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss = 0.5 * torch.mean(loss_pos + loss_neg)
        return loss


# ==========================================================
if __name__ == "__main__":
    # ---------------- Hyperparameters ---------------------
    Cuda            = True
    distributed     = False
    sync_bn         = False
    fp16            = False

    dataset_path    = "./siamese_data_train"
    val_dataset_path= "./siamese_data_val"

    input_shape     = [64, 64]
    num_classes     = 4
    pretrained      = True
    model_path      = ""
    Init_Epoch      = 0
    Epoch           = 100
    batch_size      = 32

    Init_lr         = 1e-3
    Min_lr          = 1e-5
    optimizer_type  = "sgd"
    momentum        = 0.9
    weight_decay    = 5e-4
    lr_decay_type   = "cos"
    save_period     = 10
    save_dir        = "logs"
    num_workers     = 4

    # ----------------- Device setup -----------------------
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
    else:
        device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank  = 0
        rank        = 0

    # ----------------- Model -------------------------------
    model = CMCNet(input_channels=3, num_classes=num_classes, pretrained=pretrained)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    # ----------------- Loss functions ----------------------
    margin = 5.0
    loss_match_fn = ContrastiveLoss(margin=margin)
    loss_cls_fn   = nn.CrossEntropyLoss()
    loss_weights  = {'alpha':1.0, 'beta':1.0, 'gamma':0.1}

    # ----------------- Logging -----------------------------
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    # ----------------- Data -------------------------------
    train_dataset = SiameseDataset(input_shape, dataset_path, random_flag=True, autoaugment_flag=True)
    val_dataset   = SiameseDataset(input_shape, val_dataset_path, random_flag=False, autoaugment_flag=False)
    num_train, num_val = len(train_dataset), len(val_dataset)

    if local_rank == 0:
        show_config(model_path=model_path, input_shape=input_shape,
                    Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size,
                    Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type,
                    momentum=momentum, lr_decay_type=lr_decay_type,
                    save_period=save_period, save_dir=save_dir,
                    num_workers=num_workers, num_train=num_train, num_val=num_val)

    # ----------------- Optimizer --------------------------
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == "adam" else 1e-1
    lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit  = min(max(batch_size / nbs * Min_lr,  lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        "adam": optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        "sgd" : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size
    gen     = DataLoader(train_dataset, shuffle=True,  batch_size=batch_size,
                         num_workers=num_workers, pin_memory=True, drop_last=True,
                         collate_fn=dataset_collate)
    gen_val = DataLoader(val_dataset,   shuffle=False, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=True, drop_last=True,
                         collate_fn=dataset_collate)

    # ======================================================
    for epoch in range(Init_Epoch, Epoch):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        model_train.train()

        total_loss, total_match_loss, total_cls_loss = 0.0, 0.0, 0.0
        pbar = tqdm(total=epoch_step, desc=f"Epoch {epoch+1}/{Epoch} Train")

        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step: break
            (img_cc, img_mlo), (match_labels, cc_labels, mlo_labels) = batch

            match_labels = match_labels.view(-1)
            img_cc, img_mlo = img_cc.to(device), img_mlo.to(device)
            match_labels = match_labels.to(device)
            cc_labels, mlo_labels = cc_labels.to(device), mlo_labels.to(device)

            optimizer.zero_grad()
            distance, cc_logits, mlo_logits = model_train((img_cc, img_mlo))

            loss_m = loss_match_fn(distance, match_labels)
            loss_c_cc = loss_cls_fn(cc_logits, cc_labels)
            loss_c_mlo = loss_cls_fn(mlo_logits, mlo_labels)

            loss = loss_weights["gamma"]*loss_m + loss_weights["alpha"]*loss_c_cc + loss_weights["beta"]*loss_c_mlo
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_match_loss += loss_m.item()
            total_cls_loss += (loss_c_cc.item()+loss_c_mlo.item())/2
            pbar.set_postfix(loss=total_loss/(iteration+1),
                             match=total_match_loss/(iteration+1),
                             cls=total_cls_loss/(iteration+1))
            pbar.update(1)
        pbar.close()

        # ---------------- Validation -----------------------
        model_train.eval()
        val_total_loss, val_match_loss, val_cls_loss = 0.0, 0.0, 0.0
        val_match_correct = val_cls_cc_correct = val_cls_mlo_correct = 0
        val_total = 0
        match_threshold = margin / 2.0
        pbar = tqdm(total=epoch_step_val, desc=f"Epoch {epoch+1}/{Epoch} Val")

        with torch.no_grad():
            for iteration, batch in enumerate(gen_val):
                if iteration >= epoch_step_val: break
                (img_cc, img_mlo), (match_labels, cc_labels, mlo_labels) = batch
                match_labels = match_labels.view(-1)
                img_cc, img_mlo = img_cc.to(device), img_mlo.to(device)
                match_labels = match_labels.to(device)
                cc_labels, mlo_labels = cc_labels.to(device), mlo_labels.to(device)

                distance, cc_logits, mlo_logits = model_train((img_cc, img_mlo))
                loss_m = loss_match_fn(distance, match_labels)
                loss_c_cc = loss_cls_fn(cc_logits, cc_labels)
                loss_c_mlo = loss_cls_fn(mlo_logits, mlo_labels)
                loss = loss_weights["gamma"]*loss_m + loss_weights["alpha"]*loss_c_cc + loss_weights["beta"]*loss_c_mlo

                val_total_loss += loss.item()
                val_match_loss += loss_m.item()
                val_cls_loss += (loss_c_cc.item()+loss_c_mlo.item())/2

                # accuracy
                pred_match = (distance < match_threshold).long()
                val_match_correct += (pred_match == match_labels.long()).sum().item()
                val_cls_cc_correct += (torch.argmax(cc_logits,1) == cc_labels).sum().item()
                val_cls_mlo_correct+= (torch.argmax(mlo_logits,1)== mlo_labels).sum().item()
                val_total += match_labels.size(0)

                pbar.set_postfix(val_loss=val_total_loss/(iteration+1))
                pbar.update(1)
        pbar.close()

        # metrics
        avg_train_loss = total_loss / epoch_step
        avg_val_loss = val_total_loss / epoch_step_val
        match_acc = val_match_correct / val_total
        cls_acc_cc = val_cls_cc_correct / val_total
        cls_acc_mlo= val_cls_mlo_correct / val_total
        print(f"Epoch {epoch+1}/{Epoch} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} "
              f"| MatchAcc {match_acc:.4f} | CC {cls_acc_cc:.4f} | MLO {cls_acc_mlo:.4f}")

        if loss_history:
            loss_history.append_loss(epoch+1, avg_train_loss, avg_val_loss)

        if (epoch+1)%save_period==0 or epoch+1==Epoch:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"ep{epoch+1:03d}-loss{avg_train_loss:.4f}-val{avg_val_loss:.4f}.pth"))

    if loss_history:
        loss_history.writer.close()
