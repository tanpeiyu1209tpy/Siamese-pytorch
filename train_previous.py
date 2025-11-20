import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# [ 关键修改 ] 导入你新的模型和 Dataloader
from nets.cmcnet import CMCNet
from utils.dataloader import SiameseDataset, dataset_collate

from utils.callbacks import LossHistory
from utils.utils import (get_lr_scheduler, set_optimizer_lr, show_config)


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
    num_classes     = 3 # 0,1,2=lesions, 3=not-lesion
    pretrained      = True
    model_path      = ""
    Init_Epoch      = 0
    Epoch           = 50
    batch_size      = 32

    # --- [ Plan E 修复 ] ---
    # 学习率太高导致过拟合，我们降到 1e-5
    Init_lr         = 1e-5
    Min_lr          = 1e-7 # 相应降低
    # --- [ 修复结束 ] ---
    
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

    # ----------------- Data -------------------------------
    # [!] 我们在实例化 Dataloader *之前* 定义 Loss
    # 因为我们需要 Dataloader 统计的数据来计算权重
    print(f"Loading training data from: {dataset_path}")
    train_dataset = SiameseDataset(input_shape, dataset_path, random_flag=True, autoaugment_flag=True)
    
    print(f"Loading validation data from: {val_dataset_path}")
    if not os.path.exists(val_dataset_path):
        print(f"Warning: Validation path not found: {val_dataset_path}")
        val_dataset = None
        num_val = 0
    else:
        val_dataset   = SiameseDataset(input_shape, val_dataset_path, random_flag=False, autoaugment_flag=False)
        num_val = len(val_dataset)
        
    num_train = len(train_dataset)

    # ----------------- Loss functions ----------------------
    margin = 10.0
    loss_match_fn = ContrastiveLoss(margin=margin)
    
    # --- [ Plan E 修复：加权损失 ] ---
    # 根据 Dataloader 统计的数据来解决类别不平衡问题
    
    # 1. 计算总的正负样本数 (用于分类)
    # 你的分类任务是 0,1,2 (positive) vs 3 (negative)
    total_pos = train_dataset.pos_cc_count + train_dataset.pos_mlo_count
    total_neg = train_dataset.neg_cc_count + train_dataset.neg_mlo_count
    
    # 2. 为类别 3 (negative) 计算权重
    # 我们有 3 个 positive classes (0,1,2), 1 个 negative class (3)
    
    # 防止除以 0
    if total_pos == 0: total_pos = 1
    if total_neg == 0: total_neg = 1 # 虽然这不应该发生
    
    # 平衡策略：给样本少的类更高的权重
    # 我们希望 (weight_pos * total_pos) 和 (weight_neg * total_neg) 大致相等
    # 简单起见，我们给 "positive" 类别一个统一的、更高的权重
    
    # 权重 = (总样本数 / 类别数) / 类别样本数
    # total_samples = total_pos + total_neg
    # weight_pos = (total_samples / 4) / (total_pos / 3) # (假设 pos 均匀分布在 3 个类)
    # weight_neg = (total_samples / 4) / (total_neg / 1)
    
    # 简化的策略：直接根据 pos/neg 比例来赋权
    # 我们有 3 个 pos 类, 1 个 neg 类
    # pos 类的总权重 = total_neg 
    # neg 类的总权重 = total_pos
    # 每个 pos 类的权重 = total_neg / 3
    # neg 类的权重 = total_pos / 1
    # [!] 这会导致梯度爆炸。
    
    # 最终策略：归一化 (让权重之和为 4)
    weight_pos = (total_pos + total_neg) / (2 * total_pos) # 权重给 3 个 pos 类
    weight_neg = (total_pos + total_neg) / (2 * total_neg) # 权重给 1 个 neg 类
    
    # [!] 修正：PyTorch 的权重是 1 / N_k。我们直接用最简单的方法：
    # 给稀有类（Positive）更高的权重
    weight_for_pos_class = total_neg / total_pos
    
    # 确保权重至少为 1
    if weight_for_pos_class < 1.0:
        weight_for_pos_class = 1.0

    # weights tensor: [Cls 0, Cls 1, Cls 2, Cls 3]
    class_weights = torch.tensor([
        weight_for_pos_class, 
        weight_for_pos_class, 
        weight_for_pos_class, 
        1.0 # 数量多的负样本，权重为 1
    ]).to(device)
    
    print(f"[Init] Total Pos: {total_pos}, Total Neg: {total_neg}")
    print(f"[Init] Imbalance Ratio (Neg/Pos): {total_neg/total_pos:.2f}")
    print(f"[Init] Applying Class Weights: [{weight_for_pos_class:.2f}, {weight_for_pos_class:.2f}, {weight_for_pos_class:.2f}, 1.00]")
    
    loss_cls_fn   = nn.CrossEntropyLoss(weight=class_weights)
    # --- [ Plan E 修复结束 ] ---

    # --- [ Plan C 修复：隔离问题 ] ---
    # 我们仍然保持 gamma=0.0，直到分类任务被成功解决
    loss_weights  = {'alpha':1.0, 'beta':1.0, 'gamma':0.0}
    # --- [ 修复结束 ] ---

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
    # [!] 使用 1e-5 的极低学习率
    Init_lr_fit = Init_lr
    Min_lr_fit  = Min_lr

    optimizer = {
        "adam": optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        "sgd" : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size
    
    if epoch_step == 0:
        raise ValueError("训练数据集过小，epoch_step 为 0。请增加数据或减小 batch_size。")

    gen     = DataLoader(train_dataset, shuffle=True,  batch_size=batch_size,
                         num_workers=num_workers, pin_memory=True, drop_last=True,
                         collate_fn=dataset_collate)
    
    if val_dataset and epoch_step_val > 0:
        gen_val = DataLoader(val_dataset,   shuffle=False, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True, drop_last=True,
                             collate_fn=dataset_collate)
    else:
        print("Warning: Validation set is too small or missing. Skipping validation.")
        gen_val = None
        epoch_step_val = 0 # 确保验证循环被跳过

    # ======================================================
    for epoch in range(Init_Epoch, Epoch):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        model_train.train()

        total_loss, total_match_loss, total_cls_loss = 0.0, 0.0, 0.0
        pbar = tqdm(total=epoch_step, desc=f"Epoch {epoch+1}/{Epoch} Train")

        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step: break
            (img_cc, img_mlo), (match_labels, cc_labels, mlo_labels) = batch
            
            # [!] collate_fn 现在返回 (B*2, ...)
            match_labels = match_labels.to(device)
            img_cc, img_mlo = img_cc.to(device), img_mlo.to(device)
            cc_labels, mlo_labels = cc_labels.to(device), mlo_labels.to(device)


            optimizer.zero_grad()
            distance, cc_logits, mlo_logits = model_train((img_cc, img_mlo))

            loss_m = loss_match_fn(distance, match_labels.view(-1)) # 确保 label 是一维
            loss_c_cc = loss_cls_fn(cc_logits, cc_labels)
            loss_c_mlo = loss_cls_fn(mlo_logits, mlo_labels)

            loss = loss_weights["gamma"]*loss_m + loss_weights["alpha"]*loss_c_cc + loss_weights["beta"]*loss_c_mlo
            
            if fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
        if not gen_val or local_rank != 0:
            if loss_history:
                 loss_history.append_loss(epoch+1, total_loss / epoch_step, 0) # 记录 train loss
            continue # 如果没有验证集或不是主进程，则跳过

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
                
                match_labels = match_labels.to(device)
                img_cc, img_mlo = img_cc.to(device), img_mlo.to(device)
                cc_labels, mlo_labels = cc_labels.to(device), mlo_labels.to(device)

                distance, cc_logits, mlo_logits = model_train((img_cc, img_mlo))
                loss_m = loss_match_fn(distance, match_labels.view(-1))
                loss_c_cc = loss_cls_fn(cc_logits, cc_labels)
                loss_c_mlo = loss_cls_fn(mlo_logits, mlo_labels)
                loss = loss_weights["gamma"]*loss_m + loss_weights["alpha"]*loss_c_cc + loss_weights["beta"]*loss_c_mlo

                val_total_loss += loss.item()
                val_match_loss += loss_m.item()
                val_cls_loss += (loss_c_cc.item()+loss_c_mlo.item())/2

                # accuracy
                pred_match = (distance < match_threshold).long()
                val_match_correct += (pred_match == match_labels.view(-1).long()).sum().item()
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
