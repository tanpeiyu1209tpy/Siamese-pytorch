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
# 我们不再需要 fit_one_epoch 或 load_dataset

if __name__ == "__main__":
    #----------------------------------------------------#
    #   Cuda
    #----------------------------------------------------#
    Cuda            = True
    distributed     = False
    sync_bn         = False
    fp16            = False
    
    #----------------------------------------------------#
    #   [ 关键修改 ] 数据集路径
    #   指向你用 "paper_strategy" 脚本生成的路径
    #----------------------------------------------------#
    dataset_path    = "/kaggle/working/siamese_data_train"
    # --- [ 新增 ] ---
    #   [ 关键修改 ] 验证集路径
    #   请确保你已手动创建此文件夹并移入了约 10-20% 的 *病人* 数据
    #----------------------------------------------------#
    val_dataset_path = "/kaggle/working/siamese_data_val" 
    
    #----------------------------------------------------#
    #   [ 关键修改 ] 输入图像的大小
    #   必须匹配你 dataloader.py 和数据准备脚本中的 PATCH_SIZE
    #----------------------------------------------------#
    input_shape     = [64, 64]
    
    #----------------------------------------------------#
    #   [ 关键修改 ] 分类任务的类别数
    #   (Mass, Calc, Focal) + 1 (Not-Lesion) = 4
    #----------------------------------------------------#
    num_classes     = 4
    
    #-------------------------------#
    #   预训练
    #-------------------------------#
    pretrained      = True
    model_path      = ""

    #------------------------------------------------------#
    #   训练参数
    #------------------------------------------------------#
    Init_Epoch      = 0
    Epoch           = 100
    batch_size      = 32 # (dataloader 会返回 32*2 = 64 对)
    
    #------------------------------------------------------------------#
    #   学习率、优化器
    #------------------------------------------------------------------#
    Init_lr             = 1e-3
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 5e-4
    lr_decay_type       = 'cos'
    save_period         = 10
    save_dir            = 'logs'
    num_workers         = 4

    #------------------------------------------------------#
    #   DDP 设置
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank  = 0
        rank        = 0

    #------------------------------------------------------#
    #   [ 关键修改 ] 实例化你的 CMCNet 模型
    #------------------------------------------------------#
    model = CMCNet(input_channels=3, num_classes=num_classes, pretrained=pretrained)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……")
            print("\nFail To Load Key:", str(no_load_key)[:500], "……")

    #------------------------------------------------------#
    #   [ 关键修改 ] 在这里定义你的 Loss
    #------------------------------------------------------#
    # 1. 匹配损失 (用于 Match Head)
    loss_match_fn = nn.BCEWithLogitsLoss()
    # 2. 分类损失 (用于两个 Class Head)
    loss_cls_fn = nn.CrossEntropyLoss()
    
    # 论文中的损失权重 (Eq. 2), 你可以调整
    loss_weights = {
        'alpha': 1.0, # cls_cc
        'beta':  1.0, # cls_mlo
        'gamma': 1.0  # match
    }

    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None
        
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #----------------------------------------------------#
    #   [ 关键修改 ] 实例化你的多任务 Dataloader
    #----------------------------------------------------#
    print(f"Loading training data from: {dataset_path}")
    train_dataset   = SiameseDataset(input_shape, dataset_path, random_flag=True, autoaugment_flag=True)
    
    # --- [ 关键修改 ] ---
    #   使用独立的验证集路径
    #----------------------
    if not os.path.exists(val_dataset_path):
        print(f"Warning: Validation path not found: {val_dataset_path}")
        print("Using training data for validation (NOT RECOMMENDED).")
        val_dataset_path = dataset_path # Fallback
    else:
        print(f"Loading validation data from: {val_dataset_path}")
        
    val_dataset     = SiameseDataset(input_shape, val_dataset_path, random_flag=False, autoaugment_flag=False)
    
    num_train       = len(train_dataset)
    num_val         = len(val_dataset)

    if local_rank == 0:
        show_config(
            model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
 
    if True:
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen     = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)

        #------------------------------------------------------#
        #   [ 关键修改 ] 移除了 fit_one_epoch，重写训练循环
        #------------------------------------------------------#
        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            # --- 训练 TRAIN ---
            model_train.train()
            total_loss = 0
            total_match_loss = 0
            total_cls_loss = 0
            
            if local_rank == 0:
                print(f'\n--- Epoch {epoch + 1}/{Epoch} ---')
                pbar = tqdm(total=epoch_step, desc=f'Train', postfix=dict, mininterval=0.3)
                
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_step:
                    break
                
                # [!] Dataloader 返回 images 元组 和 labels 元组
                images, labels = batch
                images_cc, images_mlo = images[0], images[1]
                match_targets, cls_cc_targets, cls_mlo_targets = labels

                if Cuda:
                    images_cc = images_cc.cuda(local_rank)
                    images_mlo = images_mlo.cuda(local_rank)
                    match_targets = match_targets.cuda(local_rank)
                    cls_cc_targets = cls_cc_targets.cuda(local_rank)
                    cls_mlo_targets = cls_mlo_targets.cuda(local_rank)

                optimizer.zero_grad()
                
                # [!] 模型返回三个输出
                match_preds, cls_cc_preds, cls_mlo_preds = model_train((images_cc, images_mlo))
                
                # [!] 计算三个 Loss
                loss_m = loss_match_fn(match_preds, match_targets)
                loss_c_cc = loss_cls_fn(cls_cc_preds, cls_cc_targets)
                loss_c_mlo = loss_cls_fn(cls_mlo_preds, cls_mlo_targets)
                
                # [!] 论文中的多任务损失 (Eq. 2)
                loss = (loss_weights['gamma'] * loss_m) + \
                       (loss_weights['alpha'] * loss_c_cc) + \
                       (loss_weights['beta'] * loss_c_mlo)
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_match_loss += loss_m.item()
                total_cls_loss += (loss_c_cc.item() + loss_c_mlo.item()) / 2
                
                if local_rank == 0:
                    pbar.set_postfix(**{'loss'     : total_loss / (iteration + 1), 
                                        'match_loss' : total_match_loss / (iteration + 1),
                                        'cls_loss' : total_cls_loss / (iteration + 1),
                                        'lr'       : optimizer.param_groups[0]['lr']})
                    pbar.update(1)
            
            if local_rank == 0:
                pbar.close()

            # --- 验证 VALIDATION ---
            model_train.eval()
            val_total_loss = 0
            val_match_loss = 0
            val_cls_loss = 0
            
            # --- [ 新增 ] 用于计算 Accuracy ---
            val_match_correct = 0
            val_cls_cc_correct = 0
            val_cls_mlo_correct = 0
            val_total_samples = 0
            # --- [ 新增结束 ] ---
            
            if local_rank == 0:
                print('--- Validation ---')
                pbar = tqdm(total=epoch_step_val, desc=f'Val  ', postfix=dict, mininterval=0.3)
                
            for iteration, batch in enumerate(gen_val):
                if iteration >= epoch_step_val:
                    break
                
                with torch.no_grad():
                    images, labels = batch
                    images_cc, images_mlo = images[0], images[1]
                    match_targets, cls_cc_targets, cls_mlo_targets = labels

                    if Cuda:
                        images_cc = images_cc.cuda(local_rank)
                        images_mlo = images_mlo.cuda(local_rank)
                        match_targets = match_targets.cuda(local_rank)
                        cls_cc_targets = cls_cc_targets.cuda(local_rank)
                    cls_mlo_targets = cls_mlo_targets.cuda(local_rank)

                    match_preds, cls_cc_preds, cls_mlo_preds = model_train((images_cc, images_mlo))
                    
                    loss_m = loss_match_fn(match_preds, match_targets)
                    loss_c_cc = loss_cls_fn(cls_cc_preds, cls_cc_targets)
                    loss_c_mlo = loss_cls_fn(cls_mlo_preds, cls_mlo_targets)
                    loss = (loss_weights['gamma'] * loss_m) + \
                           (loss_weights['alpha'] * loss_c_cc) + \
                           (loss_weights['beta'] * loss_c_mlo)
                    
                    val_total_loss += loss.item()
                    val_match_loss += loss_m.item()
                    val_cls_loss += (loss_c_cc.item() + loss_c_mlo.item()) / 2
                    
                    # --- [ 新增 ] 计算 Accuracy ---
                    # 1. Match Accuracy
                    match_preds_binary = (match_preds > 0).float()
                    val_match_correct += (match_preds_binary == match_targets).sum().item()
                    
                    # 2. CC Class Accuracy
                    cls_cc_preds_labels = torch.argmax(cls_cc_preds, dim=1)
                    val_cls_cc_correct += (cls_cc_preds_labels == cls_cc_targets).sum().item()
                    
                    # 3. MLO Class Accuracy
                    cls_mlo_preds_labels = torch.argmax(cls_mlo_preds, dim=1)
                    val_cls_mlo_correct += (cls_mlo_preds_labels == cls_mlo_targets).sum().item()
                    
                    val_total_samples += match_targets.size(0)
                    # --- [ 新增结束 ] ---
                    
                if local_rank == 0:
                    # --- [ 修改 ] 更新 pbar ---
                    running_match_acc = val_match_correct / val_total_samples if val_total_samples > 0 else 0
                    running_cls_acc = (val_cls_cc_correct + val_cls_mlo_correct) / (val_total_samples * 2) if val_total_samples > 0 else 0
                    
                    pbar.set_postfix(**{
                        'val_loss' : val_total_loss / (iteration + 1),
                        'match_acc': f"{running_match_acc:.4f}",
                        'cls_acc'  : f"{running_cls_acc:.4f}"
                    })
                    # --- [ 修改结束 ] ---
                    pbar.update(1)
            
            # --- Epoch 结束 ---
            if local_rank == 0:
                pbar.close()
                avg_train_loss = total_loss / epoch_step
                avg_val_loss = val_total_loss / epoch_step_val
                
                # --- [ 新增 ] 计算最终 Accuracy ---
                final_match_acc = val_match_correct / val_total_samples
                final_cls_cc_acc = val_cls_cc_correct / val_total_samples
                final_cls_mlo_acc = val_cls_mlo_correct / val_total_samples
                avg_cls_acc = (final_cls_cc_acc + final_cls_mlo_acc) / 2
                # --- [ 新增结束 ] ---
                
                # --- [ 修改 ] 更新打印信息 ---
                print(f'Epoch {epoch + 1}/{Epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                print(f'  [Validation] Match Acc: {final_match_acc:.4f}, Avg Class Acc: {avg_cls_acc:.4f} (CC: {final_cls_cc_acc:.4f}, MLO: {final_cls_mlo_acc:.4f})')
                # --- [ 修改结束 ] ---
                
                #if loss_history:
                #    loss_history.append_loss(avg_train_loss, avg_val_loss)
                if loss_history:
                    loss_history.append_loss(epoch + 1, avg_train_loss, avg_val_loss)
                
                if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
                    # --- [ 修改 ] 保存文件名中加入 val_loss ---
                    torch.save(model.state_dict(), os.path.join(save_dir, f'ep{epoch + 1:03d}-loss{avg_train_loss:.4f}-val_loss{avg_val_loss:.4f}.pth'))

        if local_rank == 0:
            loss_history.writer.close()
