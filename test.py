import torch
import torch.nn.functional as F
from nets.cmcnet import CMCNet  # 假设您的模型文件
from utils.dataloader import SiameseDataset, siamese_collate
from collections import defaultdict
import os
import random
from PIL import Image
from torchvision import transforms


# --- 1. 定义推理数据集 (无随机增强) ---
# 必须使用与验证集相同的无随机性Transforms
class InferenceDataset(Dataset):
    def __init__(self, patch_list, input_size=(64, 64)):
        self.patch_list = patch_list
        # 必须使用与验证集相同的无随机性Transforms
        self.to_tensor = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        # patch_list 应该是 (path, patient_id, view, original_box_info)
        path, pid, view, box_info = self.patch_list[idx]
        img = Image.open(path).convert("RGB")
        img_tensor = self.to_tensor(img)
        
        # 返回图像和元数据，用于后续分组
        return img_tensor, pid, view, box_info


# --- 2. 核心推理和筛选逻辑 ---

def run_full_inference(model, candidate_data_path, margin=15.0, device='cuda'):
    
    model.eval()
    
    # 假设这里有一个函数 load_all_candidates 负责加载并分组 YOLO 补丁
    # data_groups = {pid: { 'CC': [patch_info_list], 'MLO': [patch_info_list] }}
    data_groups = load_all_candidates(candidate_data_path) 
    
    final_detections = []
    threshold = margin / 2.0  # 匹配阈值，例如 7.5

    with torch.no_grad():
        for pid, views in data_groups.items():
            cc_patches = views.get('CC', [])
            mlo_patches = views.get('MLO', [])
            
            # --- 步骤 2a: 分组特征提取 ---
            
            # 为简化起见，这里假设我们只提取特征 (f1, f2)
            # 在实际操作中，您会批量处理所有 CC 补丁，然后批量处理所有 MLO 补丁
            
            cc_features = {} # {patch_id: feature_vector}
            mlo_features = {}
            
            # 简化：仅处理分类和特征提取
            # (省略批量处理代码，假设您已经获得了每个补丁的 feature vector)
            
            # 假设 patch_info 是 (tensor, id, box_info, features)
            # --- 步骤 2b: N x M 匹配和筛选 ---
            
            # 记录找到匹配的补丁ID
            cc_matched_ids = set()
            mlo_matched_ids = set()
            
            for i, p_cc in enumerate(cc_patches):
                for j, p_mlo in enumerate(mlo_patches):
                    
                    # 实际操作：将 p_cc 和 p_mlo 的张量输入模型
                    # 重新运行模型的前向传播以获取特征
                    
                    img_cc, img_mlo = p_cc['tensor'].unsqueeze(0), p_mlo['tensor'].unsqueeze(0)
                    img_cc, img_mlo = img_cc.to(device), img_mlo.to(device)

                    # CMCNet forward pass
                    dist, cc_logits, mlo_logits = model((img_cc, img_mlo)) 
                    
                    distance = dist.item()
                    
                    # 匹配判断
                    if distance < threshold:
                        cc_matched_ids.add(p_cc['id'])
                        mlo_matched_ids.add(p_mlo['id'])
                        
                        # (可选：记录最佳匹配对)

            # --- 步骤 2c: 应用最终过滤规则 ---
            
            # 过滤规则：只有在另一个视图中找到匹配的补丁，才保留该候选补丁。
            
            for p_cc in cc_patches:
                if p_cc['id'] in cc_matched_ids:
                    # 分类结果 (例如，预测概率)
                    score = F.softmax(p_cc['logits'], dim=1).max().item() 
                    final_detections.append({'pid': pid, 'view': 'CC', 'box': p_cc['box'], 'score': score})
                    
            for p_mlo in mlo_patches:
                if p_mlo['id'] in mlo_matched_ids:
                    score = F.softmax(p_mlo['logits'], dim=1).max().item()
                    final_detections.append({'pid': pid, 'view': 'MLO', 'box': p_mlo['box'], 'score': score})


    return final_detections

# --- 辅助函数：需要您根据实际 YOLO 输出文件结构实现 ---
def load_all_candidates(root_dir):
    # 此函数应加载所有补丁，并将它们按 patient_id 和 view 分组
    # 必须确保加载时应用了 InferenceDataset 的无随机性 transforms
    
    # 示例结构：
    # patches = [{'id': 'cc_1', 'tensor': tensor, 'box': box, 'view': 'CC', 'pid': 'A'}, ...]
    # 然后按 pid 分组
    
    print("WARNING: load_all_candidates must be implemented by the user.")
    return {}

# -----------------------------------------------------
# --- MAIN EXECUTION ---
# -----------------------------------------------------
if __name__ == "__main__":
    # 加载最佳模型权重
    model = CMCNet(input_channels=3, num_classes=3, pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("cmcnet_logs/best_model.pth", map_location=device))
    model.to(device)

    # 运行推理 (假设您有INbreast测试集对应的YOLO输出目录)
    candidate_patches_path = "path/to/inbreast_yolo_candidates"
    results = run_full_inference(model, candidate_patches_path, margin=15.0, device=device)
    
    print(f"Final filtered detections: {len(results)}")
    # (后续：将结果保存为FROC曲线格式，计算AUC)
