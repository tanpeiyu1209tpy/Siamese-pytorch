import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class CMCNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=4, pretrained=True):
        """
        初始化 CMCNet 模型。
        
        Args:
            input_channels (int): 输入图像的通道数。VGG16 预训练模型需要 3 通道。
            num_classes (int): 你的分类任务的类别总数。
                               (e.g., 3 个病灶类 + 1 个背景类 = 4)
            pretrained (bool): 是否加载 ImageNet 预训练权重。
        """
        super(CMCNet, self).__init__()
        
        # 1. 加载 VGG16 特征提取器 (Backbone)
        # ---------------------------------------------
        if pretrained:
            weights = VGG16_Weights.IMAGENET1K_V1
            vgg = vgg16(weights=weights)
        else:
            vgg = vgg16(weights=None)
            
        # 检查输入通道数
        if input_channels != 3:
            # 修改 VGG 的第一个卷积层以接受不同通道数的输入
            # 注意：这会破坏第一层的预训练权重
            vgg.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
            
        # 我们只需要 'features' 部分，丢弃 'avgpool' 和 'classifier'
        self.features = vgg.features
        
        # 论文中提到的 Global Average Pooling
        # 这使模型对输入 patch 尺寸的变化更具鲁棒性
        # 输出将始终为 [batch_size, 512, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # VGG16 'features' 的输出通道数为 512
        flat_shape = 512 

        # 2. 匹配头 (Metric Network) - B
        # ---------------------------------------------
        # 它接收两个 flatten 后的特征向量拼接 (512 + 512 = 1024)
        self.match_head = nn.Sequential(
            nn.Linear(flat_shape * 2, 512),
            nn.ReLU(inplace=True),
            # 输出 1 个分数 (用于 Contrastive Loss 或 BCE Loss)
            nn.Linear(512, 1) 
        )
        
        # 3. 分类头 (Classification Heads) - C
        # ---------------------------------------------
        # 你的关键修正：输出必须是 num_classes (即 4)
        self.cls_head_cc = nn.Sequential(
            nn.Linear(flat_shape, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes) # 输出 4 个类别
        )
        
        # MLO 分支的独立分类头
        self.cls_head_mlo = nn.Sequential(
            nn.Linear(flat_shape, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes) # 输出 4 个类别
        )

    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (tuple): 一个包含 (img1, img2) 的元组。
                       img1 是 CC patch, img2 是 MLO patch。
        """
        img1, img2 = x
        
        # --- 1. 特征提取 (A: Feature network) ---
        # 两个分支共享权重
        f1 = self.features(img1) # (B, 512, H/32, W/32)
        f2 = self.features(img2) # (B, 512, H/32, W/32)
        
        # 应用 Global Average Pooling
        f1 = self.avgpool(f1) # (B, 512, 1, 1)
        f2 = self.avgpool(f2) # (B, 512, 1, 1)
        
        # 展平特征
        f1 = torch.flatten(f1, 1) # (B, 512)
        f2 = torch.flatten(f2, 1) # (B, 512)

        # --- 2. 匹配任务 (B: Metric network) ---
        # 拼接两个特征向量
        x_match = torch.cat([f1, f2], dim=1) # (B, 1024)
        match_score = self.match_head(x_match) # (B, 1)

        # --- 3. 分类任务 (C: Classification) ---
        cls_score_cc = self.cls_head_cc(f1)   # (B, 4)
        cls_score_mlo = self.cls_head_mlo(f2) # (B, 4)

        # 返回三个独立的输出，用于计算三个损失
        return match_score, cls_score_cc, cls_score_mlo

if __name__ == '__main__':
    # 测试模型是否能运行
    # 假设输入是 64x64 的 3 通道图像
    # (BatchSize, Channels, Height, Width)
    cc_tensor = torch.randn(8, 3, 64, 64)
    mlo_tensor = torch.randn(8, 3, 64, 64)
    
    # 你的 num_classes 是 4
    model = CMCNet(input_channels=3, num_classes=4, pretrained=False)
    
    # 将元组传入模型
    match_output, cc_cls_output, mlo_cls_output = model((cc_tensor, mlo_tensor))
    
    print(f"Model created successfully.")
    print(f"Match output shape:   {match_output.shape}")
    print(f"CC Class output shape:  {cc_cls_output.shape}")
    print(f"MLO Class output shape: {mlo_cls_output.shape}")

    # 预期输出:
    # Match output shape:   torch.Size([8, 1])
    # CC Class output shape:  torch.Size([8, 4])
    # MLO Class output shape: torch.Size([8, 4])

'''
import torch
import torch.nn as nn

from nets.vgg import VGG16


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height) 
    
class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()
        self.vgg = VGG16(pretrained, 3)
        del self.vgg.avgpool
        del self.vgg.classifier
        
        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])

        # matching branch
        #self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect1 = torch.nn.Linear(flat_shape*2, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)
        
# ------------------------------------------------------
        # classification branch
        self.cls_head_cc = nn.Sequential(
            nn.Linear(flat_shape, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3) # output 3 classes
        )
        # MLO 分支可以独立，也可以和 CC 共享权重。论文似乎是独立的。
        self.cls_head_mlo = nn.Sequential(
            nn.Linear(flat_shape, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3)
        )
# --------------------------------------------------------

    def forward(self, x):
        '''
        x1, x2 = x
        #------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        #------------------------------------------#
        x1 = self.vgg.features(x1)
        x2 = self.vgg.features(x2)   
        #-------------------------#
        #   相减取绝对值，取l1距离
        #-------------------------#     
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        #x = torch.abs(x1 - x2)
        x = torch.cat([x1, x2], dim=1)
        #-------------------------#
        #   进行两次全连接
        #-------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        # return x
        '''
        img1, img2 = x
        
        # fetature extraction
        f1 = self.vgg.features(img1)
        f2 = self.vgg.features(img2)
        f1 = torch.flatten(f1, 1)
        f2 = torch.flatten(f2, 1)

        # pairing 
        x_match = torch.cat([f1, f2], dim=1) # concat
        x_match = self.match_fc1(x_match)
        match_score = self.match_fc2(x_match)

        # classifying 
        cls_score1 = self.cls_head_cc(f1)   # CC图片分类得分
        cls_score2 = self.cls_head_mlo(f2)  # MLO图片分类得分

        return match_score, cls_score1, cls_score2
'''

