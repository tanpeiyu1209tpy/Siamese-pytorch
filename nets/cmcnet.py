# ==========================================================
# File: nets/cmcnet.py
# ==========================================================
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class CMCNet(nn.Module):
    """
    Combined Matching and Classification Network (CMCNet)
    论文: Yan et al., "Multi-tasking Siamese Networks for Breast Mass Detection
          Using Dual-View Mammogram Matching", MLMI 2020.
    """
    def __init__(self, input_channels=3, num_classes=4, pretrained=True):
        super(CMCNet, self).__init__()

        # 1. VGG16 Backbone -------------------------------------------
        if pretrained:
            weights = VGG16_Weights.IMAGENET1K_V1
            vgg = vgg16(weights=weights)
        else:
            vgg = vgg16(weights=None)

        if input_channels != 3:
            vgg.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)

        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        flat_dim = 512

        # 2. Classification Heads -------------------------------------
        self.cls_head_cc = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self.cls_head_mlo = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: tuple(img_cc, img_mlo)
               - img_cc : (B, C, H, W)
               - img_mlo: (B, C, H, W)
        Returns:
            distance:  (B,)   欧氏距离 (用于 ContrastiveLoss)
            logits_cc: (B,C)  CC 分类输出
            logits_mlo:(B,C)  MLO 分类输出
        """
        img_cc, img_mlo = x

        # --- Feature extraction (shared weights) ---
        f1 = self.features(img_cc)
        f2 = self.features(img_mlo)
        f1 = self.avgpool(f1)
        f2 = self.avgpool(f2)
        f1 = torch.flatten(f1, 1)
        f2 = torch.flatten(f2, 1)

        # --- L2 Distance for contrastive loss ---
        distance = torch.norm(f1 - f2, p=2, dim=1)

        # --- Classification branches ---
        logits_cc = self.cls_head_cc(f1)
        logits_mlo = self.cls_head_mlo(f2)

        return distance, logits_cc, logits_mlo


# ---------- Quick unit test ----------
if __name__ == "__main__":
    model = CMCNet(input_channels=3, num_classes=4, pretrained=False)
    cc = torch.randn(8, 3, 64, 64)
    mlo = torch.randn(8, 3, 64, 64)
    d, cc_out, mlo_out = model((cc, mlo))
    print("distance:", d.shape)        # (8,)
    print("cc_out:", cc_out.shape)     # (8,4)
    print("mlo_out:", mlo_out.shape)   # (8,4)
