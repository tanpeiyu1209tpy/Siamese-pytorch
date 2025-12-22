# ==========================================================
# File: nets/cmcnet.py
# ==========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class CMCNet(nn.Module):
    """
    Combined Matching and Classification Network (CMCNet)
    Yan et al., "Multi-tasking Siamese Networks for Breast Mass Detection
    Using Dual-View Mammogram Matching", MLMI 2020
    """
    def __init__(self, input_channels=3, num_classes=3, pretrained=True):
        super().__init__()

        # --------------------------------------------------
        # 1. VGG16 backbone (shared)
        # --------------------------------------------------
        if pretrained:
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg = vgg16(weights=None)

        # ⚠️ Only replace first conv if NOT RGB
        if input_channels != 3:
            vgg.features[0] = nn.Conv2d(
                input_channels, 64, kernel_size=3, padding=1, bias=True
            )
            nn.init.kaiming_normal_(
                vgg.features[0].weight, mode="fan_out", nonlinearity="relu"
            )

        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat_dim = 512

        # --------------------------------------------------
        # 2. Classification heads (CC / MLO)
        # --------------------------------------------------
        self.cls_head_cc = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self.cls_head_mlo = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # --------------------------------------------------
        # 3. Metric head (shared Siamese mapping)
        # --------------------------------------------------
        self.metric_head = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.flat_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (img_cc, img_mlo)
               img_cc : (B, C, H, W)
               img_mlo: (B, C, H, W)
        Returns:
            distance : (B,)
            logits_cc: (B, num_classes)
            logits_mlo:(B, num_classes)
        """
        img_cc, img_mlo = x

        # --- Shared feature extraction ---
        f_cc = self.avgpool(self.features(img_cc))
        f_mlo = self.avgpool(self.features(img_mlo))

        f_cc = torch.flatten(f_cc, 1)   # (B, 512)
        f_mlo = torch.flatten(f_mlo, 1) # (B, 512)

        # --- Classification ---
        logits_cc = self.cls_head_cc(f_cc)
        logits_mlo = self.cls_head_mlo(f_mlo)

        # --- Metric learning ---
        z_cc = self.metric_head(f_cc)
        z_mlo = self.metric_head(f_mlo)

        # (optional but recommended) L2 normalization
        z_cc = F.normalize(z_cc, p=2, dim=1)
        z_mlo = F.normalize(z_mlo, p=2, dim=1)

        distance = torch.norm(z_cc - z_mlo, p=2, dim=1)

        return distance, logits_cc, logits_mlo


# --------------------------------------------------
# Quick sanity check
# --------------------------------------------------
if __name__ == "__main__":
    model = CMCNet(input_channels=3, num_classes=4, pretrained=False)
    cc = torch.randn(8, 3, 128, 128)
    mlo = torch.randn(8, 3, 128, 128)
    d, cc_out, mlo_out = model((cc, mlo))
    print("distance:", d.shape)
    print("cc_out:", cc_out.shape)
    print("mlo_out:", mlo_out.shape)
