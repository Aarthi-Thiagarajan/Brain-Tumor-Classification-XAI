import torch
import torch.nn as nn
import timm


class HybridSwinViT(nn.Module):
    """Swin Transformer + Vision Transformer feature fusion classifier."""

    def __init__(self, num_classes, pretrained=True, dropout=0.3):
        super().__init__()
        self.swin = timm.create_model("swin_tiny_patch4_window7_224",
                                      pretrained=pretrained, num_classes=0)
        self.vit = timm.create_model("vit_small_patch16_224",
                                     pretrained=pretrained, num_classes=0)
        dim = self.swin.num_features + self.vit.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(torch.cat([self.swin(x), self.vit(x)], dim=1))
