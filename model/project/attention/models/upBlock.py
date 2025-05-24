import torch.nn as nn

from attention.models.spatialAttention import SpatialAttention

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.GELU(),
            SpatialAttention()
        )

    def forward(self, x):
        return self.up(x)