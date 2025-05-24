import torch
import torch.nn as nn

from attention.models.chanelAttention import ChannelAttention
from attention.models.spatialAttention import SpatialAttention

class FusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(2*channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            ChannelAttention(channels),
            SpatialAttention()
        )
    
    def forward(self, cover_feat, secret_feat):
        x = torch.cat([cover_feat, secret_feat], dim=1)
        return self.fusion(x)