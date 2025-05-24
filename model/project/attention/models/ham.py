import torch
import torch.nn as nn
import torch.nn.functional as F

class HAMModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.reduction = max(1, channels // reduction)
        
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, self.reduction, 1),
            nn.ReLU(),
            nn.Conv2d(self.reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_att(x)
        x = ca * x
        
        # Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg, mx], dim=1))
        
        return sa * x