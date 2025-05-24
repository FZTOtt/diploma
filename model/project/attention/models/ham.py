import torch
import torch.nn as nn
import torch.nn.functional as F

class HAMModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
    
    def forward(self, x):
        # Channel
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        channel_att = self.fc(avg).view(b, c, 1, 1)
        x = x * channel_att
        
        # Spatial
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att