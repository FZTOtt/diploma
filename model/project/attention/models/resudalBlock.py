import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        residual = x
        x = self.conv(x)
        g = self.gate(x)
        return residual + x * g