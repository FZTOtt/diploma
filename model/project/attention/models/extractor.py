import torch.nn as nn
import torch
import torch.nn.functional as F
from attention.models.downBlock import DownBlock
from attention.models.ham import HAMModule
from attention.models.resudalBlock import ResidualBlock
from attention.models.upBlock import UpBlock

class Extractor(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = DownBlock(in_channels, base_channels)
        self.enc2 = DownBlock(base_channels, base_channels*2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(base_channels*2) for _ in range(4)]
        )
        
        # Decoder с skip-connections
        self.dec1 = UpBlock(base_channels*2, base_channels)
        self.dec2 = UpBlock(base_channels*2, base_channels)  # skip from enc1
        
        # Attention в последнем слое
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            HAMModule(base_channels),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        b = self.bottleneck(e2)
        
        d1 = self.dec1(b)
        d1 = torch.cat([d1, e1], dim=1)
        
        d2 = self.dec2(d1)
        return self.final(d2)
