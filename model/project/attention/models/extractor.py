import torch.nn as nn
import torch
import torch.nn.functional as F
from attention.models.downBlock import DownBlock
from attention.models.generator import AdaptiveFusionBlock, HierarchicalDecoder, MultiScaleEncoder
from attention.models.ham import HAMModule
from attention.models.resudalBlock import ResidualBlock
from attention.models.upBlock import UpBlock

class Extractor(nn.Module):
    def __init__(self, in_c=3, base_c=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            DownBlock(in_c, base_c, use_attn=True),
            DownBlock(base_c, base_c*2, use_attn=True)
        )
        
        self.decoder = nn.Sequential(
            UpBlock(base_c*2, base_c, use_attn=True),
            UpBlock(base_c, in_c)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class ImprovedExtractor(nn.Module):
    def __init__(self, in_c=3, base_c=64):
        super().__init__()
        self.encoder = MultiScaleEncoder(in_c, base_c)
        self.fusion = nn.ModuleList([
            AdaptiveFusionBlock(base_c * (2**i)) for i in range(3)
        ])
        self.decoder = HierarchicalDecoder(base_c * 8, in_c)
        
    def forward(self, stego):
        features = self.encoder(stego)
        # Обратное слияние с learnable weights
        merged = [self.fusion[i](f, torch.zeros_like(f)) for i, f in enumerate(features)]
        return self.decoder(merged)