import torch.nn as nn
import torch
import torch.nn.functional as F
from attention.models.downBlock import DownBlock
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
