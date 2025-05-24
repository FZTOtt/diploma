import torch
import torch.nn as nn
from attention.models.downBlock import DownBlock
from attention.models.fusionBlock import FusionBlock
from attention.models.ham import HAMModule
import torch.nn.functional as F

from attention.models.resudalBlock import ResidualBlock
from attention.models.upBlock import UpBlock

class Generator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # Encoder для cover и secret
        self.cover_encoder = nn.Sequential(
            DownBlock(in_channels, base_channels),
            DownBlock(base_channels, base_channels*2)
        )
        
        self.secret_encoder = nn.Sequential(
            DownBlock(in_channels, base_channels),
            DownBlock(base_channels, base_channels*2)
        )
        
        # Многоуровневое слияние
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(base_channels*2),
            FusionBlock(base_channels*2)
        ])
        
        # Обработка с residual блоками
        self.process = nn.Sequential(
            *[ResidualBlock(base_channels*2) for _ in range(4)]
        )
        
        # Decoder с attention
        self.decoder = nn.Sequential(
            UpBlock(base_channels*2, base_channels),
            HAMModule(base_channels),
            UpBlock(base_channels, in_channels),
            nn.Tanh()
        )

    def forward(self, cover, secret):
        # Кодируем оба входа
        cover_feat = self.cover_encoder(cover)
        secret_feat = self.secret_encoder(secret)
        
        # Многоуровневое слияние
        fused = self.fusion_blocks[0](cover_feat, secret_feat)
        fused = self.fusion_blocks[1](fused, secret_feat)
        
        # Обработка признаков
        processed = self.process(fused)
        
        # Декодирование
        output = self.decoder(processed)
        
        return output
