import torch
import torch.nn as nn
from attention.models.downBlock import DownBlock
from attention.models.fusionBlock import FusionBlock
from attention.models.ham import HAMModule
import torch.nn.functional as F

from attention.models.resudalBlock import ResidualBlock
from attention.models.upBlock import UpBlock

class Generator(nn.Module):
    def __init__(self, in_c=3, base_c=64):
        super().__init__()
        
        # Энкодеры
        self.cover_enc = nn.Sequential(
            DownBlock(in_c, base_c),
            DownBlock(base_c, base_c*2, use_attn=True)
        )
        
        self.secret_enc = nn.Sequential(
            DownBlock(in_c, base_c),
            DownBlock(base_c, base_c*2)
        )

        # Боттлнек
        self.fusion = nn.Sequential(
            ResidualBlock(base_c*4),
            ResidualBlock(base_c*4),
            HAMModule(base_c*4)  # Добавим внимание в генератор
        )

        # Декодер
        self.decoder = nn.Sequential(
            UpBlock(base_c*4, base_c*2),
            UpBlock(base_c*2, in_c)
        )

    def forward(self, cover, secret):
        c_feat = self.cover_enc(cover)
        s_feat = self.secret_enc(secret)
        fused = torch.cat([c_feat, s_feat], dim=1)
        processed = self.fusion(fused)
        return self.decoder(processed)
