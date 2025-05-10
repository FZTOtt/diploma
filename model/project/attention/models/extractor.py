import torch.nn as nn
import torch.nn.functional as F
from attention.models.ham import HAMModule   # или просто from .ham import HAMModule

class Extractor(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        # 8 блоков Conv3×3 + IN + ReLU
        layers = []
        for _ in range(8):
            layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1))
            layers.append(nn.InstanceNorm2d(base_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = base_channels  # после первого цикла in_channels==base_channels
        
        self.conv_blocks = nn.Sequential(*layers)
        
        # два HAM-модуля подряд
        self.ham1 = HAMModule(base_channels)
        self.ham2 = HAMModule(base_channels)
        
        # финальный свёрточный слой
        self.final_conv = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        x: stego-изображение [B,3,H,W]
        """
        # 1) 8× Conv3×IN×ReLU
        x = self.conv_blocks(x)           # [B, base_channels, H, W]
        
        # 2) HAM 1
        x = self.ham1(x)                  # [B, base_channels, H, W]
        # 3) HAM 2
        x = self.ham2(x)                  # [B, base_channels, H, W]
        
        # 4) финальный свёрточный слой + Tanh
        x = self.tanh(self.final_conv(x)) # [B, 3, H, W]
        return x
