import torch.nn as nn
import torch.nn.functional as F
from attention.models.ham import HAMModule

class Extractor(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        input_layers = []
        for _ in range(2):
            input_layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1))
            input_layers.append(nn.InstanceNorm2d(base_channels))
            input_layers.append(nn.ReLU(inplace=True))
            in_channels = base_channels
        
        self.conv_start_blocks = nn.Sequential(*input_layers)
        
        # два HAM-модуля подряд
        self.ham1 = HAMModule(base_channels)

        middle_layers = []
        for _ in range(2):
            middle_layers.append(nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1))
            middle_layers.append(nn.InstanceNorm2d(base_channels))
            middle_layers.append(nn.ReLU(inplace=True))

        self.conv_middle_blocks = nn.Sequential(*middle_layers)

        self.ham2 = HAMModule(base_channels)
        
        # финальный свёрточный слой
        output_layers = []
        for _ in range(3):
            output_layers.append(nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1))
            output_layers.append(nn.InstanceNorm2d(base_channels))
            output_layers.append(nn.ReLU(inplace=True))

        self.conv_output_blocks = nn.Sequential(*output_layers)

        self.final_conv = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        x: stego-изображение [B,3,H,W]
        """
        x = self.conv_start_blocks(x)     # [B, base_channels, H, W]
        
        x = self.ham1(x)                  # [B, base_channels, H, W]

        x = self.conv_middle_blocks(x)

        x = self.ham2(x)                  # [B, base_channels, H, W]

        x = self.conv_output_blocks(x)
        
        x = self.tanh(self.final_conv(x)) # [B, 3, H, W]
        return x
