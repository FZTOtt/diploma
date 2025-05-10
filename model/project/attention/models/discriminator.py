import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 6, base_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4, 1)

    def forward(self, cover: torch.Tensor, stego: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cover, stego], dim=1)  # [B, 6, H, W]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        score = self.fc(x)
        return score