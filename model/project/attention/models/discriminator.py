import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=6, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(base_channels),
            
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(base_channels*2),
            
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(base_channels*4),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels*4, 1)
        )

    def forward(self, cover: torch.Tensor, stego: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cover, stego], dim=1)
        return self.net(x)