import torch
import torch.nn as nn

from attention.models.downBlock import DownBlock

class Discriminator(nn.Module):
    def __init__(self, in_c=6, base_c=64):
        super().__init__()
        self.net = nn.Sequential(
            # Downsampling blocks
            DownBlock(in_c, base_c),
            DownBlock(base_c, base_c*2),
            DownBlock(base_c*2, base_c*4),
            
            # Final processing
            nn.Conv2d(base_c*4, base_c*8, 4, padding=1),
            nn.LeakyReLU(0.2),
            
            # Global pooling and final output
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_c*8, 1, 1)
        )

    def forward(self, cover, stego):
        x = torch.cat([cover, stego], dim=1)
        x = self.net(x)
        return x.view(x.size(0), -1)