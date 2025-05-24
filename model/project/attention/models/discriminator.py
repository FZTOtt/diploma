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
    
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_c=3, base_c=64):
        super().__init__()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_c*2, base_c),
            PatchDiscriminator(in_c*2, base_c*2),
            PatchDiscriminator(in_c*2, base_c*4)
        ])
        
    def forward(self, cover, stego):
        x = torch.cat([cover, stego], dim=1)
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)
        return outputs

class PatchDiscriminator(nn.Module):
    def __init__(self, in_c, base_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, base_c, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            DownBlock(base_c, base_c*2),
            DownBlock(base_c*2, base_c*4),
            nn.Conv2d(base_c*4, 1, 4, padding=1)
        )
        
    def forward(self, x):
        return self.net(x)