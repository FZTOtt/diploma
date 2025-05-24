import torch.nn as nn

from attention.models.ham import HAMModule
from attention.models.spatialAttention import SpatialAttention

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, use_attn=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2)
        ]
        if use_attn:
            layers.append(HAMModule(out_c))
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)