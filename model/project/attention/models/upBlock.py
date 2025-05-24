import torch.nn as nn

from attention.models.ham import HAMModule
from attention.models.spatialAttention import SpatialAttention

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attn=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 
                             stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        if use_attn:
            self.attn = HAMModule(out_channels)
        else:
            self.attn = None

    def forward(self, x):
        x = self.conv(x)
        if self.attn is not None:
            x = self.attn(x)
        return x