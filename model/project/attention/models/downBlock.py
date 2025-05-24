import torch.nn as nn
import torch

from attention.models.chanelAttention import ChannelAttention

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.GELU(),
            ChannelAttention(out_c)
        )

    def forward(self, x):
        return self.conv(x)