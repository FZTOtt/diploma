import torch.nn as nn
import torch
import torch.nn.functional as F

class HaarWavelet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        filters = []
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        for f in (ll, lh, hl, hh):
            filters.append(f)
        weight = torch.stack(filters, dim=0).unsqueeze(1)
        weight = weight.repeat(in_channels, 1, 1, 1)
        self.register_buffer('weight', weight)
        self.in_channels = in_channels
        self.dwt = lambda x: F.conv2d(x, self.weight, stride=2, groups=in_channels)
        self.iwt = nn.ConvTranspose2d(4*in_channels, in_channels, kernel_size=2, stride=2, groups=in_channels, bias=False)
        self.iwt.weight.data = self.weight

    def forward(self, x):
        return self.dwt(x)

    def inverse(self, x):
        return self.iwt(x)