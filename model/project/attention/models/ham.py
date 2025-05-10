import torch
import torch.nn as nn
import torch.nn.functional as F

class HAMModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.channel_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1
        )
        
        self.spatial_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            padding=3
        )
    def forward(self, x):

        # x: [B, C, H, W]
        b, c, h, w = x.size()

        # --- Channel Attention ---
        # descriptor: [B, C]
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, 1, c) # [B,1,C]
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, 1, c) # [B,1,C]
        # shared conv1d
        avg_att = self.channel_conv(avg_pool) # [B,1,C]
        max_att = self.channel_conv(max_pool) # [B,1,C]
        # sum and activation
        attn_channel = torch.sigmoid(avg_att + max_att).view(b, c, 1, 1)
        x_channel = x * attn_channel

        # --- Spatial Attention ---
        # descriptor: [B, 1, H, W]
        avg_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_map = torch.cat([avg_spatial, max_spatial], dim=1)  # [B,2,H,W]
        attn_spatial = torch.sigmoid(self.spatial_conv(spatial_map))  # [B,1,H,W]
        x_out = x_channel * attn_spatial

        return x_out