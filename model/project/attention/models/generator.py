import torch
import torch.nn as nn
from attention.models.ham import HAMModule
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        input_cover_layers = []

        input_cover_layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1))
        input_cover_layers.append(nn.InstanceNorm2d(base_channels))
        input_cover_layers.append(nn.ReLU(inplace=True))

        input_cover_layers.append(nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1))
        input_cover_layers.append(nn.InstanceNorm2d(base_channels))
        input_cover_layers.append(nn.ReLU(inplace=True))

        self.conv_cover_blocks = nn.Sequential(*input_cover_layers)

        input_secret_layers = []

        input_secret_layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1))
        input_secret_layers.append(nn.InstanceNorm2d(base_channels))
        input_secret_layers.append(nn.ReLU(inplace=True))

        input_secret_layers.append(nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1))
        input_secret_layers.append(nn.InstanceNorm2d(base_channels))
        input_secret_layers.append(nn.ReLU(inplace=True))

        self.conv_secret_blocks = nn.Sequential(*input_secret_layers)

        fusion_layers = []

        fusion_layers.append(nn.Conv2d(base_channels, base_channels, kernel_size=1))
        fusion_layers.append(nn.InstanceNorm2d(base_channels))
        fusion_layers.append(nn.ReLU(inplace=True))

        for _ in range(2):
            fusion_layers.append(nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1))
            fusion_layers.append(nn.InstanceNorm2d(base_channels))
            fusion_layers.append(nn.ReLU(inplace=True))

        self.fusion_layers = nn.Sequential(*fusion_layers)

        self.ham1 = HAMModule(base_channels)

        up_sampling_layers = []

        up_sampling_layers.append(nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1))
        up_sampling_layers.append(nn.InstanceNorm2d(base_channels))
        up_sampling_layers.append(nn.ReLU(inplace=True))

        for _ in range(2):
            up_sampling_layers.append(nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1))
            up_sampling_layers.append(nn.InstanceNorm2d(base_channels))
            up_sampling_layers.append(nn.ReLU(inplace=True))

        self.up_sampling_layers = nn.Sequential(*up_sampling_layers)

        self.ham2 = HAMModule(base_channels)

        output_layers = []

        for _ in range(2):
            output_layers.append(nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1))
            # output_layers.append(nn.InstanceNorm2d(base_channels))
            output_layers.append(nn.ReLU(inplace=True))

        output_layers.append(nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1))
        
        self.output_layers = nn.Sequential(*output_layers)
        self.tanh = nn.Tanh()

    def forward(self, cover, secret):

        # отдельно идёт первая пара conv

        c = self.conv_cover_blocks(cover)
        
        s = self.conv_secret_blocks(secret)
        
        # поэлементная сумма признаков
        x = c + s
        
        # fusion-блок
        x = self.fusion_layers(x)
        
        # HAM 1
        x = self.ham1(x)
        
        # up-sampling
        x = self.up_sampling_layers(x)
        
        # HAM 2
        x = self.ham2(x)

        x = self.output_layers(x)
        
        # выход
        x = self.tanh(x)
        return x
