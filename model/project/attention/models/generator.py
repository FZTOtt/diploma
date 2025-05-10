import torch
import torch.nn as nn
from attention.models.ham import HAMModule
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.cover_conv3 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.cover_conv4 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)

        self.secret_conv3 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.secret_conv4 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)

        self.fusion_conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.fusion_conv3_a = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.fusion_conv3_b = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        self.ham1 = HAMModule(base_channels)

        self.deconv4 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.post_conv3_a = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.post_conv3_b = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        self.ham2 = HAMModule(base_channels)

        self.final_conv3 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.final_conv3_a = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, cover, secret):

        # отдельно идёт первая пара conv
        c = F.relu(self.cover_conv3(cover))
        c = F.relu(self.cover_conv4(c))
        
        s = F.relu(self.secret_conv3(secret))
        s = F.relu(self.secret_conv4(s))
        
        # поэлементная сумма признаков
        x = c + s
        
        # fusion-блок
        x = F.relu(self.fusion_conv1(x))
        x = F.relu(self.fusion_conv3_a(x))
        x = F.relu(self.fusion_conv3_b(x))
        
        # HAM 1
        x = self.ham1(x)
        
        # up-sampling
        x = F.relu(self.deconv4(x))
        x = F.relu(self.post_conv3_a(x))
        x = F.relu(self.post_conv3_b(x))
        
        # HAM 2
        x = self.ham2(x)

        x = F.relu(self.final_conv3(x))
        x = F.relu(self.final_conv3(x))
        x = self.final_conv3_a(x)
        
        # выход
        x = self.tanh(x)
        return x
