import torch
import torch.nn as nn
from attention.models.downBlock import DownBlock
from attention.models.fusionBlock import FusionBlock
from attention.models.ham import HAMModule
import torch.nn.functional as F

from attention.models.resudalBlock import ResidualBlock
from attention.models.upBlock import UpBlock

class Generator(nn.Module):
    def __init__(self, in_c=3, base_c=64):
        super().__init__()
        
        # Энкодеры
        self.cover_enc = nn.Sequential(
            DownBlock(in_c, base_c),
            DownBlock(base_c, base_c*2, use_attn=True)
        )
        
        self.secret_enc = nn.Sequential(
            DownBlock(in_c, base_c),
            DownBlock(base_c, base_c*2)
        )

        # Боттлнек
        self.fusion = nn.Sequential(
            ResidualBlock(base_c*4),
            ResidualBlock(base_c*4),
            HAMModule(base_c*4)  # Добавим внимание в генератор
        )

        # Декодер
        self.decoder = nn.Sequential(
            UpBlock(base_c*4, base_c*2),
            UpBlock(base_c*2, in_c)
        )

    def forward(self, cover, secret):
        c_feat = self.cover_enc(cover)
        s_feat = self.secret_enc(secret)
        fused = torch.cat([c_feat, s_feat], dim=1)
        processed = self.fusion(fused)
        return self.decoder(processed)
    

class ImprovedGenerator(nn.Module):
    def __init__(self, in_c=3, base_c=64):
        super().__init__()
        
        # Многоуровневые энкодеры
        self.cover_enc = MultiScaleEncoder(in_c, base_c)
        self.secret_enc = MultiScaleEncoder(in_c, base_c)

        # Блоки адаптивного внедрения для каждого масштаба
        self.fusion_blocks = nn.ModuleList([
            AdaptiveFusionBlock(base_c * (2**i)) for i in range(3)
        ])
        
        # Декодер с skip-connections
        self.decoder = HierarchicalDecoder(base_c * 8, in_c)

    def forward(self, cover, secret):
        # Извлечение признаков разных масштабов
        cover_features = self.cover_enc(cover)  # [f1, f2, f3]
        secret_features = self.secret_enc(secret)
        
        # Многоуровневое слияние
        fused_features = []
        for c_feat, s_feat, fusion_block in zip(cover_features, secret_features, self.fusion_blocks):
            fused = fusion_block(c_feat, s_feat)
            fused_features.append(fused)
        
        # Иерархический декодинг
        return self.decoder(fused_features)

class AdaptiveFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.texture_att = TextureAttention(channels)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, cover_feat, secret_feat):
        # Генерация текстуры для маскирования
        texture_mask = self.texture_att(cover_feat)
        
        # Адаптивное слияние каналов
        combined = torch.cat([cover_feat, secret_feat * texture_mask], dim=1)
        channel_weights = self.channel_gate(combined)
        
        return cover_feat + channel_weights * secret_feat

class TextureAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.gradient = GradientFilter()
        
    def forward(self, x):
        # Вычисление градиентов для определения текстурных областей
        gradients = self.gradient(x)
        return self.conv(gradients)

class GradientFilter(nn.Module):
    def __init__(self):
        super().__init__()
        # Динамическая инициализация ядер для любого числа каналов
        self.register_buffer('kernel_x', None)
        self.register_buffer('kernel_y', None)

    def forward(self, x):
        channels = x.size(1)
        
        # Создаем ядра при первом вызове или при изменении числа каналов
        if self.kernel_x is None or self.kernel_x.size(0) != channels:
            sobel_x = torch.tensor([[[1, 0, -1], 
                                   [2, 0, -2], 
                                   [1, 0, -1]]], dtype=torch.float32)
            sobel_x = sobel_x.repeat(channels, 1, 1, 1)  # [channels, 1, 3, 3]
            self.register_buffer('kernel_x', sobel_x.to(x.device))
            
            sobel_y = torch.tensor([[[1, 2, 1], 
                                   [0, 0, 0], 
                                   [-1, -2, -1]]], dtype=torch.float32)
            sobel_y = sobel_y.repeat(channels, 1, 1, 1)
            self.register_buffer('kernel_y', sobel_y.to(x.device))

        g_x = F.conv2d(x, self.kernel_x, padding=1, groups=channels)
        g_y = F.conv2d(x, self.kernel_y, padding=1, groups=channels)
        return torch.sqrt(g_x**2 + g_y**2 + 1e-6)

class HierarchicalDecoder(nn.Module):
    def __init__(self, in_c=3, base_c=64):
        super().__init__()
        
        # Энкодеры
        self.cover_enc = nn.Sequential(
            DownBlock(in_c, base_c),
            DownBlock(base_c, base_c*2)
        )
        
        self.secret_enc = nn.Sequential(
            DownBlock(in_c, base_c),
            DownBlock(base_c, base_c*2)
        )

        # Боттлнек
        self.fusion = nn.Sequential(
            nn.Conv2d(base_c*4, base_c*4, 3, padding=1),
            ResidualBlock(base_c*4),
            HAMModule(base_c*4)
        )

        # Упрощенный декодер
        self.decoder = nn.Sequential(
            UpBlock(base_c*4, base_c*2),
            UpBlock(base_c*2, base_c),
            UpBlock(base_c, in_c),
            nn.Tanh()
        )

    def forward(self, cover, secret):
        c_feat = self.cover_enc(cover)
        s_feat = self.secret_enc(secret)
        
        fused = torch.cat([c_feat, s_feat], dim=1)
        processed = self.fusion(fused)
        return self.decoder(processed)

class MultiScaleEncoder(nn.Module):
    def __init__(self, in_c, base_c):
        super().__init__()
        self.down1 = DownBlock(in_c, base_c)
        self.down2 = DownBlock(base_c, base_c*2)
        self.down3 = DownBlock(base_c*2, base_c*4)
        
    def forward(self, x):
        f1 = self.down1(x)  # 1/2
        f2 = self.down2(f1) # 1/4
        f3 = self.down3(f2) # 1/8
        return [f1, f2, f3]