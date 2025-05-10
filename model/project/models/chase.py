import torch.nn as nn
from models.haar_wavelet import HaarWavelet 
import torch
from models.inn_block import INNBlock
from models.utils import chaotic_permute, inverse_permute
from utils.helps import rgb_to_ycrcb
import torch.nn.functional as F

class CHASE(nn.Module):
    def __init__(self, num_blocks=16):
        super().__init__()
        self.haar_y = HaarWavelet(in_channels=1)
        self.haar_secret = HaarWavelet(in_channels=3)

        self.inn_blocks = nn.ModuleList([INNBlock(16) for _ in range(num_blocks)])

        self.haar_y_inv = self.haar_y.inverse
        self.haar_secret_inv = self.haar_secret.inverse

        self.sigmoid = nn.Sigmoid()

    def forward_hide(self, cover, secret, key=0.5):

        cov_y = rgb_to_ycrcb(cover)[:, :1]
        sec_perm, idx = chaotic_permute(secret, key=key)

        y_wave = self.haar_y(cov_y)     # [B, 4, H/2, W/2]
        s_wave = self.haar_secret(sec_perm) # [B, 12, H/2, W/2]

        print(f"y_wave shape: {y_wave.shape}")  # Ожидается [B, 4, H/2, W/2]
        print(f"s_wave shape: {s_wave.shape}")  # Ожидается [B, 12, H/2, W/2]

        x_wave = torch.cat([y_wave, s_wave], dim=1)


        for block in self.inn_blocks:
            x_wave = block(x_wave)
            
        # Разделение на стего и потери
        stego_wave, r = torch.split(x_wave, [4, 12], dim=1)
        
        # Обратное преобразование Y-канала
        stego_spatial = self.haar_y_inv(stego_wave)
        return self.sigmoid(stego_spatial), r, idx

    def forward_reveal(self, stego, r, idx):
        # Вейвлет для стего (Y-канал)
        stego_wave = self.haar_y(stego)
        
        # Объединение с потерями
        x_wave = torch.cat([stego_wave, r], dim=1)
        
        # Обратный проход через INN блоки
        for block in reversed(self.inn_blocks):
            x_wave = block.inverse(x_wave)
            
        # Разделение на Y и секрет
        y_wave_rec, s_wave_rec = torch.split(x_wave, [4, 12], dim=1)
        
        # Обратное преобразование
        y_rec = self.haar_y_inv(y_wave_rec)
        sec_scr = self.haar_secret_inv(s_wave_rec)

        assert not torch.isnan(y_rec).any(), "NaN in y_rec"
        assert not torch.isnan(sec_scr).any(), "NaN in sec_scr"
        
        # Обратная перестановка
        sec = inverse_permute(sec_scr, idx)
        return y_rec, sec