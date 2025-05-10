from utils.chaos_permutation import ChaosPermutation
import torch

def prepare_inputs(cover_rgb, secret_rgb, chaos: ChaosPermutation):
    """
    Преобразует cover и secret изображения согласно статье CHASE.
    
    cover_rgb: [3, H, W] — RGB host image
    secret_rgb: [3, H, W] — RGB secret image
    chaos: ChaosPermutation instance
    """
    # 1. Cover image → Y channel
    cover_ycrcb = rgb_to_ycrcb(cover_rgb)
    cover_y = cover_ycrcb[:, 0:1, :, :]  # форма: [B, 1, H, W]

    # 2. Secret image → YCrCb
    secret_ycrcb = rgb_to_ycrcb(secret_rgb)

    # 3. Scramble secret image
    secret_ycrcb_scrambled = chaos.permute(secret_ycrcb)

    return cover_y.unsqueeze(0), secret_ycrcb_scrambled.unsqueeze(0)

def rgb_to_ycrcb(image_rgb):
    """ Convert RGB tensor to YCrCb tensor """
    r = image_rgb[:, 0, :, :]  # Извлекаем R-канал
    g = image_rgb[:, 1, :, :]  # Извлекаем G-канал
    b = image_rgb[:, 2, :, :]  # Извлекаем B-канал
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 0.5 * (r - y) * 0.713 + 128
    cb = 0.5 * (b - y) * 0.564 + 128
    
    return torch.stack([y, cr, cb], dim=1)

def ycrcb_to_rgb(image_ycrcb):
    """ Convert YCrCb tensor to RGB tensor """
    y = image_ycrcb[0, :, :]
    cr = image_ycrcb[1, :, :]
    cb = image_ycrcb[2, :, :]
    
    r = y + 1.403 * (cr - 128)
    g = y - 0.714 * (cr - 128) - 0.344 * (cb - 128)
    b = y + 1.773 * (cb - 128)
    
    return torch.clamp(torch.stack([r, g, b], dim=0), 0, 1)