import torch
import torch.nn.functional as F

def psnr(a, b):
    mse = F.mse_loss(a, b)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

ssim_index = ssim(stego, cover, data_range=1.0)