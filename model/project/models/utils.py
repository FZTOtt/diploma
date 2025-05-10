import torch

def chaotic_permute(x, mu=3.99, key=0.5):
    B, C, H, W = x.shape
    seq = torch.zeros(H*W, device=x.device)
    seq[0] = key
    for i in range(1, H*W):
        seq[i] = mu * seq[i-1] * (1 - seq[i-1])
    _, idx = torch.sort(seq)
    x_flat = x.view(B, C, -1)
    x_perm = torch.gather(x_flat, 2, idx.unsqueeze(0).unsqueeze(0).expand(B, C, H*W))
    return x_perm.view(B, C, H, W), idx

def inverse_permute(x, idx):
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)
    inv = torch.zeros_like(x_flat)
    inv.scatter_(2, idx.unsqueeze(0).unsqueeze(0).expand(B, C, H*W), x_flat)
    return inv.view(B, C, H, W)