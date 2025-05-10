import torch.nn as nn
from models.affine_coupling import AffineCouplingLayer

class INNBlock(nn.Module):
    def __init__(self, in_channels=16):  # Исправлено на 16 каналов
        super().__init__()
        self.affine = AffineCouplingLayer(in_channels=in_channels)
        
    def forward(self, x):
        return self.affine(x)
    
    def inverse(self, y):
        return self.affine.inverse(y)