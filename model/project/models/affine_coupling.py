import torch.nn as nn
import torch

class AffineCouplingLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.05))
        self.split = in_channels // 2
        self.psi = nn.Sequential(
            nn.Conv2d(self.split, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.split, 3, padding=1)
        )
        self.phi = nn.Sequential(
            nn.Conv2d(self.split, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.split, 3, padding=1)
        )
        self.rho = nn.Sequential(
            nn.Conv2d(self.split, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.split, 3, padding=1)
        )
        self.eta = nn.Sequential(
            nn.Conv2d(self.split, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.split, 3, padding=1)
        )

        for layer in [self.psi, self.phi, self.rho, self.eta]:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        s1 = torch.sigmoid(self.alpha * self.psi(x2))
        t1 = self.phi(x2)
        y1 = s1 * x1 + t1
        s2 = torch.sigmoid(self.alpha * self.rho(y1))
        t2 = self.eta(y1)
        y2 = s2 * x2 + t2
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y):
        y1, y2 = torch.chunk(y, 2, dim=1)
        s2 = torch.sigmoid(self.alpha * self.rho(y1))
        t2 = self.eta(y1)
        x2 = (y2 - t2) / (s2 + 1e-8)
        s1 = torch.sigmoid(self.alpha * self.psi(x2))
        t1 = self.phi(x2)
        x1 = (y1 - t1) / (s1 + 1e-8)
        return torch.cat([x1, x2], dim=1)