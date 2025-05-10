import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch, stride in [(64,2),(128,2),(256,2),(512,2)]:
            layers += [nn.Conv2d(in_ch, out_ch, 4, stride, 1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, True)]
            in_ch = out_ch
        layers += [nn.Conv2d(512, 1, 4, 1, 0), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, x): return self.model(x).view(-1)
