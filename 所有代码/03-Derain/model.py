"""Three-scale residual rain estimation network."""
import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.LeakyReLU(.01), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class UpsampleBlock(nn.Module):
    def __init__(self, scale, channels=32):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.LeakyReLU(.01), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return self.net(F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False))


class DeRainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.LeakyReLU(.01))
        self.branches = nn.ModuleList(nn.Sequential(*(ResBlock() for _ in range(8))) for _ in range(3))
        self.ups = nn.ModuleList([UpsampleBlock(2), UpsampleBlock(4)])
        self.fuse = nn.Conv2d(96, 32, 3, padding=1)
        self.rain = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x, return_residual=False):
        # Pad to a multiple of four so all scales align, including odd input sizes.
        h, w = x.shape[-2:]
        z = self.stem(F.pad(x, (0, (-w) % 4, 0, (-h) % 4), mode='replicate'))
        p2, p3 = F.max_pool2d(z, 2), F.max_pool2d(z, 4)
        f1 = z + self.branches[0](z)
        f2 = self.ups[0](p2 + self.branches[1](p2))
        f3 = self.ups[1](p3 + self.branches[2](p3))
        rain = self.rain(self.fuse(torch.cat((f1, f2, f3), dim=1)))[..., :h, :w]
        out = x - rain
        return (out, rain) if return_residual else out
