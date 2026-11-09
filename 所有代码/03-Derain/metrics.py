"""Image metrics, averaged per image. SSIM uses an 11x11 valid Gaussian window.
Y uses full-range BT.601 luma: 0.299 R + 0.587 G + 0.114 B.
Inputs are NCHW (or CHW) in [0,1]; both inputs are clipped.
"""
import torch
from torch.nn import functional as F


def _prepare(a, b, channel):
    if a.ndim == 3:
        a, b = a.unsqueeze(0), b.unsqueeze(0)
    if a.shape != b.shape or a.ndim != 4 or a.shape[1] != 3:
        raise ValueError('expected matching RGB CHW or NCHW tensors')
    a, b = a.clamp(0, 1), b.clamp(0, 1)
    if channel.upper() == 'Y':
        weights = a.new_tensor([.299, .587, .114]).view(1, 3, 1, 1)
        a, b = (a * weights).sum(1, keepdim=True), (b * weights).sum(1, keepdim=True)
    elif channel.upper() != 'RGB':
        raise ValueError('channel must be RGB or Y')
    return a, b


def calc_psnr(a, b, channel='RGB'):
    a, b = _prepare(a, b, channel)
    return (-10 * torch.log10((a-b).square().mean((1, 2, 3)))).mean()


def calc_mae(a, b, channel='RGB'):
    a, b = _prepare(a, b, channel)
    return (a-b).abs().mean()


def calc_ssim(a, b, channel='RGB'):
    a, b = _prepare(a, b, channel)
    if min(a.shape[-2:]) < 11:
        raise ValueError('SSIM requires dimensions of at least 11')
    coords = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    g = torch.exp(-coords.square() / (2 * 1.5**2)); g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(a.shape[1], 1, 11, 11).contiguous()
    def conv(x):
        return F.conv2d(x, window, groups=a.shape[1])
    mu1, mu2 = conv(a), conv(b)
    v1, v2 = conv(a*a)-mu1.square(), conv(b*b)-mu2.square()
    cov = conv(a*b)-mu1*mu2
    return (((2*mu1*mu2+.01**2)*(2*cov+.03**2)) /
            ((mu1.square()+mu2.square()+.01**2)*(v1+v2+.03**2))).mean()
