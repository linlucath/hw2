"""WV3 reduced-resolution indexes translated from Pansharpening_test.

Input arrays are HWC float64, in the original 11-bit radiometric scale.
Q2n alone performs MATLAB's uint16 saturation/rounding, as in q2n.m.
"""
import numpy as np
from scipy.ndimage import correlate
from scipy.signal import fftconvolve
from skimage.metrics import structural_similarity


def _conj(x):
    return np.concatenate((x[..., :1], -x[..., 1:]), axis=-1)


def _onion(a, b):
    n = a.shape[-1]
    if n == 1:
        return a * b
    h = n // 2
    a1, b1 = a[..., :h], _conj(a[..., h:])
    c, d = b[..., :h], _conj(b[..., h:])
    if n == 2:
        return np.concatenate((a1*c-d*b1, a1*d+c*b1), axis=-1)
    return np.concatenate((_onion(a1, c)-_onion(d, _conj(b1)),
                           _onion(_conj(a1), d)+_onion(c, b1)), axis=-1)


def _onions_quality(a, b):
    a, b = a.astype(np.float64), _conj(b.astype(np.float64))
    avg = a.mean(axis=(0, 1))
    std = a.std(axis=(0, 1), ddof=1)
    std[std == 0] = np.finfo(float).eps
    a = (a - avg) / std + 1
    # m-file treats a zero reference mean specially, even when std is nonzero.
    signs = np.ones(a.shape[2]); signs[1:] = -1
    b = np.where(avg == 0, b + signs, (b - signs * avg) / std + signs)
    m1, m2 = a.mean(axis=(0, 1)), b.mean(axis=(0, 1))
    t2 = np.linalg.norm(m1) * np.linalg.norm(m2)
    t4 = np.sum(m1*m1) + np.sum(m2*m2)
    factor = a.shape[0]*a.shape[1] / (a.shape[0]*a.shape[1]-1)
    t3 = factor*(np.mean(np.sum(a*a, axis=2)) + np.mean(np.sum(b*b, axis=2)) - t4)
    bias = 2*t2/t4
    if t3 == 0:
        q = np.zeros_like(m1); q[-1] = bias
        return q
    return factor*(_onion(a, b).mean(axis=(0, 1)) - _onion(m1, m2))*bias*2/t3


def q2n(gt, fused, block=32):
    h, w, channels = gt.shape
    assert channels == 8
    pad_h, pad_w = (-h) % block, (-w) % block
    def prep(x):
        x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode='symmetric')
        return np.clip(np.floor(x + 0.5), 0, 65535).astype(np.uint16)
    a, b = prep(gt), prep(fused)
    return np.mean([np.linalg.norm(_onions_quality(a[i:i+block, j:j+block], b[i:i+block, j:j+block]))
                    for i in range(0, h, block) for j in range(0, w, block)])


def _qi(a, b):
    kernel = np.ones((32, 32))
    def sm(x): return fftconvolve(x, kernel, mode='valid')
    x, y = sm(a), sm(b)
    xx, yy, xy = sm(a*a), sm(b*b), sm(a*b)
    n = 1024
    mul = x*y; sm2 = x*x+y*y
    d1 = n*(xx+yy)-sm2; den = d1*sm2
    q = np.ones_like(den)
    mask = (d1 == 0) & (sm2 != 0)
    q[mask] = 2*mul[mask]/sm2[mask]
    mask = den != 0
    q[mask] = (4*(n*xy-mul)*mul)[mask]/den[mask]
    return q.mean()


def indexes_evaluation(fused, gt, ratio=4, cut=30, threshold=False):
    fused, gt = np.asarray(fused, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if cut:
        fused, gt = fused[cut-1:-cut, cut-1:-cut], gt[cut-1:-cut, cut-1:-cut]
    if threshold:
        fused = np.clip(fused, 0, 2048)
    q = q2n(gt, fused)
    qavg = np.mean([_qi(gt[..., i], fused[..., i]) for i in range(gt.shape[2])])
    dot = np.sum(gt*fused, axis=2)
    norms = np.sqrt(np.sum(gt*gt, axis=2)*np.sum(fused*fused, axis=2))
    mask = norms != 0
    sam = np.degrees(np.arccos(np.clip(dot[mask]/norms[mask], -1, 1))).mean()
    ergas = 100/ratio*np.sqrt(np.mean(np.mean((gt-fused)**2, axis=(0, 1))/gt.mean(axis=(0, 1))**2))
    sobel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=float)
    def grad(im):
        im = im[1:-1, 1:-1]
        return np.stack([np.hypot(correlate(im[..., i], sobel, mode='constant'),
                                  correlate(im[..., i], sobel.T, mode='constant'))
                         for i in range(im.shape[2])], axis=2)
    a, b = grad(gt), grad(fused)
    scc = np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))
    mse = np.mean((gt-fused)**2)
    psnr = 10*np.log10(2047**2/mse) if mse else float('inf')
    ssim = structural_similarity(gt, fused, data_range=2047, channel_axis=2)
    return dict(Q2n=q, Q_avg=qavg, SAM=sam, ERGAS=ergas, SCC=scc, PSNR=psnr, SSIM=ssim)
