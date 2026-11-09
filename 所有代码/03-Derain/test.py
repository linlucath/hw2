"""Evaluate the best checkpoint and create assignment and report figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from data import TestDatasetMat, TrainValDatasetH5
from metrics import calc_mae, calc_psnr, calc_ssim
from model import DeRainNet
from train import ROOT, device_auto

OUT = ROOT/'results'


def scores(a, b):
    return {c: {name: float(fn(a, b, c).cpu()) for name, fn in
                [('psnr', calc_psnr), ('ssim', calc_ssim), ('mae', calc_mae)]} for c in ('RGB', 'Y')}


def mean_scores(items):
    return {c: {m: float(np.mean([v[c][m] for v in items])) for m in ('psnr', 'ssim', 'mae')} for c in ('RGB', 'Y')}


def image(t):
    return t.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()


def save_fig(fig, path):
    fig.savefig(path, dpi=180, bbox_inches='tight'); plt.close(fig)


def plots(inputs, outputs, rains, targets, test_scores):
    for i, (a, b, gt) in enumerate(zip(inputs, outputs, targets), 1):
        raw = np.concatenate([image(a), image(b), image(gt)], axis=1)
        png = Image.fromarray(np.rint(raw*255).astype(np.uint8))
        png.save(ROOT/f'Derain的结果{i}.png'); png.save(OUT/f'Derain的结果{i}.png')
        png.save(OUT/f'test_{i}_raw_side_by_side.png')
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, im, title in zip(axes, (a,b,gt), ('Input', 'Derained', 'Ground truth')):
            ax.imshow(image(im)); ax.set_title(title); ax.axis('off')
        before, after = test_scores[i-1]['before']['RGB'], test_scores[i-1]['after']['RGB']
        fig.suptitle(f"Test {i}  |  RGB PSNR {before['psnr']:.2f} → {after['psnr']:.2f} dB  |  SSIM {before['ssim']:.4f} → {after['ssim']:.4f}")
        save_fig(fig, OUT/f'test_{i}_comparison.png')
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for i in range(3):
        # Rain residual is signed: center zero at mid-gray for honest visualization.
        images = [image(inputs[i]), image(outputs[i]), np.clip(.5 + rains[i].cpu().permute(1,2,0).numpy(),0,1), image(targets[i])]
        for j, (ax, im) in enumerate(zip(axes[i], images)):
            ax.imshow(im); ax.axis('off')
            if i == 0: ax.set_title(['Input', 'Derained', 'Predicted rain (zero=gray)', 'Ground truth'][j])
    save_fig(fig, OUT/'all_test_results.png')
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for i in range(3):
        # Fixed, reproducible center crop, 120x160.
        for j, im in enumerate((inputs[i], outputs[i], targets[i])):
            axes[i,j].imshow(image(im)[100:220,160:320]); axes[i,j].axis('off')
            if i == 0: axes[i,j].set_title(['Input', 'Derained', 'Ground truth'][j])
    fig.suptitle('Center crops (120 × 160 pixels)'); save_fig(fig, OUT/'detail_zoom_comparison.png')
    # Histories are optional; they are not included in the code-only submission.
    if not (OUT/'training_history.json').exists() or not (OUT/'ablation_results.json').exists():
        print('Training or ablation history unavailable; skipping history plots.')
        return
    hist = json.loads((OUT/'training_history.json').read_text())
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for key,label in [('train_loss','Train'),('val_loss','Validation')]: axes[0].plot([r['epoch'] for r in hist],[r[key] for r in hist],label=label)
    for ax, metric in zip(axes[1:],('psnr','ssim')):
        for split in ('val','test'):
            ax.plot([r['epoch'] for r in hist],[r[f'{split}_{metric}'] for r in hist],label=split.title())
    for ax,title in zip(axes,('L1 loss','RGB PSNR (dB)','RGB SSIM')):
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); save_fig(fig,OUT/'training_curves.png')
    ab = json.loads((OUT/'ablation_results.json').read_text())
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, metric in zip(axes,('psnr','ssim')):
        names = [r['config'] for r in ab]; vals = [r['test'][metric] for r in ab]
        bars = ax.bar(names, vals, color=['#457b9d','#e9c46a','#e76f51']); ax.bar_label(bars, fmt='%.3f', padding=3)
        ax.set_ylim(min(vals)*.95,max(vals)*1.02); ax.set_title(f'Test RGB {metric.upper()}'); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Previous ablation (25 epochs, batch 64; different branch skip architecture)')
    fig.tight_layout(); save_fig(fig, OUT/'ablation_comparison.png')


def main():
    OUT.mkdir(exist_ok=True); device = device_auto()
    model = DeRainNet().to(device)
    checkpoint = torch.load(ROOT/'checkpoints/best_model.pth', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model']); model.eval()
    val = TrainValDatasetH5(ROOT/'derain_small/Rain200L_val_chunks.h5')
    test = TestDatasetMat(ROOT/'derain_small/test12_chunks.mat')
    result = {'checkpoint_epoch': checkpoint['epoch'], 'metric_definition': 'Clipped [0,1]; RGB or full-range BT.601 Y; SSIM Gaussian 11x11 sigma 1.5 valid; per-image mean', 'validation': {}, 'test': {}}
    with torch.inference_mode():
        for name, dataset in [('validation', val), ('test', test)]:
            before, after, per_image = [], [], []
            outputs, rains = [], []
            for a, b in zip(dataset.inputs, dataset.targets):
                a, b = a.unsqueeze(0).to(device), b.unsqueeze(0).to(device)
                pred, rain = model(a, return_residual=True)
                pre, post = scores(a,b), scores(pred,b)
                before.append(pre); after.append(post); per_image.append({'before':pre, 'after':post})
                if name == 'test': outputs.append(pred[0].cpu()); rains.append(rain[0].cpu())
            result[name] = {'before':mean_scores(before), 'after':mean_scores(after)}
            if name == 'test':
                result[name]['images'] = per_image
                plots(test.inputs, outputs, rains, test.targets, per_image)
    (OUT/'test_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
