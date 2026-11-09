"""Evaluate all five methods on the 20 WV3 samples and create report assets."""
import importlib.util
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pansharpening_metrics import indexes_evaluation

ROOT = Path(__file__).resolve().parent
DICNN = ROOT / '01-DiCNN'
FUSION = ROOT / '02-FusionNet'
OUT = ROOT / 'pansharpening_results'


def load_class(folder, name):
    spec = importlib.util.spec_from_file_location(name, folder / 'model.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return getattr(module, name)


def models(device):
    result = {}
    for folder, cls, name, official in [(DICNN, 'DiCNN', 'DiCNN', 'dicnn1.pth'), (FUSION, 'FusionNet', 'FusionNet', 'fusionnet.pth')]:
        klass = load_class(folder, cls)
        ckpt = torch.load(folder / 'checkpoints/best_model.pth', map_location='cpu', weights_only=False)
        model = klass().to(device); model.load_state_dict(ckpt['model']); model.eval()
        result[name + ' (self-trained)'] = model
        model = (klass(hidden_channels=64) if cls == 'DiCNN' else klass()).to(device)
        state = torch.load(folder / official, map_location='cpu', weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith('backbone.')}
        model.load_state_dict(state); model.eval()
        result[name + ' (DLPan)'] = model
    return result


def rgb(a):
    # WV3: red/green/blue bands 5/3/2, 1-based.
    x = a[..., [4, 2, 1]]
    lo, hi = np.percentile(x, (2, 98))
    return np.clip((x-lo)/(hi-lo+1e-12), 0, 1)


def plot_metrics(summary, path):
    from matplotlib.patches import Patch
    names = ['LMS', 'DiCNN (self-trained)', 'FusionNet (self-trained)']
    colors = ['#95a5a6', '#2878b5', '#e79037']
    metrics = [('Q2n', '↑'), ('Q_avg', '↑'), ('SAM', '↓'), ('ERGAS', '↓'), ('SCC', '↑')]
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.3))
    for ax, (metric, direction) in zip(axes, metrics):
        vals = [summary.loc[name, metric + ' Mean'] for name in names]
        bars = ax.bar(range(3), vals, color=colors, width=0.68, zorder=3)
        ax.bar_label(bars, labels=[f'{v:.4f}' for v in vals], padding=4, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_title(f'{metric} ({direction})', fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.grid(axis='y', alpha=.2, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(colors, ['LMS', 'DiCNN', 'FusionNet'])],
               loc='lower center', ncol=3, frameon=False)
    fig.suptitle('WV3 test set: mean quality metrics', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, .08, 1, .92)); fig.savefig(path, dpi=240); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    nets = models(device)
    rows = []; images = None
    with h5py.File(ROOT / 'pansharpening_data/test_wv3_multiExm1.h5') as f:
        with torch.no_grad():
            for i in range(20):
                lms = np.asarray(f['lms'][i], np.float32)/2047
                pan = np.asarray(f['pan'][i], np.float32)/2047
                gt = np.moveaxis(np.asarray(f['gt'][i], np.float64), 0, -1)
                outputs = {'LMS': np.moveaxis(lms, 0, -1)*2047}
                for name, model in nets.items():
                    pred = model(torch.from_numpy(lms[None]).to(device), torch.from_numpy(pan[None]).to(device))
                    outputs[name] = np.moveaxis(pred.cpu().numpy()[0], 0, -1)*2047
                for name, pred in outputs.items():
                    metrics = indexes_evaluation(pred, gt)
                    rows.append({'Image': i+1, 'Method': name, **metrics})
                    print(i+1, name, metrics, flush=True)
                if i == 0:
                    images = (pan[0], gt, outputs)
    df = pd.DataFrame(rows)
    summary = df.groupby('Method', sort=False).agg({x:['mean', 'std'] for x in ['Q2n','Q_avg','SAM','ERGAS','SCC','PSNR','SSIM']})
    summary.columns = [f'{a} {b.title()}' for a,b in summary.columns]
    with pd.ExcelWriter(ROOT / 'pansharpening的结果.xlsx', engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='Summary_指标汇总')
        df.to_excel(writer, sheet_name='Per_Image_20_Test_Samples', index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes = 'B2'
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width = 19
    if all((folder / 'training/log.csv').exists() for folder in (DICNN, FUSION)):
        fig, ax = plt.subplots(1,2,figsize=(12,4))
        for a, folder, name in zip(ax, (DICNN,FUSION), ('DiCNN','FusionNet')):
            log = pd.read_csv(folder / 'training/log.csv')
            a.plot(log.epoch, log.train_mse, label='Train'); a.plot(log.epoch, log.valid_mse, label='Validation')
            a.set(title=name, xlabel='Epoch', ylabel='MSE', yscale='log'); a.legend(); a.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(OUT/'pansharpening_training_curves.png', dpi=240); plt.close(fig)
    else:
        print('Training histories unavailable; skipping training curves.', flush=True)
    pan, gt, o = images
    names = ['PAN','LMS','DiCNN (self-trained)','FusionNet (self-trained)','GT']
    vals = [pan, o['LMS'], o[names[2]], o[names[3]], gt]
    fig, ax = plt.subplots(2,5,figsize=(16,6))
    vmax = np.percentile(np.abs(o['LMS']-gt).mean(axis=2), 99)
    for j,(name, val) in enumerate(zip(names,vals)):
        ax[0,j].imshow(val, cmap='gray' if j==0 else None) if j==0 else ax[0,j].imshow(rgb(val))
        ax[0,j].set_title(name)
        ax[1,j].axis('off')
        if j in (1,2,3):
            ax[1,j].imshow(np.abs(val-gt).mean(axis=2), cmap='inferno', vmin=0, vmax=vmax)
            ax[1,j].set_title('Mean |SR - GT| (DN)')
        ax[0,j].axis('off')
    fig.tight_layout(); fig.savefig(OUT/'pansharpening_visual_comparison.png', dpi=240); plt.close(fig)
    plot_metrics(summary, OUT / 'pansharpening_metrics_chart.png')


if __name__ == '__main__':
    main()
