"""Train DeRainNet; datasets can be preloaded onto the chosen device."""
import argparse
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import TestDatasetMat, TrainValDatasetH5
from metrics import calc_psnr, calc_ssim
from model import DeRainNet

ROOT = Path(__file__).resolve().parent


def device_auto():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def load_data(device, resident=True):
    datasets = [TrainValDatasetH5(ROOT/'derain_small/Rain200L_train_chunks.h5'),
                TrainValDatasetH5(ROOT/'derain_small/Rain200L_val_chunks.h5'),
                TestDatasetMat(ROOT/'derain_small/test12_chunks.mat')]
    target = device if resident else torch.device('cpu')
    return [(d.inputs.to(target), d.targets.to(target)) for d in datasets]


@torch.inference_mode()
def evaluate(model, data, device, batch_size=8):
    model.eval()
    x, y = data
    totals = torch.zeros(3, device=device)
    for i in range(0, len(x), batch_size):
        a, b = x[i:i+batch_size].to(device), y[i:i+batch_size].to(device)
        pred = model(a)
        totals += torch.stack((F.l1_loss(pred, b), calc_psnr(pred, b), calc_ssim(pred, b))) * len(a)
    return (totals / len(x)).cpu().tolist()


def save_history(history):
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/training_history.json').write_text(json.dumps(history, indent=2)+'\n')
    with (ROOT/'results/training_history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='auto')
    p.add_argument('--no-resident', action='store_true')
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = device_auto() if args.device == 'auto' else torch.device(args.device)
    print(f'device={device} resident={not args.no_resident}', flush=True)
    train, val, test = load_data(device, not args.no_resident)
    model = DeRainNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-5)
    (ROOT/'checkpoints').mkdir(exist_ok=True)
    history, best = [], float('inf')
    x, y = train
    for epoch in range(1, args.epochs+1):
        start = time.time(); model.train(); total = torch.zeros((), device=device)
        for ids in torch.randperm(len(x), device=x.device).split(args.batch_size):
            a, b = x[ids].to(device), y[ids].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.l1_loss(model(a), b); loss.backward(); opt.step()
            total += loss.detach()*len(ids)
        vl, vp, vs = evaluate(model, val, device)
        _, tp, ts = evaluate(model, test, device, 1)
        row = dict(epoch=epoch, lr=opt.param_groups[0]['lr'], train_loss=total.item()/len(x), val_loss=vl,
                   val_psnr=vp, val_ssim=vs, test_psnr=tp, test_ssim=ts, seconds=time.time()-start)
        history.append(row); scheduler.step()
        state = {'model': model.state_dict(), 'epoch': epoch, 'val_loss': vl, 'seed': args.seed}
        torch.save(state, ROOT/'checkpoints/last_model.pth')
        if vl < best:
            best = vl; torch.save(state, ROOT/'checkpoints/best_model.pth')
        save_history(history)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
