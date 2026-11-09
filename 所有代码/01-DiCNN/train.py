import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import WVDataset
from model import DiCNN

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; explicitly select --device cpu if desired')
    device = torch.device(args.device)
    train = DataLoader(WVDataset(ROOT / 'pansharpening_data/train_small.h5'), batch_size=args.batch_size, shuffle=True)
    valid = DataLoader(WVDataset(ROOT / 'pansharpening_data/valid_small.h5'), batch_size=args.batch_size)
    model = DiCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    criterion = nn.MSELoss()
    out = Path(__file__).resolve().parent / 'training'
    out.mkdir(exist_ok=True)
    best = float('inf')
    with (out / 'log.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'lr', 'train_mse', 'valid_mse', 'seconds'])
        for epoch in range(1, args.epochs + 1):
            start = time.monotonic()
            model.train()
            total = 0.0
            lr = optimizer.param_groups[0]['lr']
            for lms, pan, gt in train:
                lms, pan, gt = lms.to(device), pan.to(device), gt.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(lms, pan), gt)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(gt)
            model.eval()
            val = 0.0
            with torch.no_grad():
                for lms, pan, gt in valid:
                    lms, pan, gt = lms.to(device), pan.to(device), gt.to(device)
                    val += criterion(model(lms, pan), gt).item() * len(gt)
            train_loss, val_loss = total / len(train.dataset), val / len(valid.dataset)
            scheduler.step()
            checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'val_mse': val_loss, 'args': vars(args)}
            if val_loss < best:
                best = val_loss
                torch.save(checkpoint, out / 'best.pth')
            writer.writerow([epoch, lr, train_loss, val_loss, time.monotonic() - start])
            file.flush()
            print(f'{epoch}/{args.epochs} lr={lr:.8g} train={train_loss:.8g} val={val_loss:.8g}', flush=True)
        torch.save(checkpoint, out / 'last.pth')


if __name__ == '__main__':
    main()
