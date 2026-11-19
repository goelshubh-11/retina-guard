"""
train.py - Retina Guard
Fine-tune EfficientNet-B3 on the APTOS 2019 Blindness Detection dataset.

    python train.py --data /path/to/aptos --epochs 12 --out retina_b3.pth

--data must contain train.csv (id_code, diagnosis) and a train_images/ folder.
The saved file loads directly in the app sidebar ("Model weights").
Use --limit 64 --epochs 1 for a quick smoke test.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from app.preprocessing import full_preprocess
from app.model import build_model, DEVICE, IMAGENET_MEAN, IMAGENET_STD


class Aptos(Dataset):
    """Applies the same preprocessing the app uses at inference time."""

    def __init__(self, paths, labels, train: bool, ben_graham: bool):
        self.paths, self.labels, self.ben = list(paths), list(labels), ben_graham
        aug = [
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomRotation(30), T.ColorJitter(brightness=0.2, contrast=0.2),
        ] if train else []
        self.tf = T.Compose(aug + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # shortcut: full_preprocess also builds an unused "original" copy, upgrade if CPU-bound
        _, processed = full_preprocess(Image.open(self.paths[i]), apply_ben_graham=self.ben)
        return self.tf(Image.fromarray(processed)), self.labels[i]


def find_data(root: Path):
    csvs = list(root.rglob("train.csv"))
    if not csvs:
        raise SystemExit(f"No train.csv found under {root}")
    img_dir = csvs[0].parent / "train_images"
    if not img_dir.is_dir():
        raise SystemExit(f"Expected an images folder at {img_dir}")
    return csvs[0], img_dir


def evaluate(model, loader):
    model.eval()
    preds, truth = [], []
    with torch.no_grad():
        for x, y in loader:
            preds += model(x.to(DEVICE)).argmax(1).cpu().tolist()
            truth += y.tolist()
    acc = sum(p == t for p, t in zip(preds, truth)) / len(truth)
    return acc, cohen_kappa_score(truth, preds, weights="quadratic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="retina_b3.pth")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="use only N images (smoke test)")
    ap.add_argument("--no-ben-graham", action="store_true")
    args = ap.parse_args()

    csv, img_dir = find_data(Path(args.data))
    df = pd.read_csv(csv)
    if args.limit:
        df = df.sample(args.limit, random_state=0)
    df["path"] = df["id_code"].map(lambda i: img_dir / f"{i}.png")

    tr, va = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=None if args.limit else df["diagnosis"],
    )
    ben = not args.no_ben_graham
    tr_dl = DataLoader(Aptos(tr.path, tr.diagnosis, True, ben), args.batch,
                       shuffle=True, num_workers=args.workers)
    va_dl = DataLoader(Aptos(va.path, va.diagnosis, False, ben), args.batch,
                       num_workers=args.workers)

    # Mild inverse-frequency weights: APTOS is about half "No DR".
    counts = tr.diagnosis.value_counts().reindex(range(5), fill_value=1).clip(lower=1)
    w = (1 / counts) ** 0.5
    weights = torch.tensor((w / w.mean()).values, dtype=torch.float32, device=DEVICE)

    model = build_model(pretrained=True)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    cuda = DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=cuda)

    print(f"device={DEVICE}  train={len(tr)}  val={len(va)}")
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=cuda):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * len(y)
        sched.step()
        acc, kappa = evaluate(model, va_dl)
        print(f"epoch {ep:02d}/{args.epochs}  loss {run / len(tr):.4f}  "
              f"val acc {acc:.3f}  val QWK {kappa:.3f}  ({time.time() - t0:.0f}s)")
        if kappa > best:
            best = kappa
            # float(): a NumPy scalar here makes torch>=2.6 refuse to load the file by default
            torch.save({"model_state_dict": model.state_dict(), "kappa": float(kappa)}, args.out)
            print(f"  saved {args.out}")
    print(f"best validation QWK: {best:.3f}")


if __name__ == "__main__":
    main()
