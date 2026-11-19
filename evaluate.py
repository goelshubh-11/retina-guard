"""
evaluate.py - Retina Guard
Evaluate a trained checkpoint on the same 20% validation split train.py used.

    python evaluate.py --data /kaggle/input --weights retina_b3.pth

Prints accuracy, quadratic weighted kappa, per-grade precision/recall and a
confusion matrix, and saves confusion_matrix.png and predictions.csv.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from app.model import build_model, DEVICE, CLASS_NAMES
from train import Aptos, find_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--no-ben-graham", action="store_true")
    args = ap.parse_args()

    csv, img_dir = find_data(Path(args.data))
    df = pd.read_csv(csv)
    df["path"] = df["id_code"].map(lambda i: img_dir / f"{i}.png")
    # Must match train.py exactly, otherwise images the model trained on leak in.
    _, va = train_test_split(df, test_size=0.2, random_state=42, stratify=df["diagnosis"])

    model = build_model(pretrained=False, weights_path=args.weights)
    # build_model loads with strict=False; reload strictly so a mismatch cannot pass silently.
    state = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model.eval()

    dl = DataLoader(Aptos(va.path, va.diagnosis, False, not args.no_ben_graham),
                    args.batch, num_workers=args.workers)
    preds, truth = [], []
    with torch.no_grad():
        for x, y in dl:
            preds += model(x.to(DEVICE)).argmax(1).cpu().tolist()
            truth += y.tolist()
    preds, truth = np.array(preds), np.array(truth)

    cm = confusion_matrix(truth, preds, labels=range(5))
    print(f"\nimages: {len(truth)}")
    print(f"accuracy:            {(preds == truth).mean():.3f}")
    print(f"within one grade:    {(abs(preds - truth) <= 1).mean():.3f}")
    print(f"quadratic kappa:     {cohen_kappa_score(truth, preds, weights='quadratic'):.3f}\n")
    print(classification_report(truth, preds, labels=range(5), target_names=CLASS_NAMES, digits=3))
    print("confusion matrix (rows = true grade, columns = predicted):")
    print(cm)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id_code": va.id_code.values, "true": truth, "pred": preds}).to_csv(
        out / "predictions.csv", index=False)

    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(5), CLASS_NAMES)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title("Confusion matrix (validation set)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=200)
    print(f"\nsaved {out / 'confusion_matrix.png'} and {out / 'predictions.csv'}")


if __name__ == "__main__":
    main()
