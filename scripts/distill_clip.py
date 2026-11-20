import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from color_image.config import ROOT, ModelConfig
from color_image.data import ColorizationDataset
from color_image.model import GrayCLIPDistiller


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "coco")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "clip" / "gray_clip.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N images per split")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    kw = dict(batch_size=a.bs, num_workers=a.workers, pin_memory=True)
    train_dl = DataLoader(ColorizationDataset(a.data, "train", 256, train=True, limit=a.limit),
                          shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(ColorizationDataset(a.data, "val", 256, limit=a.limit), **kw)

    GrayCLIPDistiller(ModelConfig().clip_id).fit(train_dl, val_dl, a.out, a.epochs, a.lr, a.wd, a.warmup,
                                                  a.patience or None, a.min_delta)


if __name__ == "__main__":
    main()
