import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from color_image.config import ROOT, LossConfig
from color_image.data import ColorizationDataset
from color_image.losses import compute_prior


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "coco")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=LossConfig().prior)
    a = ap.parse_args()

    dl = DataLoader(ColorizationDataset(a.data, "train", 256, limit=a.n), batch_size=32, num_workers=a.workers)
    p = compute_prior(dl)
    np.save(a.out, p)
    print(f"saved {a.out}  | bins used {int((p > 0).sum())}/313 | top-5 share {np.sort(p)[-5:].sum():.2f}")


if __name__ == "__main__":
    main()
