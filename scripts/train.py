import argparse
from pathlib import Path

from color_image.config import ROOT, TrainConfig
from color_image.train import Trainer

GAN_DEFAULTS = dict(out=ROOT / "runs" / "phase_c", init=ROOT / "runs" / "phase_b" / "best.pt",
                    epochs=4, lr_lora=5e-5, lr_new=1e-4, patience=0)


def main() -> None:
    d = TrainConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan", action="store_true", help="GAN: add the adversarial loss")
    ap.add_argument("--data", type=Path, default=d.data)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--init", default=None, help="generator start weights ('none' = from scratch)")
    ap.add_argument("--clip-ckpt", default=str(d.clip_ckpt), help="distilled GrayCLIP weights, or 'none' for stock CLIP")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=d.batch_size)
    ap.add_argument("--accum", type=int, default=d.accum)
    ap.add_argument("--lr-lora", type=float, default=None)
    ap.add_argument("--lr-new", type=float, default=None)
    ap.add_argument("--lr-d", type=float, default=d.lr_d)
    ap.add_argument("--w-adv", type=float, default=d.w_adv)
    ap.add_argument("--d-warmup", type=int, default=d.d_warmup, help="optimiser steps of D-only training")
    ap.add_argument("--no-diffaug", action="store_true")
    ap.add_argument("--wd", type=float, default=d.wd)
    ap.add_argument("--warmup", type=int, default=d.warmup)
    ap.add_argument("--patience", type=int, default=None, help="0 = no early stopping")
    ap.add_argument("--min-delta", type=float, default=d.min_delta)
    ap.add_argument("--workers", type=int, default=d.workers)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--limit", type=int, default=None, help="first N train images")
    ap.add_argument("--val-limit", type=int, default=None, help="first N val images")
    ap.add_argument("--resume", type=Path, default=None)
    a = ap.parse_args()

    base = GAN_DEFAULTS if a.gan else dict(out=d.out, init=d.init, epochs=d.epochs, lr_lora=d.lr_lora,
                                           lr_new=d.lr_new, patience=d.patience)
    pick = lambda name: getattr(a, name) if getattr(a, name) is not None else base[name]
    init = pick("init")
    if isinstance(init, str):
        init = None if init.lower() == "none" else Path(init)

    Trainer(TrainConfig(
        data=a.data, out=pick("out"), clip_ckpt=None if a.clip_ckpt.lower() == "none" else Path(a.clip_ckpt),
        epochs=pick("epochs"), batch_size=a.bs, accum=a.accum, lr_lora=pick("lr_lora"), lr_new=pick("lr_new"),
        wd=a.wd, warmup=a.warmup, patience=pick("patience"), min_delta=a.min_delta, workers=a.workers,
        seed=a.seed, limit=a.limit, val_limit=a.val_limit, resume=a.resume,
        gan=a.gan, init=init, lr_d=a.lr_d, w_adv=a.w_adv, d_warmup=a.d_warmup, diffaug=not a.no_diffaug)).fit()


if __name__ == "__main__":
    main()
