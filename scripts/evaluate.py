import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from color_image import metrics
from color_image.common import EvalDataset, gray_rgb, split_lab

ROOT = Path(__file__).resolve().parents[1]


def load_pngs(folder: Path, names: list[str]) -> torch.Tensor:
    arr = [np.asarray(Image.open(folder / f"{Path(n).stem}.png").convert("RGB")) for n in names]
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).float() / 255


def save_pngs(rgb: torch.Tensor, folder: Path, names: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    arr = (rgb.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    for a, n in zip(arr, names):
        Image.fromarray(a).save(folder / f"{Path(n).stem}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score saved predictions against the ground truth of a split.")
    ap.add_argument("--pred", nargs="+", required=True, help="prediction folders of <stem>.png, as [name=]dir")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "coco")
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n", type=int, default=0, help="0 = whole split")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "eval")
    ap.add_argument("--realism", action="store_true",
                    help="also compute FID and KID (downloads the Inception weights once)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    preds = {}
    for p in args.pred:
        name, _, d = p.rpartition("=")
        preds[name or Path(d).name] = Path(d)

    ds = EvalDataset(args.data, args.split, args.size, limit=args.n or None)
    net = metrics.make_lpips(device)
    prior = metrics.ab_prior(args.data, args.size, cache=args.data / f"ab_prior_{args.size}.npy", device=device)
    weights = metrics.bin_weights(prior)

    scores: dict[str, dict[str, list[np.ndarray]]] = {}
    names: list[str] = []
    gt_dir, gray_dir = args.out / f"gt_{args.split}", args.out / f"gray_{args.split}"
    for batch in DataLoader(ds, batch_size=args.batch_size):
        gt = batch["rgb"].to(device)
        names += batch["name"]
        methods = {"gray": gray_rgb(split_lab(gt)[0])}
        methods |= {m: load_pngs(d, batch["name"]).to(device) for m, d in preds.items()}
        if args.realism:
            save_pngs(gt, gt_dir, batch["name"])
            save_pngs(methods["gray"], gray_dir, batch["name"])
        with torch.no_grad():
            for m, pred in methods.items():
                for k, v in metrics.score(pred, gt, net, weights).items():
                    scores.setdefault(m, {}).setdefault(k, []).append(v.cpu().numpy())
            for k, v in metrics.score(gt, gt, net, weights).items():
                scores.setdefault("ground_truth", {}).setdefault(k, []).append(v.cpu().numpy())
    per = {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in scores.items()}

    with open(args.out / "per_image.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = list(next(iter(per.values())))
        w.writerow(["model", "image", *keys])
        for m, d in per.items():
            for i, n in enumerate(names):
                w.writerow([m, n, *(f"{d[k][i]:.6f}" for k in keys)])

    summary: dict = {"split": args.split, "size": args.size, "images": len(names), "seed": args.seed,
                     "bootstrap_resamples": args.boot, "models": {}, "paired_diff": {}}
    for m, d in per.items():
        summary["models"][m] = {k: dict(zip(("mean", "lo", "hi"), metrics.bootstrap_ci(v, args.boot, seed=args.seed)))
                                for k, v in d.items()}
    for a, b in itertools.combinations(preds, 2):
        summary["paired_diff"][f"{a} - {b}"] = {
            k: dict(zip(("mean", "lo", "hi"), metrics.paired_diff_ci(per[a][k], per[b][k], n=args.boot, seed=args.seed)))
            for k in per[a]}
    if args.realism:
        real = metrics.inception_features(gt_dir, device)
        for m, d in {"gray": gray_dir, **preds}.items():
            feats = metrics.inception_features(d, device)
            summary["models"][m]["fid"] = metrics.fid(real, feats)
            summary["models"][m]["kid"] = dict(zip(("mean", "std"), metrics.kid(real, feats, seed=args.seed)))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    def table(title: str, cols: tuple[str, ...]) -> None:
        print(f"\n{title}")
        print(f"{'model':14s}" + "".join(f"{c:>18s}" for c in cols))
        for m, r in summary["models"].items():
            print(f"{m:14s}" + "".join(f"{r[c]['mean']:9.3f} ±{(r[c]['hi'] - r[c]['lo']) / 2:6.3f}  " for c in cols))

    print(f"{len(names)} {args.split} images, {args.boot} bootstrap resamples (95% CI, ± = half-width)")
    table("Colour (primary; cf_gap = prediction - truth, its absolute mean is the colourfulness gap)",
          ("auc_ab_balanced", "cf_gap", "vivid_share", "mean_chroma", "colorfulness"))
    table("Closeness to the ground truth (reference; auc_ab is secondary: gray scores well on it)",
          ("psnr", "ssim", "lpips_vgg", "auc_ab"))
    if args.realism:
        print(f"\nRealism (N={len(names)}: biased, compare only with each other; KID is mean ± std over subsets)")
        print(f"{'model':14s}{'fid':>10s}{'kid':>20s}")
        for m, r in summary["models"].items():
            if "fid" in r:
                print(f"{m:14s}{r['fid']:10.2f}{r['kid']['mean']:12.4f} ±{r['kid']['std']:.4f}")
    print(f"saved {args.out / 'summary.json'} and {args.out / 'per_image.csv'}")


if __name__ == "__main__":
    main()
