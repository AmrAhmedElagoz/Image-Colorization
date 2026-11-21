import argparse
import base64
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from color_image.common import EvalDataset, gray_rgb, split_lab

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["psnr", "ssim", "lpips_vgg", "auc_ab", "auc_ab_balanced", "colorfulness", "cf_gap", "mean_chroma",
           "vivid_share"]
REFERENCE = ("gray", "ground_truth")


def to_jpeg_b64(img: Image.Image, size: int) -> str:
    buf = io.BytesIO()
    img.resize((size, size), Image.LANCZOS).save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def read_per_image(path: Path) -> tuple[dict[str, dict[str, list[float]]], list[str]]:
    per: dict[str, dict[str, list[float]]] = {}
    names: dict[str, list[str]] = {}
    for row in csv.DictReader(open(path)):
        m = row["model"]
        per.setdefault(m, {k: [] for k in METRICS})
        names.setdefault(m, []).append(row["image"])
        for k in METRICS:
            per[m][k].append(round(float(row[k]), 4))
    ref = names["gray"]
    if any(n != ref for n in names.values()):
        raise SystemExit("per_image.csv has a different image order per model")
    return per, ref


def pick_samples(per: dict, methods: list[str], k: int) -> list[int]:
    """Evenly spaced along the PSNR advantage of the 2nd method over the 1st (or the PSNR of the only method)."""
    psnr = np.array(per[methods[0]]["psnr"])
    key = np.array(per[methods[1]]["psnr"]) - psnr if len(methods) > 1 else psnr
    order = np.argsort(key)
    return sorted({int(order[int(q)]) for q in np.linspace(0, len(order) - 1, min(k, len(order)))})


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the interactive evaluation explorer (one HTML file).")
    ap.add_argument("--eval", type=Path, default=ROOT / "runs" / "eval", help="folder written by evaluate.py")
    ap.add_argument("--pred", nargs="+", required=True, help="the same [name=]dir arguments given to evaluate.py")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "coco")
    ap.add_argument("--samples", type=int, default=30, help="number of photos to embed")
    ap.add_argument("--thumb", type=int, default=176, help="thumbnail size in pixels")
    ap.add_argument("--out", type=Path, default=None, help="default: <eval>/explorer.html")
    args = ap.parse_args()

    summary = json.load(open(args.eval / "summary.json"))
    per, names = read_per_image(args.eval / "per_image.csv")
    preds = {}
    for p in args.pred:
        name, _, d = p.rpartition("=")
        preds[name or Path(d).name] = Path(d)
    missing = set(preds) ^ (set(per) - set(REFERENCE))
    if missing:
        raise SystemExit(f"--pred names and the methods in per_image.csv differ: {sorted(missing)}")

    ds = EvalDataset(args.data, summary["split"], summary["size"], limit=len(names))
    if ds.names != names:
        raise SystemExit("image order in per_image.csv does not match the split file")

    methods = list(preds)
    samples = []
    for i in pick_samples(per, methods, args.samples):
        gt = ds[i]["rgb"]
        to_pil = lambda t: Image.fromarray((t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        imgs = {"gray": to_pil(gray_rgb(split_lab(gt[None])[0])[0]), "ground_truth": to_pil(gt)}
        for m, d in preds.items():
            imgs[m] = Image.open(d / f"{Path(names[i]).stem}.png").convert("RGB")
        samples.append({"i": i, "name": names[i], "imgs": {k: to_jpeg_b64(v, args.thumb) for k, v in imgs.items()}})

    data = {"models": ["gray", *methods, "ground_truth"], "metrics": METRICS, "per": per, "n": len(names),
            "summary": summary["models"], "samples": samples, "split": summary["split"], "seed": summary["seed"],
            "boot": summary["bootstrap_resamples"]}
    template = (ROOT / "scripts" / "templates" / "eval_explorer.html").read_text()
    out = args.out or args.eval / "explorer.html"
    out.write_text(template.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(samples)} photos, methods: {', '.join(methods)})")


if __name__ == "__main__":
    main()
