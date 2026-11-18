from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections.abc import Callable

import lpips
import time
from tqdm import tqdm
import json

from .config import Config
from .common import EvalDataset, psnr, split_lab, compose, auc_ab, colorfulness, gray_rgb

matplotlib.use("Agg")

import colorizers  # type: ignore # noqa: E402 

Predictor = Callable[[torch.Tensor], torch.Tensor]


class Baseline:

    def __init__(self, conf: Config):
        self.conf = conf
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ds = EvalDataset(conf.data, conf.split, conf.size, limit=conf.n or None)
        self.lpips = lpips.LPIPS(net="vgg", verbose=False).eval().requires_grad_(False).to(self.device)
        self.metrics: dict[str, dict] = {}
        self.grid: dict[str, torch.Tensor] = {}


    def base1(self) -> dict[str, Predictor]:

        wanted = [m for m in ("eccv16", "siggraph17") if m in self.conf.model_list]
        nets = {m: getattr(colorizers, m)(pretrained=True).eval().to(self.device) for m in wanted}

        return {m: torch.no_grad()(lambda rgb, net=net: net(split_lab(rgb)[0])) for m, net in nets.items()}

    def base2(self):
        ...


    def evaluate(self, name: str, predict: Predictor) -> dict:
        img_dir = self.conf.out / name

        if self.conf.save_images:
            img_dir.mkdir(parents=True, exist_ok=True)

        loader = DataLoader(self.ds, batch_size=self.conf.batch_size, num_workers=self.conf.workers)
        scores: dict[str, list[torch.Tensor]] = {k: [] for k in ("psnr", "lpips_vgg", "auc_ab", "colorfulness",
                                                                 "colorfulness_gt")}
        rows: list[torch.Tensor] = []
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        with torch.no_grad():
            for batch in tqdm(loader, desc=name):
                rgb = batch["rgb"].to(self.device)
                L, ab_gt = split_lab(rgb)
                ab = predict(rgb).float()
                pred = compose(L, ab)
                scores["psnr"].append(psnr(pred, rgb))
                scores["lpips_vgg"].append(self.lpips(pred * 2 - 1, rgb * 2 - 1).flatten())
                scores["auc_ab"].append(auc_ab(ab, ab_gt))
                scores["colorfulness"].append(colorfulness(pred))
                scores["colorfulness_gt"].append(colorfulness(rgb))

                if sum(map(len, rows)) < self.conf.grid_rows:
                    rows.append(pred.cpu())
                if self.conf.save_images:
                    arr = (pred.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
                    for a, fname in zip(arr, batch["name"]):
                        Image.fromarray(a).save(img_dir / f"{Path(fname).stem}.png")

        res = {k: torch.cat(v).mean().item() for k, v in scores.items()}
        res.update(images=len(self.ds), split=self.conf.split, size=self.conf.size,
                   sec_per_image=(time.time() - t0) / len(self.ds))

        self.metrics[name] = res
        self.grid[name] = torch.cat(rows)[: self.conf.grid_rows]
        return res

    def save_metrics(self) -> Path:
        path = self.conf.out / "metrics.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        path.write_text(json.dumps({**old, **self.metrics}, indent=2))
        return path

    def save_grid(self) -> Path:
        TITLES = {"eccv16": "Zhang ECCV16", "siggraph17": "Zhang SIGGRAPH17", "sd_lora": "SD-LoRA"}

        n = min(self.conf.grid_rows, len(self.ds))
        gt = torch.stack([self.ds[i]["rgb"] for i in range(n)])
        cols = [("Grayscale input", gray_rgb(split_lab(gt)[0]))]
        cols += [(TITLES[m], imgs[:n]) for m, imgs in self.grid.items()]
        cols += [("Ground truth", gt)]

        s, g, head = self.conf.size, 6, 34
        canvas = Image.new("RGB", (len(cols) * s + (len(cols) - 1) * g, head + n * s + (n - 1) * g), "white")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        for c, (title, imgs) in enumerate(cols):
            x = c * (s + g)
            draw.text((x + (s - draw.textlength(title, font=font)) / 2, 6), title, fill="black", font=font)
            for r, img in enumerate(imgs):
                arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                canvas.paste(Image.fromarray(arr), (x, head + r * (s + g)))
        path = self.conf.out / "grid.jpg"
        canvas.save(path, quality=92)
        return path

def baseline():
    conf= Config()
    conf.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(conf.seed)
    b = Baseline(conf)
    print(f"{len(b.ds)} {conf.split} images at {conf.size}x{conf.size}, device {b.device}")

    groups = [(b.base1, {"eccv16", "siggraph17"})]
    for build, names in groups:
        if names & set(conf.model_list):
            for name, predict in build().items():
                b.evaluate(name, predict)
            del predict
            torch.cuda.empty_cache()

    print(f"\n{'model':12s} {'PSNR':>7s} {'LPIPS':>7s} {'AuC':>7s} {'CF':>7s} {'CF gt':>7s} {'s/img':>7s}")
    for name, r in b.metrics.items():
        print(f"{name:12s} {r['psnr']:7.2f} {r['lpips_vgg']:7.4f} {100 * r['auc_ab']:6.2f}% "
              f"{r['colorfulness']:7.2f} {r['colorfulness_gt']:7.2f} {r['sec_per_image']:7.3f}")
    print(f"saved {b.save_metrics()} and {b.save_grid()}")
    if conf.save_images:
        print(f"predictions in {conf.out}/<model>/")

