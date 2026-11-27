import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from .common import colorfulness, compose, gray_rgb, psnr
from .config import LossConfig, ModelConfig, TrainConfig
from .data import ColorizationDataset
from .disc import DinoDiscriminator, diff_augment
from .losses import ColorizationLoss, d_hinge, g_hinge
from .metrics import lpips_vgg, make_lpips, ssim_rgb
from .model import SDUNet


class Trainer:
    """
    gan=False: SDUNet trained with L1 + LPIPS + rebalanced colour CE.
    gan=True:  same losses + w_adv * hinge GAN loss from a DINOv2 discriminator, generator usually started from the SDUNet weights (conf.init).
    """

    def __init__(self, conf: TrainConfig):
        self.conf = conf
        self.dev = "cuda"
        torch.manual_seed(conf.seed)
        if conf.clip_ckpt is not None and not Path(conf.clip_ckpt).exists():
            raise FileNotFoundError(f"{conf.clip_ckpt} not found: run scripts/distill_clip.py or pass --clip-ckpt none")

        self.model = SDUNet(ModelConfig(clip_ckpt=conf.clip_ckpt)).to(self.dev)
        if conf.init is not None:
            self.model.load(conf.init)
            print(f"generator initialised from {conf.init}")

        self.crit = ColorizationLoss(LossConfig()).to(self.dev)
        self.lpips_eval = make_lpips(self.dev)               # vgg, as in the evaluation

        kw = dict(num_workers=conf.workers, pin_memory=True)
        train_ds = ColorizationDataset(conf.data, "train", conf.size, train=True, limit=conf.limit)
        self.val_ds = ColorizationDataset(conf.data, "val", conf.size, limit=conf.val_limit)
        self.train_dl = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True, drop_last=True, **kw)
        self.val_dl = DataLoader(self.val_ds, batch_size=conf.batch_size, **kw)

        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        lora = [p for n, p in named if "lora_" in n]
        new = [p for n, p in named if "lora_" not in n]       # skip convs + colour head
        self.params = lora + new
        self.opt = torch.optim.AdamW(
            [{"params": lora, "lr": conf.lr_lora}, {"params": new, "lr": conf.lr_new}], weight_decay=conf.wd)

        self.steps_per_epoch = max(1, len(self.train_dl) // conf.accum)
        total = conf.epochs * self.steps_per_epoch
        warm = max(1, min(conf.warmup, total // 10))
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

        self.D, self.step = None, 0

        # for GAN
        if conf.gan:
            self.D = DinoDiscriminator(conf.dino_id).to(self.dev)
            self.d_params = list(self.D.heads.parameters())
            self.d_opt = torch.optim.AdamW(self.d_params, lr=conf.lr_d, betas=(0.5, 0.999), weight_decay=0.0)
            print(f"discriminator: frozen {conf.dino_id} + heads {sum(p.numel() for p in self.d_params) / 1e6:.2f}M | "
                  f"w_adv {conf.w_adv}, D-only warm-up {conf.d_warmup} steps")

        print(f"trainable {sum(p.numel() for p in self.params) / 1e6:.2f}M "
              f"(lora {sum(p.numel() for p in lora) / 1e6:.2f}M, new {sum(p.numel() for p in new) / 1e6:.2f}M) | "
              f"train {len(train_ds)} val {len(self.val_ds)} | {self.steps_per_epoch} optimiser steps/epoch, "
              f"effective batch {conf.batch_size * conf.accum}")

    def _forward(self, batch: dict) -> tuple[dict, torch.Tensor, dict[str, float]]:
        L, ab, rgb = (batch[k].to(self.dev, non_blocking=True) for k in ("L", "ab", "rgb"))
        out = self.model(L)
        loss, parts = self.crit(out, self.model.color_head(out["feat"]), ab, rgb)
        return out, loss, parts

    def _disc(self, rgb: torch.Tensor, L: torch.Tensor) -> list[torch.Tensor]:
        return self.D(*diff_augment(rgb, L)) if self.conf.diffaug else self.D(rgb, L)

    def _zero_grads(self) -> None:
        self.opt.zero_grad(set_to_none=True)
        if self.D is not None:
            self.d_opt.zero_grad(set_to_none=True)

    def train_epoch(self, ep: int) -> dict[str, float]:
        """One epoch. With the GAN on, every micro-batch does a generator pass, then a discriminator pass."""
        c = self.conf
        self.model.train()
        self.model.gray_clip.eval() # frozen
        if self.D is not None:
            self.D.train()
            
        sums, cnt, skipped = defaultdict(float), defaultdict(int), 0
        self._zero_grads()
        bar = tqdm(self.train_dl, desc=f"train {ep}", leave=False)
        for i, b in enumerate(bar):

            # colorize with SDUNet
            out, loss, parts = self._forward(b)

            if self.D is not None:
                L, rgb = b["L"].to(self.dev), b["rgb"].to(self.dev)
                fake = compose(L, out["ab"])                 # the input L + our colours: D can only judge color

                # D should have some steps to learn first before adjusting G
                if self.step >= c.d_warmup:
                    # G pass: gradient goes through D into the generator, but D's heads must not learn from it
                    self.D.heads.requires_grad_(False)
                    adv = g_hinge(self._disc(fake, L))

                    # back for D to learn (it will learn from real photo)
                    self.D.heads.requires_grad_(True)

                    # the effect of D on loss, ~0.2 vs 0.05
                    loss = loss + c.w_adv * adv
                    parts["adv"] = adv.item()

            # G learns
            if not torch.isfinite(loss):
                skipped += 1
                self._zero_grads()
                continue
            (loss / c.accum).backward()                      # frees the generator graph before the D pass; average over 4 batchs / 4

            # D learns
            if self.D is not None:
                # D pass: real photos up +1, our (detached) colourings down -1
                d_loss = d_hinge(self._disc(rgb, L), self._disc(fake.detach(), L))
                (d_loss / c.accum).backward()
                parts["d"] = d_loss.item()

            for k, v in parts.items():
                sums[k] += v * len(b["L"])
                cnt[k] += len(b["L"])

            # every 4 batchs make a change; opt & sched .step()
            if (i + 1) % c.accum == 0:
                torch.nn.utils.clip_grad_norm_(self.params, c.clip_grad)
                self.opt.step()
                self.sched.step()
                if self.D is not None:
                    torch.nn.utils.clip_grad_norm_(self.d_params, c.clip_grad)
                    self.d_opt.step()
                self._zero_grads()
                self.step += 1
                post = {"loss": f"{sums['total'] / cnt['total']:.3f}"}
                if self.D is not None:
                    post["d"] = f"{sums['d'] / cnt['d']:.3f}"
                    if cnt["adv"]:
                        post["adv"] = f"{sums['adv'] / cnt['adv']:.3f}"
                bar.set_postfix(post)
        if skipped:
            print(f"warning: {skipped} batches skipped (non-finite loss)")
        return {k: sums[k] / cnt[k] for k in sums}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Validation losses and metrics"""
        self.model.eval()
        sums, n = defaultdict(float), 0
        for b in tqdm(self.val_dl, desc="val", leave=False):
            out, _, parts = self._forward(b)
            L, rgb = b["L"].to(self.dev), b["rgb"].to(self.dev)
            pred = compose(L, out["ab"])
            k = len(L)
            for name, v in parts.items():
                sums[name] += v * k
            sums["psnr"] += psnr(pred, rgb).sum().item()
            sums["ssim"] += ssim_rgb(pred, rgb).sum().item()
            sums["lpips_vgg"] += lpips_vgg(self.lpips_eval, pred, rgb).sum().item()
            sums["colorfulness"] += colorfulness(pred).sum().item()
            sums["colorfulness_gt"] += colorfulness(rgb).sum().item()
            n += k
        return {k: v / n for k, v in sums.items()}

    @torch.no_grad()
    def save_samples(self, ep: int) -> None:
        self.model.eval()
        items = [self.val_ds[i] for i in range(min(self.conf.n_vis, len(self.val_ds)))]
        L = torch.stack([x["L"] for x in items]).to(self.dev)
        rgb = torch.stack([x["rgb"] for x in items])
        pred = compose(L, self.model(L)["ab"]).cpu()
        grid = make_grid(torch.cat([gray_rgb(L).cpu(), pred, rgb]), nrow=len(items), padding=2)
        (self.conf.out / "samples").mkdir(parents=True, exist_ok=True)
        save_image(grid, self.conf.out / "samples" / f"epoch_{ep:02d}.png")

    def _save_last(self, ep: int, best: float, bad: int) -> None:
        ck = {"model": self.model.trainable_state_dict(), "opt": self.opt.state_dict(),
              "sched": self.sched.state_dict(), "epoch": ep, "best": best, "bad": bad, "step": self.step}
        if self.D is not None:
            ck |= {"d_heads": self.D.heads.state_dict(), "d_opt": self.d_opt.state_dict()}
        torch.save(ck, self.conf.out / "last.pt")

    def fit(self) -> float:
        c = self.conf
        c.out.mkdir(parents=True, exist_ok=True)
        (c.out / "config.json").write_text(json.dumps(asdict(c), indent=1, default=str))

        key = "lpips_vgg" if self.D is not None else "total" 
        start, best, bad = 1, float("inf"), 0
        if c.resume:
            ck = torch.load(c.resume, map_location="cpu")
            self.model.load_state_dict(ck["model"], strict=False)
            self.opt.load_state_dict(ck["opt"])
            self.sched.load_state_dict(ck["sched"])
            start, best, bad, self.step = ck["epoch"] + 1, ck["best"], ck["bad"], ck.get("step", 0)
            if self.D is not None:
                self.D.heads.load_state_dict(ck["d_heads"])
                self.d_opt.load_state_dict(ck["d_opt"])
            print(f"resumed from {c.resume} at epoch {start} (best val {key} {best:.4f})")

        log, new_log = c.out / "log.csv", not (c.out / "log.csv").exists()
        best_ep = start - 1
        for ep in range(start, c.epochs + 1):
            t0 = time.time()
            tr = self.train_epoch(ep)
            va = self.evaluate()
            self.save_samples(ep)

            improved = va[key] < best
            bad = 0 if va[key] < best - c.min_delta else bad + 1
            if improved:
                best, best_ep = va[key], ep
                self.model.save(c.out / "best.pt")
            if self.D is not None:                           # keep every epoch, GAN is special!
                self.model.save(c.out / f"g_ep{ep:02d}.pt")
            self._save_last(ep, best, bad)

            row = {"epoch": ep, "lr_lora": self.opt.param_groups[0]["lr"], "seconds": round(time.time() - t0),
                   **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}

            fields = list(row) if new_log else next(csv.reader(open(log)))
            with open(log, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if new_log:
                    w.writeheader()
                    new_log = False
                w.writerow(row)

            gan = f" adv {tr.get('adv', float('nan')):.3f} d {tr['d']:.3f}" if self.D is not None else ""
            print(f"epoch {ep:3d} | train {tr['total']:.4f} (l1 {tr['l1']:.3f} lpips {tr['lpips']:.3f} ce {tr['ce']:.2f}{gan}) "
                  f"| val {va['total']:.4f} | psnr {va['psnr']:.2f} ssim {va['ssim']:.3f} lpips_vgg {va['lpips_vgg']:.3f} "
                  f"cf {va['colorfulness']:.1f}/{va['colorfulness_gt']:.1f} | {time.time() - t0:.0f}s"
                  f"{'  *best*' if improved else ''}")
            if c.patience and bad >= c.patience:
                print(f"early stop: no improvement for {c.patience} epochs (best epoch {best_ep})")
                break
        print(f"best val {key} {best:.4f} at epoch {best_ep} -> {c.out / 'best.pt'}")
        return best
