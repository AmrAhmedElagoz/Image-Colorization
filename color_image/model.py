import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPVisionModel

from .common import gray_rgb

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class GrayCLIP(nn.Module):
    """CLIP ViT-B/32 vision to read both gray images and color images and generates tokens for both.
    """

    def __init__(self, clip_id: str, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.net = CLIPVisionModel.from_pretrained(clip_id, torch_dtype=dtype)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def tokens(self, rgb: torch.Tensor) -> torch.Tensor:
        """Any RGB image in [0, 1] -> [B, 50, 768] tokens (the teacher in distillation sees colour here)."""
        with torch.autocast("cuda", enabled=False):  # keep the resize / normalisation in fp32
            x = F.interpolate(rgb.float(), size=(224, 224), mode="bicubic", antialias=True).clamp(0, 1)
            x = ((x - self.mean) / self.std).to(self.net.dtype)
        return self.net(x).last_hidden_state

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):  # this disables mixed precision; Lab -> RGB gray needs full precision, bf has 3 significant decimal digits vs fp32 7
            rgb = gray_rgb(L.float())
        return self.tokens(rgb)


class GrayCLIPDistiller:
    """
    A teacher/student dist class for CLIP to read gray images.

    Teacher: frozen CLIP looking at the COLOR photo. 
    Student: GrayCLIP looking at the GRAY version (L only).    
    """

    def __init__(self, clip_id: str, device: str | None = None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # The student keeps fp32 weights (frozen too) and runs under bf16
        self.teacher = GrayCLIP(clip_id, torch.bfloat16).requires_grad_(False).eval().to(self.dev)
        self.student = GrayCLIP(clip_id, torch.float32).to(self.dev)

    def loss(self, rgb: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            target = self.teacher.tokens(rgb).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            pred = self.student(L).float()
        return F.mse_loss(pred, target)

    def _epoch(self, loader, opt=None, sched=None) -> float:
        train = opt is not None
        self.student.train(train)
        total, n = 0.0, 0
        for batch in tqdm(loader, desc="train" if train else "val", leave=False):
            rgb, L = batch["rgb"].to(self.dev), batch["L"].to(self.dev)
            with torch.set_grad_enabled(train):
                loss = self.loss(rgb, L)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                opt.step()
                sched.step()
            total += loss.item() * len(rgb)
            n += len(rgb)
        return total / n

    @torch.no_grad()
    def evaluate(self, loader) -> float:
        return self._epoch(loader)

    def fit(self, train_dl, val_dl, out: Path, epochs: int = 25, lr: float = 2e-5, wd: float = 0.01,
            warmup: int = 100, patience: int | None = 5, min_delta: float = 0.0) -> float:
        """AdamW + linear warm-up + cosine decay + grad-clip 1.0."""

        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        opt = torch.optim.AdamW(self.student.parameters(), lr=lr, weight_decay=wd)
        total = epochs * len(train_dl)

        # linear warm-up + cosine decay
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

        best = self.evaluate(val_dl)
        print(f"epoch 0 (OG CLIP on gray): val MSE {best:.5f}")
        best_ep, bad = 0, 0
        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr = self._epoch(train_dl, opt, sched)
            va = self.evaluate(val_dl)
            flag = ""
            bad = 0 if va < best - min_delta else bad + 1  
            if va < best:                                    
                best, best_ep, flag = va, ep, "  *saved*"
                self.save(out)
                
            print(f"epoch {ep:3d}  train {tr:.5f}  val {va:.5f}  gap {va - tr:+.5f}  ({time.time() - t0:.0f}s){flag}")
            if patience is not None and bad >= patience:
                print(f"early stop: no improvement for {patience} epochs (best epoch {best_ep})")
                break
        if not out.exists():
            print("warning: validation never improved on the OG CLIP, nothing was saved")
        print(f"best val MSE {best:.5f} at epoch {best_ep} -> {out}")
        return best

    def save(self, path: Path) -> None:
        torch.save(self.student.net.state_dict(), path)

