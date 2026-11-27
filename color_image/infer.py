import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .common import compose, gray_rgb, split_lab
from .config import ROOT, ModelConfig
from .model import SDUNet

DEFAULT_CKPT = ROOT / "runs" / "phase_c_w02" / "g_ep04.pt"       # SDUNet + GAN (w_adv 0.2), the final model
DEFAULT_CLIP = ROOT / "runs" / "clip" / "gray_clip.pt"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class Colorizer:

    def __init__(self, ckpt: Path = DEFAULT_CKPT, clip_ckpt: Path | None = DEFAULT_CLIP, size: int = 256,
                 device: str | None = None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.model = SDUNet(ModelConfig(clip_ckpt=clip_ckpt)).to(self.dev).eval()
        self.model.load(ckpt)

    @torch.inference_mode()
    def __call__(self, img: Image.Image) -> Image.Image:
        """Any PIL image (colour or grey, any size) -> colourised PIL image of the same size."""
        rgb = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1)[None].float() / 255
        L = split_lab(rgb)[0]                                           # full resolution, kept on the CPU
        small = F.interpolate(rgb, size=(self.size, self.size), mode="bilinear", antialias=True, align_corners=False)
        ab = self.model(split_lab(small.to(self.dev))[0])["ab"].float().cpu()
        ab = F.interpolate(ab, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return to_pil(compose(L, ab))

    @staticmethod
    def gray(img: Image.Image) -> Image.Image:
        rgb = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1)[None].float() / 255
        return to_pil(gray_rgb(split_lab(rgb)[0]))


def to_pil(rgb: torch.Tensor) -> Image.Image:
    return Image.fromarray((rgb[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description="Colourise images (files or folders) with SDUNet.")
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "infer")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--clip-ckpt", default=str(DEFAULT_CLIP), help="distilled GrayCLIP weights, or 'none'")
    ap.add_argument("--side-by-side", action="store_true", help="save [gray | colourised] instead of the result only")
    a = ap.parse_args()

    files = [f for p in a.inputs for f in (sorted(p.rglob("*")) if p.is_dir() else [p]) if f.suffix.lower() in EXTS]
    if not files:
        ap.error("no images found")
    clip = None if a.clip_ckpt.lower() == "none" else Path(a.clip_ckpt)
    col = Colorizer(a.ckpt, clip)
    a.out.mkdir(parents=True, exist_ok=True)
    for f in files:
        img = Image.open(f)
        res = col(img)
        if a.side_by_side:
            both = Image.new("RGB", (res.width * 2, res.height))
            both.paste(col.gray(img), (0, 0))
            both.paste(res, (res.width, 0))
            res = both
        res.save(a.out / f"{f.stem}.png")
        print(f"{f} -> {a.out / f'{f.stem}.png'} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
