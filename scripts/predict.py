import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from color_image.common import EvalDataset, compose, gray_rgb, split_lab
from color_image.config import ROOT, ZHANG_DIR, ModelConfig
from color_image.model import SDUNet

DDCOLOR_DIR = ROOT / "baselines" / "ddcolor"

sys.path.insert(0, str(DDCOLOR_DIR))
from ddcolor.model import DDColor # type: ignore
from huggingface_hub import PyTorchModelHubMixin

sys.path.insert(0, str(ZHANG_DIR))
import colorizers  # type: ignore

def ours(ckpt: Path, clip_ckpt: Path, device: str):

    model = SDUNet(ModelConfig(clip_ckpt=clip_ckpt)).to(device).eval()
    model.load(ckpt)
    return lambda rgb: model(split_lab(rgb)[0])["ab"]


def ddcolor(variant: str, device: str, size: int = 512):

    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)

    net = DDColorHF.from_pretrained(f"piddnad/{variant}").to(device).eval()

    def predict(rgb: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(rgb, size=(size, size), mode="bilinear", align_corners=False)
        ab = net(gray_rgb(split_lab(x)[0]))              # [B, 2, 512, 512] in Lab units
        return F.interpolate(ab, size=rgb.shape[-2:], mode="nearest")

    return predict


def zhang(variant: str, device: str):

    net = getattr(colorizers, variant)(pretrained=True).eval().to(device)
    return lambda rgb: net(split_lab(rgb)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=("ours", "ddcolor", "zhang"))
    ap.add_argument("--ckpt", type=Path, help="ours: generator weights (best.pt / g_epXX.pt)")
    ap.add_argument("--clip-ckpt", type=Path, default=ROOT / "runs" / "clip" / "gray_clip.pt")
    ap.add_argument("--variant", default=None, help="ddcolor: ddcolor_paper (default) | ddcolor_modelscope | ...; "
                                                    "zhang: eccv16 (default) | siggraph17")
    ap.add_argument("--name", default=None, help="output folder name (default: the variant)")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "coco")
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n", type=int, default=0, help="0 = whole split")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "preds")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.model == "ours":
        if a.ckpt is None:
            ap.error("ours needs --ckpt")
        predict, name = ours(a.ckpt, a.clip_ckpt, dev), a.name or a.ckpt.parent.name
    elif a.model == "ddcolor":
        v = a.variant or "ddcolor_paper"
        predict, name = ddcolor(v, dev), a.name or v
    else:
        v = a.variant or "eccv16"
        predict, name = zhang(v, dev), a.name or f"zhang_{v}"

    out = a.out / name
    out.mkdir(parents=True, exist_ok=True)
    ds = EvalDataset(a.data, a.split, a.size, limit=a.n or None)
    t0 = time.time()
    with torch.inference_mode():
        for b in tqdm(DataLoader(ds, batch_size=a.bs, num_workers=4), desc=name):
            rgb = b["rgb"].to(dev)
            pred = compose(split_lab(rgb)[0], predict(rgb).float())
            arr = (pred.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
            for x, n in zip(arr, b["name"]):
                Image.fromarray(x).save(out / f"{Path(n).stem}.png")
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"{len(ds)} images -> {out} ({(time.time() - t0) / len(ds):.3f} s/image)")


if __name__ == "__main__":
    main()
