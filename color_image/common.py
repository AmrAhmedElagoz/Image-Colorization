import torch
import kornia.color as kc
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms as T
class EvalDataset(Dataset):
    """
    Eval dataset
    """
    def __init__(self, root: Path, split: str, size: int, limit: int | None = None):
        self.root = Path(root)
        self.names = (self.root / "splits" / f"{split}.txt").read_text().split()[:limit]
        self.tf = T.Compose([T.Resize(size, antialias=True), T.CenterCrop(size), T.ToTensor()])

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> dict:
        img = Image.open(self.root / "images" / self.names[i]).convert("RGB")
        return {"rgb": self.tf(img), "name": self.names[i]}


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = (x.float() - y.float()).pow(2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


def split_lab(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lab = kc.rgb_to_lab(rgb.clamp(0, 1)) # the lab shape (B, 3, H, W)
    return lab[:, :1], lab[:, 1:]   # channel 0 is L and 1,2 are ab

def compose(L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """Recombine the input luminance with predicted ab -> RGB in [0, 1]."""
    return kc.lab_to_rgb(torch.cat([L, ab], dim=1), clip=True)

def gray_rgb(L: torch.Tensor) -> torch.Tensor:
    return compose(L, torch.zeros_like(L).repeat(1, 2, 1, 1))

def colorfulness(rgb: torch.Tensor) -> torch.Tensor:
    """Hasler & Suesstrunk (2003), on [0, 255] scale."""
    r, g, b = (rgb * 255).unbind(1)
    rg, yb = r - g, 0.5 * (r + g) - b
    std = torch.sqrt(rg.flatten(1).var(1) + yb.flatten(1).var(1))
    mean = torch.sqrt(rg.flatten(1).mean(1) ** 2 + yb.flatten(1).mean(1) ** 2)
    return std + 0.3 * mean

def auc_ab(ab: torch.Tensor, ab_gt: torch.Tensor, max_t: int = 150) -> torch.Tensor:
    """
    base1 raw accuracy: the area under the curve of the fraction of pixels with ab error
    """
    d = (ab.float() - ab_gt.float()).norm(dim=1)
    passed = (max_t + 1 - torch.ceil(d)).clamp(0, max_t + 1)
    return (passed / (max_t + 1)).flatten(1).mean(1)


def norm_L(L: torch.Tensor) -> torch.Tensor:
    """L in [0, 100] -> [-1, 1], the range the SD VAE and CLIP expect."""
    return L / 50.0 - 1.0


def denorm_L(L_n: torch.Tensor) -> torch.Tensor:
    return (L_n + 1.0) * 50.0

AB_SCALE = 110.0

def norm_ab(ab: torch.Tensor) -> torch.Tensor:
    """ab (about +-110) -> about [-1, 1]."""
    return ab / AB_SCALE


def denorm_ab(ab_n: torch.Tensor) -> torch.Tensor:
    return ab_n * AB_SCALE
    