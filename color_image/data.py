from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .common import split_lab


def build_transform(size: int, train: bool) -> T.Compose:
    """Train: random crop (mild scale jitter) + flip. Eval: the same center crop for every method.
    No colour jitter: the target colours must stay true."""
    if train:
        return T.Compose([
            T.Resize(size, antialias=True),
            T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.9, 1.1), antialias=True),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    return T.Compose([
        T.Resize(size, antialias=True),
        T.CenterCrop(size),
        T.ToTensor(),
    ])


class ColorizationDataset(Dataset):
    """COCO colour photos -> RGB plus its Lab split.

    Returns raw (un-normalised) values: rgb in [0, 1], L in [0, 100], ab about +-110.
    Normalise on the GPU with common.norm_L / common.norm_ab.
    """

    def __init__(self, root: Path, split: str, size: int = 256, train: bool = False, limit: int | None = None):
        self.root = Path(root)
        self.names = (self.root / "splits" / f"{split}.txt").read_text().split()[:limit]
        self.tf = build_transform(size, train)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> dict:
        img = Image.open(self.root / "images" / self.names[i]).convert("RGB")
        rgb = self.tf(img)
        L, ab = split_lab(rgb[None])
        return {"rgb": rgb, "L": L[0], "ab": ab[0], "name": self.names[i]}

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