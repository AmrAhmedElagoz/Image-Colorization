import torch
import kornia.color as kc


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