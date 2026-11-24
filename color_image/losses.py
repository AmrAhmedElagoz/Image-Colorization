import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

from .common import norm_ab
from .config import Config, LossConfig


def load_hull() -> torch.Tensor:
    return torch.from_numpy(np.load(Config().pts_in_hull)).float()


@torch.no_grad()
def compute_prior(loader, down: int = 4) -> np.ndarray:
    """
    how often does each of the 313 colours appear in our training photos?
    """
    pts = load_hull()
    counts = torch.zeros(len(pts))
    for b in loader:
        # for 4: 1/4 h and w; 1/16 pixels
        ab = F.avg_pool2d(b["ab"], down)
        flat = ab.permute(0, 2, 3, 1).reshape(-1, 2) # [B*h*w, 2] (used to compare distance between two tensors in torch.cdist)
        counts += torch.bincount(torch.cdist(flat, pts).argmin(1), minlength=len(pts)).float()  # minlength = 313 to include 0 shown colors too
    return (counts / counts.sum()).numpy()


class ColorCE(nn.Module):
    """
    Zhang rebalanced classification loss.
    baselines/zhang2016/resources/softmax_cross_entropy_loss_layer.cpp (we omit the KL divergence, plain CE)
    baselines/zhang2016/resources/caffe_traininglayers.py
    """

    def __init__(self, prior: np.ndarray, conf: LossConfig):
        super().__init__()
        pts = load_hull()
        self.knn, self.sigma = conf.knn, conf.sigma
        p = torch.as_tensor(prior, dtype=torch.float32)
        d2 = torch.cdist(pts, pts).pow(2)
        p_s = (p[None] * torch.exp(-d2 / (2 * conf.sigma ** 2))).sum(1)   # Gaussian-smoothed prior
        p_s = p_s / p_s.sum()   # %

        # real popularity + a flat amount, the same for every color 
        w = 1.0 / ((1 - conf.lam) * p_s + conf.lam / len(pts))  # lam: zhang hyperparam
        w = w / (p_s * w).sum() # normalize

        self.register_buffer("pts", pts, persistent=False)
        self.register_buffer("w", w, persistent=False)

    def soft_encode(self, ab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ab [B, 2, h, w] -> soft target [B, 313, h, w], nearest-bin index [B, h, w]."""
        B, _, h, w = ab.shape
        flat = ab.permute(0, 2, 3, 1).reshape(-1, 2)
        d, idx = torch.cdist(flat, self.pts).topk(self.knn, dim=1, largest=False)

        g = torch.exp(-d.pow(2) / (2 * self.sigma ** 2))    # score the distance, then normalize % :)
        g = g / g.sum(1, keepdim=True)

        z = torch.zeros(flat.shape[0], self.pts.shape[0], device=ab.device).scatter_(1, idx, g)
        return z.view(B, h, w, -1).permute(0, 3, 1, 2), idx[:, 0].view(B, h, w)

    def forward(self, logits: torch.Tensor, ab_gt: torch.Tensor) -> torch.Tensor:

        logits = logits.float()
        ab = F.adaptive_avg_pool2d(ab_gt.float(), logits.shape[-2:])
        z, nearest = self.soft_encode(ab)
        ce = -(z * F.log_softmax(logits, dim=1)).sum(1)       # [B, h, w]
        return (self.w[nearest] * ce).mean()


class ColorizationLoss(nn.Module):

    def __init__(self, conf: LossConfig | None = None):
        super().__init__()
        self.conf = conf or LossConfig()
        self.lpips = lpips.LPIPS(net=self.conf.lpips_net, verbose=False).eval().requires_grad_(False)
        self.ce = ColorCE(np.load(self.conf.prior), self.conf)

    def train(self, mode: bool = True):
        super().train(mode)
        self.lpips.eval()                  # the LPIPS net is a fixed measuring stick
        return self

    def forward(self, out: dict, logits: torch.Tensor, ab_gt: torch.Tensor,
                rgb_gt: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """out: model output ({"ab", "rgb"}); logits: color_head(out["feat"]); *_gt: raw ground truth."""

        l1 = F.l1_loss(norm_ab(out["ab"].float()), norm_ab(ab_gt.float()))
        lp = self.lpips(out["rgb"].float() * 2 - 1, rgb_gt.float() * 2 - 1).mean()
        ce = self.ce(logits, ab_gt)
        total = self.conf.w_l1 * l1 + self.conf.w_lpips * lp + self.conf.w_ce * ce
        return total, {"l1": l1.item(), "lpips": lp.item(), "ce": ce.item(), "total": total.item()}
