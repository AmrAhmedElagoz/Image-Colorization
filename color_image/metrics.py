from pathlib import Path

import numpy as np
import torch
import kornia.metrics as km
import lpips

from .common import EvalDataset, psnr, split_lab, auc_ab, colorfulness

PTS_IN_HULL = Path(__file__).parent / "assets" / "pts_in_hull.npy"
VIVID_CHROMA = 30.0  # a convention, not a standard: only useful against the ground-truth value


def ssim_rgb(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-image SSIM on RGB in [0, 1] (11x11 gaussian window). Not on L: L is copied from the input."""
    return km.ssim(x.float(), y.float(), window_size=11, max_val=1.0).flatten(1).mean(1)


def chroma(rgb: torch.Tensor) -> torch.Tensor:
    """Per-pixel Lab chroma sqrt(a^2 + b^2), shape (B, H, W)."""
    return split_lab(rgb)[1].norm(dim=1)


def mean_chroma(rgb: torch.Tensor) -> torch.Tensor:
    return chroma(rgb).flatten(1).mean(1)


def vivid_share(rgb: torch.Tensor, thr: float = VIVID_CHROMA) -> torch.Tensor:
    """Fraction of pixels with chroma above thr; a gray-brown fallback drives this towards 0."""
    return (chroma(rgb) > thr).float().flatten(1).mean(1)


def make_lpips(device: str) -> lpips.LPIPS:
    return lpips.LPIPS(net="vgg", verbose=False).eval().requires_grad_(False).to(device)


def lpips_vgg(net: lpips.LPIPS, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return net(x * 2 - 1, y * 2 - 1).flatten()


def bin_weights(prior: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Zhang eq. 4 with lambda=0: w ~ 1 / smoothed p, over the 313 ab bins, normalised to mean 1 under p."""
    pts = np.load(PTS_IN_HULL).astype(np.float64)
    d2 = ((pts[:, None] - pts[None]) ** 2).sum(-1)
    k = np.exp(-d2 / (2 * sigma**2))
    p = k @ prior
    p /= p.sum()
    w = 1.0 / p
    return w / (p * w).sum()


def ab_prior(data: Path, size: int, stride: int = 4, cache: Path | None = None, device: str = "cpu") -> np.ndarray:
    """Frequency of the 313 ab bins over the TRAIN split only (pixels subsampled by `stride`)."""
    if cache is not None and cache.exists():
        return np.load(cache)
    pts = torch.from_numpy(np.load(PTS_IN_HULL)).float().to(device)
    counts = torch.zeros(len(pts), dtype=torch.float64, device=device)
    ds = EvalDataset(data, "train", size)
    for i in range(len(ds)):
        ab = split_lab(ds[i]["rgb"][None].to(device))[1][..., ::stride, ::stride]
        idx = torch.cdist(ab.flatten(2)[0].T, pts).argmin(1)
        counts += torch.bincount(idx, minlength=len(pts)).double()
    prior = (counts / counts.sum()).cpu().numpy()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, prior)
    return prior


def auc_ab_balanced(ab: torch.Tensor, ab_gt: torch.Tensor, weights: np.ndarray, max_t: int = 150) -> torch.Tensor:
    """AuC with each pixel weighted by the rarity of its ground-truth colour (Zhang's class-balanced AuC)."""
    pts = torch.from_numpy(np.load(PTS_IN_HULL)).float().to(ab.device)
    w = torch.from_numpy(weights).float().to(ab.device)
    d = (ab.float() - ab_gt.float()).norm(dim=1)
    passed = ((max_t + 1 - torch.ceil(d)).clamp(0, max_t + 1) / (max_t + 1)).flatten(1)
    b, _, h, wd = ab_gt.shape
    pix = ab_gt.float().permute(0, 2, 3, 1).reshape(b, h * wd, 2)
    pw = w[torch.stack([torch.cdist(p, pts).argmin(1) for p in pix])]
    return (passed * pw).sum(1) / pw.sum(1)


def score(pred: torch.Tensor, gt: torch.Tensor, net: lpips.LPIPS, weights: np.ndarray) -> dict[str, torch.Tensor]:
    """
    Per-image scores of an RGB prediction against its ground truth (both (B, 3, H, W) in [0, 1]).
    cf_gap is signed (prediction - truth); the dataset-level colourfulness gap is the absolute value of its mean.
    """
    ab, ab_gt = split_lab(pred)[1], split_lab(gt)[1]
    return {
        "psnr": psnr(pred, gt),
        "ssim": ssim_rgb(pred, gt),
        "lpips_vgg": lpips_vgg(net, pred, gt),
        "auc_ab": auc_ab(ab, ab_gt),
        "auc_ab_balanced": auc_ab_balanced(ab, ab_gt, weights),
        "colorfulness": colorfulness(pred),
        "cf_gap": colorfulness(pred) - colorfulness(gt),
        "mean_chroma": mean_chroma(pred),
        "vivid_share": vivid_share(pred),
    }


def bootstrap_ci(x: np.ndarray, n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over images."""
    x = np.asarray(x, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_diff_ci(a: np.ndarray, b: np.ndarray, **kw) -> tuple[float, float, float]:
    """Mean of (a - b) with bootstrap CI; the gap is real for this test set if the CI excludes 0."""
    return bootstrap_ci(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), **kw)


def inception_features(folder: Path, device: str = "cpu", batch_size: int = 50) -> np.ndarray:
    """Inception-v3 pool3 features (N, 2048) of every image in a folder, as used by FID and KID."""
    from pytorch_fid.fid_score import get_activations
    from pytorch_fid.inception import InceptionV3
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    return get_activations([str(f) for f in files], model, batch_size, 2048, device)


def frechet_distance(mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray, eps: float = 1e-6) -> float:
    """
    Frechet distance between two Gaussians. Same formula as pytorch-fid, without its sqrtm(..., disp=False)
    call, which recent scipy versions reject.
    """
    from scipy import linalg
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"imaginary component {np.abs(covmean.imag).max()} in sqrtm")
        covmean = covmean.real
    return float((mu1 - mu2) @ (mu1 - mu2) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fid(real: np.ndarray, fake: np.ndarray) -> float:
    """FID between two feature sets; with ~1k images it is biased, compare like with like only."""
    return frechet_distance(real.mean(0), np.cov(real, rowvar=False), fake.mean(0), np.cov(fake, rowvar=False))


def kid(real: np.ndarray, fake: np.ndarray, subsets: int = 100, subset_size: int = 500, seed: int = 0) -> tuple[float, float]:
    """
    Kernel Inception Distance (Binkowski et al. 2018): unbiased squared MMD with the cubic polynomial
    kernel (x.y / d + 1)^3 on Inception features, averaged over random subsets. Returns (mean, std over subsets).
    """
    rng, d = np.random.default_rng(seed), real.shape[1]
    m = min(subset_size, len(real), len(fake))
    vals = []
    for _ in range(subsets):
        x = real[rng.choice(len(real), m, replace=False)].astype(np.float64)
        y = fake[rng.choice(len(fake), m, replace=False)].astype(np.float64)
        kxx, kyy, kxy = ((a @ b.T / d + 1) ** 3 for a, b in ((x, x), (y, y), (x, y)))
        vals.append((kxx.sum() - np.trace(kxx) + kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))
