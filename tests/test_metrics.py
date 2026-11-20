import numpy as np
import torch

from color_image import metrics
from color_image.common import colorfulness, gray_rgb, split_lab

torch.manual_seed(0)
WEIGHTS = np.ones(313)


def rand_rgb(b: int = 2) -> torch.Tensor:
    return torch.rand(b, 3, 64, 64)


def test_ssim_identical_is_one():
    x = rand_rgb()
    assert torch.allclose(metrics.ssim_rgb(x, x), torch.ones(2), atol=1e-4)


def test_ssim_drops_with_noise():
    x = rand_rgb()
    assert (metrics.ssim_rgb(x, (x + 0.2 * torch.randn_like(x)).clamp(0, 1)) < 0.9).all()


def test_gray_has_no_colour():
    g = gray_rgb(split_lab(rand_rgb())[0])
    assert (metrics.mean_chroma(g) < 1).all()
    assert (metrics.vivid_share(g) == 0).all()
    assert (colorfulness(g) < 1).all()


def test_vivid_share_for_saturated_image():
    red = torch.zeros(1, 3, 32, 32)
    red[:, 0] = 1
    assert metrics.vivid_share(red).item() == 1.0


def test_balanced_auc_perfect_when_ab_equal():
    ab = split_lab(rand_rgb())[1]
    assert torch.allclose(metrics.auc_ab_balanced(ab, ab, WEIGHTS), torch.ones(2))


def test_bin_weights_rarer_is_heavier():
    prior = np.full(313, 1e-4)
    prior[0] = 1 - prior[1:].sum()
    w = metrics.bin_weights(prior)
    assert w[0] < w[1:].max()


def test_bootstrap_ci_brackets_mean_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small, big = rng.normal(5, 1, 50), rng.normal(5, 1, 5000)
    m, lo, hi = metrics.bootstrap_ci(small)
    assert lo < m < hi
    m2, lo2, hi2 = metrics.bootstrap_ci(big)
    assert (hi2 - lo2) < (hi - lo)


def test_paired_diff_excludes_zero_for_real_gap():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    _, lo, hi = metrics.paired_diff_ci(a + 0.5, a)
    assert lo > 0


def test_cf_gap_sign_and_zero():
    x = rand_rgb()
    g = gray_rgb(split_lab(x)[0])
    assert torch.allclose(colorfulness(x) - colorfulness(x), torch.zeros(2))
    assert ((colorfulness(g) - colorfulness(x)) < 0).all()


# KID / FID on synthetic feature clouds (no Inception needed)
def clouds(shift: float, n: int = 400, d: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, d)), rng.normal(shift, 1, (n, d))


def test_kid_same_distribution_is_near_zero():
    a, b = clouds(0.0)
    m, s = metrics.kid(a, b, subsets=50, subset_size=200)
    assert abs(m) < 3 * s + 1e-3


def test_kid_unbiased_not_systematically_positive():
    means = [metrics.kid(*clouds(0.0, seed=k), subsets=20, subset_size=50)[0] for k in range(30)]
    assert abs(np.mean(means)) < 3 * np.std(means) / np.sqrt(len(means))


def test_kid_detects_shift_and_grows_with_it():
    small = metrics.kid(*clouds(0.2), subsets=30, subset_size=200)[0]
    big = metrics.kid(*clouds(0.6), subsets=30, subset_size=200)[0]
    assert 0 < small < big


def test_fid_zero_for_identical_sets_and_grows_with_shift():
    a, _ = clouds(0.0)
    assert metrics.fid(a, a) < 1e-3
    assert metrics.fid(*clouds(0.3)) < metrics.fid(*clouds(0.8))


def test_fid_matches_pytorch_fid_formula():
    from pytorch_fid.fid_score import calculate_frechet_distance
    a, b = clouds(0.4)
    ref = calculate_frechet_distance(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))
    assert abs(metrics.fid(a, b) - ref) < 1e-6 * max(1.0, abs(ref))


def test_fid_works_when_scipy_sqrtm_has_no_disp():
    import scipy.linalg as sl
    orig = sl.sqrtm

    def no_disp(A, *args, **kw):
        if "disp" in kw:
            raise TypeError("sqrtm() got an unexpected keyword argument 'disp'")
        return orig(A, *args, **kw)

    sl.sqrtm = no_disp
    try:
        assert metrics.fid(*clouds(0.5)) > 0
    finally:
        sl.sqrtm = orig
