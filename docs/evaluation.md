# Evaluation protocol

How every method (baselines and ours) is scored. Implemented in `color_image/metrics.py`,
run by `scripts/evaluate.py`.

## Setup
- Data: the 1,000-image `test` split of `data/coco` (center crop, 256x256), identical for every method.
- Input to the scorer: one folder of `<image-stem>.png` predictions per method. The scorer never runs a model,
  so any method can be plugged in.
- Reference rows, always reported next to the methods: `gray` (the input with no colour; the floor) and
  `ground_truth` (the ceiling).
- Randomness: bootstrap and KID subsets use `--seed` (default 0). The colour-rarity weights come from the
  **train split only** (cached in `data/coco/ab_prior_256.npy`).

## Metrics
Three questions, none answered by a single number, so results are reported as a table, not a score.

| Group | Metric | Better | Notes |
|---|---|---|---|
| Colour (primary) | class-balanced AuC | higher | Zhang et al. 2016 AuC with pixels re-weighted by colour rarity (lambda = 0) |
| | colourfulness gap `cf_gap` | closer to 0 | Hasler and Suesstrunk (2003) colourfulness, prediction minus ground truth; the dataset gap is the absolute mean |
| | mean chroma, vivid-pixel share | close to ground truth | Lab chroma; "vivid" = chroma above 30 (a convention, only comparable against the ground-truth value) |
| Closeness (reference) | PSNR, SSIM (RGB), LPIPS-VGG | PSNR, SSIM higher; LPIPS lower | each penalises any colour that differs from the one true photo; the gray row shows how little they separate colourization from no colourization |
| | plain AuC | higher | secondary: dominated by near-gray pixels (a gray prediction scores close to the models) |
| Realism | FID | lower | Inception-v3 pool3 features; biased at N = 1,000, compare only within one protocol |
| | KID (mean and std over subsets) | lower | unbiased squared MMD, cubic polynomial kernel, 100 random subsets of 500 |

Statistics: every per-image metric gets a 95% percentile bootstrap CI (2,000 resamples over images).
Differences between two methods use a paired bootstrap of the per-image difference; the gap is
supported for this test set only if its interval excludes 0. FID and KID are set-level and have no per-image CI
(KID reports the spread over subsets; the subsets overlap, so this is a stability measure, not a confidence interval).

## Running
```bash
python scripts/evaluate.py --pred eccv16=runs/baselines_test1000/eccv16 \
    siggraph17=runs/baselines_test1000/siggraph17 --realism
```
Outputs `runs/eval/summary.json` (means, CIs, paired differences, FID, KID) and `runs/eval/per_image.csv`.
`--realism` downloads the Inception weights on first use.

## Limits
- One test split of 1,000 images: intervals capture which images were drawn, not training-run variance.
- No metric here is validated against human judgement for colourization; the user study is the primary evidence
  of perceived quality.
- FID and KID values are not comparable with published numbers (different data, sizes and feature extractors).

## References
- Zhang et al., Colorful Image Colorization, ECCV 2016 (arXiv:1603.08511): AuC, class rebalancing.
- Hasler and Suesstrunk, Measuring colourfulness in natural images, 2003.
- Zhang et al., The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS), CVPR 2018 (arXiv:1801.03924).
- Heusel et al., GANs Trained by a Two Time-Scale Update Rule (FID), 2017 (arXiv:1706.08500).
- Binkowski et al., Demystifying MMD GANs (KID), ICLR 2018 (arXiv:1801.01401).
