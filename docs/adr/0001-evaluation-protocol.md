# ADR 0001: Evaluation protocol and how it is validated

- Status: Proposed
- Date: 2026-11-20
- Scope: evaluation and metrics (`color_image/metrics.py`, `scripts/evaluate.py`, `docs/evaluation.md`)

## Context
We compare colorization methods (Zhang 2016 baselines, the SD+LoRA model, and ours) on a 1,000-image MS-COCO test split.
No single metric measures colorization quality: closeness to the one true photo, presence of colour, and realism are
different questions. On our own run a plain gray image ties or beats Zhang's eccv16 on PSNR, SSIM, LPIPS and plain AuC, and a
literature check found no colorization metric that is both comparable across papers and validated against human judgement
(best standard metric: 0.567 rank correlation with people in one study). FID is biased at N = 1,000, and published FIDs for the
same model differ by more than 2x between papers.

## Decision
1. **Score saved images, not live models.** `evaluate.py` reads a folder of `<stem>.png` per method, so any method plugs in unchanged.
2. **Fixed data.** The `test` split, center crop 256x256, identical for every method. Colour-rarity weights come from the train split only.
3. **Report a table, not a score.** Metrics are grouped by the question they answer:
   - colour (primary): class-balanced AuC, colourfulness gap (signed and absolute), mean chroma, vivid-pixel share;
   - closeness (reference): PSNR, SSIM (RGB), LPIPS-VGG, plain AuC (secondary);
   - realism: FID and KID (Inception-v3 pool3 features), reported together.
4. **Always show two reference rows:** `gray` (floor) and `ground_truth` (ceiling).
5. **Statistics:** 95% percentile bootstrap CIs over images (2,000 resamples); paired bootstrap for method differences;
   KID reported as mean and std over subsets (a stability measure, not a CI).
6. **Human judgement is the primary evidence of perceived quality** (blind, shuffled, forced-choice user study; protocol to be written).
   Automatic metrics are supporting evidence.
7. **Claims are limited to our test split.** We do not compare our numbers with published ones (different data, sizes, extractors).

## Alternatives considered
- *One composite score:* rejected. Any weights would be our own invention, no colorization paper does it, and it would hide the
  closeness-versus-vividness trade-off (the two Zhang models disagree on exactly that).
- *FID alone:* rejected. Biased at N = 1,000 and blind to colour saturation per se; KID added as an unbiased check.
- *CLIP-IQA / QualiClip / DINOv2-based FD:* optional later; not validated on colorization outputs.
- *Larger test set:* possible (extra held-out COCO images) but changes the agreed 1,000-image split; deferred.

## Consequences
- (+) Results are comparable across methods within one protocol, reproducible (fixed seeds, saved per-image scores), and cheap to rerun.
- (+) The gray row makes metrics that cannot tell colourization from no colourization visible instead of hiding them.
- (-) Several numbers per method; no single winner when metrics disagree.
- (-) FID/KID at N = 1,000 are only comparable with each other.
- (-) Perceived quality still needs the user study, which is not yet run.

## Validation plan
The question is whether the metrics can be trusted. We check them against references whose answer is known in advance:
the ground-truth photos of the COCO test split, a gray image (floor), and Zhang's pretrained models; people are the final reference.

| # | Check | Reference / expected answer | Pass criterion |
|---|---|---|---|
| 1 | **Known answers:** degrade the ground-truth test images in controlled ways (saturation 0 to 2, hue rotation, noise, blur); also gray vs Zhang | a photo against itself is perfect; gray is worst on the colour metrics; a degraded copy scores worse than the original; Zhang beats gray on colour | every metric moves in the expected direction for the failure it is meant to catch; the sensitivity table is published |
| 2 | **Stability:** ranking of the methods on random halves of the test set; FID and KID on subsets of N = 100, 250, 500, 1,000 | the ranking should not depend on which images were drawn | same ranking in the large majority of splits; FID/KID bias and spread at N = 1,000 documented |
| 3 | **Human agreement:** user study (about 50 images), then rank correlation between each metric and human preference | people | reported as measured; metrics that disagree are demoted to "reference" |

Already done: `baseline.py` numbers reproduced by `evaluate.py`; KID matches a naive implementation; unit tests on synthetic data.
Checks 1 and 2 validate implementation, sensitivity and stability only; whether the metrics match perception is tested by check 3.

## Open questions
- Vivid-pixel threshold (30) is a convention; check 2 can be extended to test its influence.
- User-study size: SPEC says 20 to 30 images, the literature suggests about 50 to 100.
- Whether to enlarge the test set if check 2 shows intervals too wide.
- Comparability with the SD+LoRA paper (B2): its dataset and N are unknown to us.
