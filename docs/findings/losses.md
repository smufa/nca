# Perceptual loss terms for placed targets

Status: implemented (`nca/losses.py`), tested, scheduled for a cluster A/B
(`slurm/sweeps/perceptual.toml`). No model has been exported with it yet — the
verdict is still open, and this page says exactly which measurement would settle
it.

## The problem

Pixel MSE is happy with a blurred average. On a 40x40 emoji the CA can put the
right *amount* of ink in roughly the right *place* and pay almost nothing for
softness: the error is spread thinly over the whole silhouette instead of being
concentrated where the model is wrong. The visible symptom is a washed-out,
low-contrast pattern that reads as "almost right".

A Gram-matrix / style loss is the wrong tool here, and this repo has the numbers:
`docs/findings/conditioning.md` records cross-target mixing, where a style loss
would happily swap *which* emoji a region resembles. What is needed is a term that
is sensitive to *where* the structure is, on a small canvas, with no pretrained
network in the dependency graph.

## Three terms, all cheap and local

| term | what it measures | why it is here |
|---|---|---|
| `laplacian` | L1 between 3-level Laplacian pyramids of the composited RGB | penalises a blob with the right mean colour and no crisp edges — the actual failure mode |
| `ssim` | `1 - SSIM`, 11x11 Gaussian window, standard constants | structure + contrast + luminance in one number; well behaved at 40x40 where VGG has one feature |
| `edge` | L1 on Sobel `gx` and `gy` separately | sharper than the pyramid at the finest scale; keeps a hard alpha silhouette hard |

Two design decisions carry more weight than the formulas:

- **The terms see composited RGB, not premultiplied RGBA.** Alpha is not a colour;
  a perceptual measure of `rgb * 0` measures black. The pixel term keeps the alpha
  channel, and the perceptual terms are masked by the *union of the two
  silhouettes* so the vast empty background does not dilute a thin outline.
- **`laplacian` and `edge` are relative**: divided by the target's own mean
  magnitude. Without that, the same weight means a different thing for a bold
  emoji and a faint texture, and runs stop being comparable.

## The calibration finding (the interesting bit)

Raw magnitudes at the start of a run on a damaged 32x32 heart:

| term | raw magnitude | ratio to pixel (0.024) |
|---|---|---|
| pixel | 0.024 | 1x |
| laplacian | 1.00 | 42x |
| ssim | 0.088 | 3.7x |
| edge | 0.96 | 40x |

So `laplacian = 0.5` unnormalised is not "half a term", it is a term **21x heavier
than the objective it is meant to complement** — and it wins. Measured with
`scripts/loss_ab.py --mode repair` (200 CPU iterations, restore an erased disc),
before calibration:

| | PSNR | SSIM | edge err | alpha IoU |
|---|---|---|---|---|
| pixel only | **39.2 dB** | 0.9764 | **0.133** | 1.0000 |
| + raw terms | 22.0 dB | 0.9688 | 0.437 | 0.9748 |

The perceptual run got *worse at edges* while optimising the edge term, because the
pixel objective had been reduced to noise in the total.

`LossConfig.normalize` (default on) rescales every non-pixel term so its initial
magnitude equals the pixel loss's, measured on the run's own first batch
(`Loss.calibrate`). The scales are recorded per run in `loss_calibration.json`, so
a weight can be read back as "a fraction of the pixel loss" rather than a raw
number. After calibration, same setup:

| | PSNR | SSIM | edge err | alpha IoU |
|---|---|---|---|---|
| pixel only | 38.28 dB | 0.9683 | 0.1608 | 0.9983 |
| + calibrated terms | 36.44 dB | **0.9712** | **0.1506** | **1.0000** |

That is the expected trade and the one worth testing at scale: a little PSNR for
measurably better structure and edges. 200 CPU iterations on a 48x48 grid cannot
settle it — both runs are still improving, and the difference is small enough that
seed noise matters. **The verdict needs the cluster sweep.**

## Schedule

`start` (default 0) holds the extra terms off for a fraction of the run, `ramp`
brings them in linearly. Placement first: a CA that cannot yet put the shape in the
right place gains nothing from a crispness gradient, and early perceptual weights
are spent on edge statistics of a blob that is still the wrong shape.

## How to evaluate it

```bash
# cheap CPU sanity check of a weight vector (the mechanism, not the verdict)
python -m scripts.loss_ab --target shape:heart --compare pixel \
    --compare "laplacian=0.5,ssim=0.1,edge=0.05,start=0.2,ramp=0.2"

# the real A/B: same targets/regime/capacity as cross8.mismatch.h=256
python -m nca.sweep slurm/sweeps/perceptual.toml --dry-run
```

`scripts/loss_ab.py` scores on measures none of the configurations optimise
(composited PSNR/SSIM, relative edge error, alpha IoU) — the training curve cannot
answer "is this loss better", because the curve *is* the loss being changed.

Per-term histories land in `loss_components.json` next to `loss.json`, and the
training log prints them, so a run's total can always be decomposed.

## If this is not enough

VGG/LPIPS is the next step and stays behind the same interface (a term in
`nca/losses.py` with a weight), but it buys texture statistics, not placement,
which is why it is deferred: the failures above are placement and crispness at
40x40, and a 19-layer network on a 40x40 emoji sees almost nothing.
