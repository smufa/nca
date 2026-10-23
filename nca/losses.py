"""Training losses for the CA, beyond plain pixel MSE.

The pixel term is the contract with every model already exported (``MSE`` over
channels ``0:4``), so it stays the default and the other terms are additive and
optional. They exist because MSE on a small target is happy with a blurry
average: the CA learns the right amount of ink in the wrong place, and the result
is soft. Three cheap, differentiable, torch-free-of-extra-deps terms fix most of
that:

``laplacian``
    L1 between the levels of the Laplacian pyramid of the *composited* RGB. Level
    ``k`` sees structures at scale ``2**k``, so the term penalises a blurred blob
    that has the right mean colour but no crisp edges -- the actual failure mode.

``ssim``
    ``1 - SSIM`` (11x11 Gaussian window, the standard constants). Structure,
    contrast and luminance in one number, well behaved on 56x56 emoji where a
    feature network (VGG/LPIPS) is out of its element.

``edge``
    L1 on the Sobel gradients (both directions, so a misplaced edge cannot cancel
    against a correct one). Sharper than the pyramid at the finest scale, and it
    is what makes a hard alpha silhouette stay hard.

Two decisions that matter more than the formulas:

* **The terms see composited RGB, not premultiplied RGBA.** Alpha is not a colour
  and a perceptual measure of "rgb * 0 alpha" is a measure of black; the pixel
  term keeps the alpha channel.
* **``laplacian`` and ``edge`` are relative**: divided by the target's own mean
  magnitude, so ``0.5`` means "half the edge error of the target" whether the
  target is a bold emoji or a faint texture. Without that the weights have to be
  retuned per target, and the losses stop being comparable between runs.

``start``/``ramp`` schedule the extra terms: placement first (the CA has to learn
*where* the shape goes before sharpening it means anything), then a linear ramp to
full weight. Applying them from step 0 makes the model spend capacity on edge
statistics while it cannot yet grow the shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F

TERMS = ("pixel", "laplacian", "ssim", "edge")


@dataclass(frozen=True)
class LossConfig:
    """Weights per term, plus when the non-pixel terms switch on.

    ``pixel`` keeps the historical meaning (MSE over RGBA), so a run with the
    defaults trains exactly what every earlier run trained.
    """

    pixel: float = 1.0
    laplacian: float = 0.0
    ssim: float = 0.0
    edge: float = 0.0
    start: float = 0.0  # fraction of the run before the extra terms appear
    ramp: float = 0.0  # fraction over which they reach full weight
    background: float = 1.0  # compositing colour: 1 = white, 0 = black
    # Scale each term so that its *initial* value equals the pixel term's, which
    # makes a weight "a fraction of the pixel loss" instead of a raw magnitude.
    # Measured on a damaged 32x32 heart: pixel 0.024, laplacian 1.00, ssim 0.088,
    # edge 0.96 -- so `laplacian = 0.5` unnormalised is a 20x heavier term than the
    # pixel error, and it wins, and the pattern dissolves (`scripts/loss_ab.py`).
    normalize: bool = True

    def __post_init__(self) -> None:
        for name in TERMS:
            weight = getattr(self, name)
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"loss.{name} must be a non-negative number, got {weight!r}")
        if not 0.0 <= self.start <= 1.0:
            raise ValueError(f"loss.start must be a fraction in [0, 1], got {self.start}")
        if self.ramp < 0.0:
            raise ValueError(f"loss.ramp must be >= 0, got {self.ramp}")
        if self.background not in (0.0, 1.0):
            raise ValueError(f"loss.background must be 0 or 1, got {self.background}")

    @property
    def uses_perception(self) -> bool:
        return any(getattr(self, t) for t in TERMS[1:])

    def extra_scale(self, progress: float) -> float:
        """How much of the non-pixel weight applies at ``progress`` (0..1 of the run)."""
        if not self.uses_perception:
            return 0.0
        if progress < self.start:
            return 0.0
        if self.ramp <= 0.0:
            return 1.0
        return min(1.0, (progress - self.start) / self.ramp)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LossConfig:
        if not data:
            return cls()
        if not isinstance(data, dict):
            raise ValueError(f"loss must be a table, got {type(data).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown loss key(s): {unknown}; known: {sorted(known)}")
        return cls(**data)


def _gaussian_window(size: int, sigma: float, dtype: torch.dtype, device: torch.device):
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)
    return g, (g.T @ g).unsqueeze(0).unsqueeze(0)  # [1, size], [1, 1, size, size]


class Loss:
    """The training objective: weighted pixel MSE plus optional perceptual terms.

    Call it with ``progress`` in ``[0, 1]`` so the schedule is a property of the
    loss, not of the loop. Returns ``(total, components)`` where ``components`` are
    detached scalars for logging -- the per-term numbers are the whole point of
    having terms, since the total alone cannot say *what* improved.
    """

    def __init__(self, cfg: LossConfig | dict[str, Any] | None = None) -> None:
        self.cfg = cfg if isinstance(cfg, LossConfig) else LossConfig.from_dict(cfg)
        self._windows: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}
        # Per-term scale, set once by `calibrate` (1.0 = unnormalised).
        self._norm: dict[str, float] = {}
        self.calibrated = False

    def calibrate(self, x: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Balance the terms against the pixel loss at the start of a run.

        Returns the scale per term (``1/``magnitude ratio) so the run can record
        what the weights actually meant -- without it, ``laplacian = 0.5`` says
        nothing about how hard that term pulls.
        """
        cfg = self.cfg
        if not (cfg.normalize and cfg.uses_perception):
            self.calibrated = True
            return {}
        _, components = self._evaluate(x.detach(), target)
        pixel = float(components.get("pixel", 0.0)) or 1e-8
        for term in TERMS[1:]:
            if term in components and components[term] > 0:
                self._norm[term] = pixel / float(components[term])
        self.calibrated = True
        return dict(self._norm)

    # -- terms -------------------------------------------------------------

    @staticmethod
    def composite(x: torch.Tensor, background: float = 1.0) -> torch.Tensor:
        """Premultiplied RGBA -> RGB over ``background`` (the renderer's rule)."""
        alpha = x[:, 3:4].clamp(0.0, 1.0)
        return x[:, :3] + (1.0 - alpha) * background

    @staticmethod
    def _alpha_mask(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Union of the two silhouettes: where either image has ink.

        Outside it both images are background, and averaging over that vast empty
        region is what dilutes a thin outline to nothing.
        """
        return torch.maximum(pred[:, 3:4].clamp(0, 1), target[:, 3:4].clamp(0, 1))

    def _ssim_window(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = (dtype, device)
        if key not in self._windows:
            _, window = _gaussian_window(11, 1.5, dtype, device)
            self._windows[key] = window
        return self._windows[key]

    def _laplacian(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        levels = 3
        err = pred.new_zeros(pred.shape[0])
        norm = target.new_zeros(())

        def down(t: torch.Tensor) -> torch.Tensor:
            return F.avg_pool2d(t, 2, 2)

        def up(t: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
            return F.interpolate(t, size=size, mode="bilinear", align_corners=False)

        a, b, m = pred, target, mask
        for _ in range(levels):
            size = a.shape[-2:]
            lap_a = a - up(down(a), size)
            lap_b = b - up(down(b), size)
            # Summed over levels, so each level contributes its own share of detail.
            err = err + (m * (lap_a - lap_b).abs()).flatten(1).sum(1)
            norm = norm + (m * lap_b.abs()).flatten(1).sum()
            a, b, m = down(a), down(b), down(m)
        # Relative: 1.0 means "as much wrong detail as the target has detail", so
        # the weight reads the same for a bold emoji and a faint texture.
        return err / norm.clamp_min(1e-6)

    def _edge(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        kx = pred.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = kx.transpose(-1, -2)

        def grad(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            gx = F.conv2d(t, kx.expand(t.shape[1], 1, 3, 3), padding=1, groups=t.shape[1])
            gy = F.conv2d(t, ky.expand(t.shape[1], 1, 3, 3), padding=1, groups=t.shape[1])
            return gx, gy

        px, py = grad(pred)
        tx, ty = grad(target)
        # L1 on each direction: matching magnitudes but flipped signs must not cancel.
        err = (mask * (px - tx).abs()).flatten(1).sum(1) + (mask * (py - ty).abs()).flatten(1).sum(1)
        norm = (mask * (tx.abs() + ty.abs())).flatten(1).sum(1)
        # Relative to the target's own edge energy: 1.0 = "the gradient is as wrong
        # as the target is edgy", i.e. a completely flat prediction.
        return err / norm.clamp_min(1e-6)

    def _ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        window = self._ssim_window(pred.dtype, pred.device)
        channels = pred.shape[1]
        w = window.expand(channels, 1, *window.shape[-2:])

        def moments(t: torch.Tensor):
            mu = F.conv2d(t, w, padding=5, groups=channels)
            var = F.conv2d(t * t, w, padding=5, groups=channels) - mu * mu
            return mu, var

        mu_x, var_x = moments(pred)
        mu_y, var_y = moments(target)
        cov = F.conv2d(pred * target, w, padding=5, groups=channels) - mu_x * mu_y
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
        )
        return 1.0 - ssim.mean(dim=(1, 2, 3))

    # -- the objective -----------------------------------------------------

    def _evaluate(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Unweighted per-sample terms: ``pixel`` and whichever extras are enabled."""
        cfg = self.cfg
        components: dict[str, torch.Tensor] = {
            "pixel": (x[:, :4] - target).pow(2).mean(dim=(1, 2, 3))
        }
        if not cfg.uses_perception:
            return components["pixel"], components

        mask = self._alpha_mask(x, target)
        # Masking the composite (not the diff) keeps the background out of the
        # statistics without inventing a silhouette for the edges.
        pred_rgb = self.composite(x[:, :4], cfg.background) * mask
        target_rgb = self.composite(target, cfg.background) * mask
        if cfg.laplacian:
            components["laplacian"] = self._laplacian(pred_rgb, target_rgb, mask)
        if cfg.ssim:
            components["ssim"] = self._ssim(pred_rgb, target_rgb)
        if cfg.edge:
            components["edge"] = self._edge(pred_rgb, target_rgb, mask)
        return components["pixel"], components

    def __call__(
        self, x: torch.Tensor, target: torch.Tensor, progress: float = 1.0
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        _, raw = self._evaluate(x, target)
        total = cfg.pixel * raw["pixel"]
        components = {k: v.detach().mean() for k, v in raw.items()}

        scale = cfg.extra_scale(progress)
        for term in TERMS[1:]:
            if term not in raw:
                continue
            weight = getattr(cfg, term) * (self._norm.get(term, 1.0) if cfg.normalize else 1.0)
            if weight:
                total = total + (scale * weight) * raw[term]
        return total, components
