"""Rendering helpers for NCA state grids."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import NeuralCA


def to_alpha(x: torch.Tensor) -> torch.Tensor:
    """Clamp the alpha channel to ``[0, 1]``. Shape ``[..., C, H, W]``."""
    return x[:, 3:4].clamp(0.0, 1.0)


def to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Composite premultiplied RGB over a white background."""
    return 1.0 - to_alpha(x) + x[:, :3]


def render(states: torch.Tensor) -> np.ndarray:
    """``[B, C, H, W]`` tensor -> ``[B, H, W, 3]`` uint8 array."""
    img = to_rgb(states).clamp(0.0, 1.0)
    arr = img.detach().permute(0, 2, 3, 1).cpu().numpy()
    return np.uint8(arr * 255)


def tile(images: np.ndarray, ncols: int | None = None, gap: int = 2) -> np.ndarray:
    """Tile a stack of ``[H, W, 3]`` images into a single grid image."""
    images = np.asarray(images)
    n = len(images)
    if n == 0:
        raise ValueError("nothing to tile")
    if ncols is None:
        ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    h, w = images.shape[1:3]
    canvas = np.full(
        (nrows * h + (nrows + 1) * gap, ncols * w + (ncols + 1) * gap, 3),
        255,
        dtype=np.uint8,
    )
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        y = gap + r * (h + gap)
        x = gap + c * (w + gap)
        canvas[y : y + h, x : x + w] = img
    return canvas


def save_image(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)


def _resize(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return arr
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


@torch.no_grad()
def growth_frames(
    ca: NeuralCA,
    seed: torch.Tensor,
    steps: int,
    device: torch.device,
    frame_every: int = 2,
    scale: int = 2,
    cond: torch.Tensor | None = None,
) -> list[np.ndarray]:
    """Run the CA from a seed and collect rendered frames."""
    ca.eval()
    x = seed.unsqueeze(0).to(device)
    frames = []
    for i in range(steps + 1):
        if i % frame_every == 0:
            frames.append(_resize(render(x)[0], scale))
        if i < steps:
            x = ca(x, cond=cond)
    return frames


def save_gif(frames: list[np.ndarray], path: str | Path, duration: int = 60) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )


def plot_loss(losses: list[float], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=110)
    ax.plot(losses, linewidth=0.8, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (MSE)")
    ax.set_title("NCA training loss")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
