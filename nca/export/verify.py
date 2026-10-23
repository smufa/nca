"""Rollout parity: torch vs ONNX, over random fire masks.

This is the gate AGENTS.md calls "training and inference must agree": the two
rollouts share the masks, so any difference is the export, not the sampling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..model import NeuralCA
from .graph import ExportStep

def _make_fire_masks(steps: int, grid: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [(rng.random((1, 1, grid, grid)) < 0.5).astype(np.float32) for _ in range(steps)]


@torch.no_grad()
def _torch_rollout(
    ca: NeuralCA,
    grid: int,
    masks: list[np.ndarray],
    channel_n: int,
    cond: torch.Tensor | None = None,
) -> np.ndarray:
    step = ExportStep(ca).eval()
    x = torch.zeros(1, channel_n, grid, grid)
    x[0, 3:, grid // 2, grid // 2] = 1.0
    weight = ca.perception_weight()
    s = torch.tensor(1.0)
    for m in masks:
        x = step(x, weight, torch.from_numpy(m), s, cond)
    return x.numpy()


def _onnx_rollout(
    onnx_path: str | Path,
    grid: int,
    masks: list[np.ndarray],
    channel_n: int,
    weight: np.ndarray,
    cond: np.ndarray | None = None,
) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.zeros((1, channel_n, grid, grid), np.float32)
    x[0, 3:, grid // 2, grid // 2] = 1.0
    for m in masks:
        feeds = {
            "x": x,
            "perception_weight": weight,
            "fire_mask": m,
            "step_size": np.array(1.0, dtype=np.float32),
        }
        if cond is not None:
            feeds["cond"] = cond
        x = sess.run(["x_next"], feeds)[0]
    return x


def verify(
    ca: NeuralCA,
    onnx_path: str | Path,
    grid: int,
    steps: int,
    dynamic: bool,
) -> dict:
    """Compare PyTorch and ONNX rollouts at one or two grid sizes."""
    channel_n = ca.channel_n
    rng = np.random.default_rng(0)
    weight = ca.perception_weight().detach().numpy()
    # Parity only needs *some* consistent selector; use the first target.
    cond = None
    if ca.cond_n:
        cond = torch.zeros(1, ca.cond_n, 1, 1)
        cond[0, 0] = 1.0
    grids = [grid, grid + 8] if dynamic else [grid]
    results = {}
    for g in grids:
        masks = _make_fire_masks(steps, g, rng)
        ref = _torch_rollout(ca, g, masks, channel_n, cond)
        out = _onnx_rollout(
            onnx_path, g, masks, channel_n, weight,
            cond.numpy() if cond is not None else None,
        )
        diff = float(np.max(np.abs(ref - out)))
        results[str(g)] = diff
        status = "ok" if diff < 1e-4 else "FAIL"
        print(f"  parity @ {g}x{g}: max|diff| = {diff:.2e}  [{status}]")
        if diff >= 1e-4:
            raise RuntimeError(f"ONNX parity check failed at {g}x{g}: {diff}")
    return results


