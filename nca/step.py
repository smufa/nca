"""The torch-free primitives of one CA step.

Everything here is a *constant* the model, the headless GPU bench, the exporter
and the frontend must agree on: the perception basis, the depthwise weight it
expands to, the seed state, and the channel->pixel mapping. Four languages
(Python/torch, Python/numpy, WGSL, TypeScript) execute the step today, and before
this module each of them carried its own copy of these numbers -- the same bug
class `nca/spec.py` fixed for shapes.

Rule: **nothing in this module may import torch**. `nca/model.py` imports it,
`scripts/shader_bench.py` imports it, `nca/viz.py` imports it, the tests import
it -- that is what makes it a shared definition rather than one implementation's
convenience. `frontend/src/nca.ts` is the JavaScript mirror; `tests/test_step.py`
fails if the two drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ALPHA_CHANNEL = 3
ALIVE_THRESHOLD = 0.1

#: The canonical perception basis, flattened 3x3 kernels: identity, Sobel-x/8,
#: Sobel-y/8. This is `frontend/src/nca.ts:CANONICAL_KERNELS`, number for number.
CANONICAL_KERNELS: tuple[tuple[float, ...], ...] = (
    (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    (-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0),
    (-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0),
)


def canonical_kernels(angle: float = 0.0) -> np.ndarray:
    """The three 3x3 kernels as ``[3, 3, 3]``, rotated by ``angle`` radians.

    ``angle`` rotates dx/dy (identity untouched), which is what the frontend's
    kernel lab exposes; it is the same rotation `nca.ts:rotateKernels` applies.
    Sobel is normalised by 8, as in the reference implementation.
    """
    ident = np.array(CANONICAL_KERNELS[0], dtype=np.float32).reshape(3, 3)
    dx = np.array(CANONICAL_KERNELS[1], dtype=np.float32).reshape(3, 3) / 8.0
    dy = np.array(CANONICAL_KERNELS[2], dtype=np.float32).reshape(3, 3) / 8.0
    c, s = math.cos(angle), math.sin(angle)
    return np.stack([ident, c * dx - s * dy, s * dx + c * dy]).astype(np.float32)


def perception_weight(channel_n: int, angle: float = 0.0) -> np.ndarray:
    """The depthwise ``[3C, 1, 3, 3]`` perception weight every channel shares.

    This is the ONNX graph's ``perception_weight`` input and the WGSL step's
    ``w_ident/w_dx/w_dy``: one shared ``(identity, dx, dy)`` triple repeated over
    channels, never a learned per-channel kernel.
    """
    kernels = canonical_kernels(angle)  # [3,3,3]
    weight = np.tile(kernels, (channel_n, 1, 1))  # [3C, 3, 3], channel-major
    return weight.reshape(3 * channel_n, 1, 3, 3)


def make_seed(channel_n: int, height: int, width: int) -> np.ndarray:
    """A single live cell in the middle: ``[C, H, W]``, RGB = 0, the rest = 1.

    Matches `nca.ts:seedState` (flat ``[C*H*W]``, channel-major, same cell).
    """
    seed = np.zeros((channel_n, height, width), dtype=np.float32)
    seed[ALPHA_CHANNEL:, height // 2, width // 2] = 1.0
    return seed


@dataclass(frozen=True)
class ChannelMapping:
    """Which state channels are shown as R/G/B and how they are composited.

    ``a >= 0`` means the channels are premultiplied and composited over ``bg``
    (0 = black, 1 = white): ``out = channel + (1 - alpha) * bg``. ``a = -1``
    shows the three channels directly (opaque), which is how the frontend's
    "hidden channel" views are drawn. Mirrors `nca.ts:ChannelMapping`.
    """

    r: int = 0
    g: int = 1
    b: int = 2
    a: int = ALPHA_CHANNEL
    bg: float = 1.0


DEFAULT_MAPPING = ChannelMapping()


def map_channels(
    state: np.ndarray,
    mapping: ChannelMapping = DEFAULT_MAPPING,
    *,
    h: int | None = None,
    w: int | None = None,
) -> np.ndarray:
    """``[C, H, W]`` (or flat ``[C*H*W]`` + ``h``/``w``) -> ``[H, W, 4]`` uint8 RGBA.

    The Python twin of `nca.ts:mapChannels`; ``draw.wgsl`` does the same per pixel.
    Rounds rather than truncates, because the TS side writes into a
    ``Uint8ClampedArray`` (nearest) -- a 1-LSB mismatch between the Python gif and
    the canvas is exactly the kind of drift a "mirror" claim invites.
    """
    x = np.asarray(state, dtype=np.float32)
    if x.ndim == 1:
        if h is None or w is None:
            raise ValueError("flat state needs h/w; the channel count is not inferable")
        x = x.reshape(-1, h, w)
    if x.ndim != 3:
        raise ValueError(f"expected [C, H, W], got shape {x.shape}")
    _, h, w = x.shape
    alpha = np.clip(x[mapping.a], 0.0, 1.0) if mapping.a >= 0 else np.ones((h, w), np.float32)
    inv = (1.0 - alpha) * mapping.bg
    out = np.empty((h, w, 4), dtype=np.uint8)
    for i, c in enumerate((mapping.r, mapping.g, mapping.b)):
        out[..., i] = np.rint(np.clip(x[c] + inv, 0.0, 1.0) * 255.0)
    out[..., 3] = 255
    return out

