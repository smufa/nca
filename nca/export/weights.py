"""`weights.bin`: the MLP as flat float32 for the WGSL engine.

The offsets are `ModelSpec.weight_offsets()`, so this file only serialises; it must
never derive a layout of its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..model import NeuralCA
from ..spec import ModelSpec

def save_weights(ca: NeuralCA, model_dir: Path, spec: ModelSpec) -> dict:
    """Dump the update-rule MLP as a flat float32 blob for the WGSL engine.

    The layout is owned by ``ModelSpec.weight_offsets()`` (see there for the block
    order). The only transformation applied here is padding each ``W1`` row from
    ``perception_n`` to ``perception_padded_n`` so the shader can read rows as
    ``vec4``. At ``C=16`` without conditioning that pad is a no-op, so existing
    blobs stay byte-identical.
    """
    h, p, p4 = spec.hidden_n, spec.perception_n, spec.perception_padded_n

    def f32(t: torch.Tensor) -> np.ndarray:
        return t.detach().reshape(-1).cpu().numpy().astype("<f4")

    w1 = ca.update[0].weight.detach().reshape(h, p).cpu().numpy().astype("<f4")
    if p4 != p:
        w1 = np.pad(w1, ((0, 0), (0, p4 - p)))
    parts: list[np.ndarray] = [w1.reshape(-1), f32(ca.update[0].bias)]
    # Intermediate hidden -> hidden blocks, in update order (conv 2..hidden_layers).
    for li in range(1, spec.hidden_layers):
        parts.append(f32(ca.update[2 * li].weight))
        parts.append(f32(ca.update[2 * li].bias))
    parts.append(f32(ca.update[-1].weight))
    parts.append(f32(ca.update[-1].bias))

    blob = np.concatenate(parts)
    if blob.size != spec.weights_count():
        raise RuntimeError(
            f"weights.bin layout mismatch: {blob.size} floats written, spec says "
            f"{spec.weights_count()} (offsets {spec.weight_offsets()})"
        )
    (model_dir / "weights.bin").write_bytes(blob.tobytes())
    return spec.weights_meta()


# Meta fields that describe *which study the model belongs to* rather than the
# exported graph. A re-export regenerates everything else from the checkpoint, so
# these are carried over from the existing meta (the picker groups by them).
