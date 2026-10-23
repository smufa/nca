"""Restore a model from its own committed export.

An export directory is meant to be self-describing, and this module is what makes
that true in practice: `meta.json` carries the spec, `weights.bin` carries every
update-rule parameter exactly as the WGSL engine reads them, and for a
``--learn-perception`` model ``meta.perception_kernel`` carries the trained Sobel.
So a model can be rebuilt with **no training run**: no checkpoint, no ``torch.save``
file, nothing left in ``/tmp``.

That matters because the run directories are the only thing an export does not
contain, and they are the thing that disappears (a cluster scratch dir, a reboot).
Without this, 24 of the 45 committed exports could never be re-exported again —
which would make an ONNX opset bump or an exporter fix impossible to apply.

The inverse of ``save_weights`` is ``ModelSpec.state_from_weights``; this module
adds the parts that live in the meta (the learned perception kernel) and builds
the module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..spec import ModelSpec


def state_from_export(model_dir: str | Path) -> tuple[ModelSpec, dict[str, np.ndarray]]:
    """``(spec, state_dict)`` for the model in ``model_dir``, as numpy arrays."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    spec = ModelSpec.from_meta(meta)
    blob = np.fromfile(model_dir / "weights.bin", dtype="<f4")
    state = spec.state_from_weights(blob)

    if spec.learn_perception:
        # meta.perception_kernel is the whole [3,3,3] (identity, dx, dy) triple;
        # only dx/dy are parameters (the identity row is a non-persistent buffer).
        kernels = np.asarray(meta["perception_kernel"], dtype=np.float32)
        state["sobel"] = kernels[9:].reshape(2, 3, 3).copy()
    return spec, state


def model_from_export(model_dir: str | Path):
    """Build a ``NeuralCA`` from ``model_dir`` (drop-in for a checkpoint load)."""
    import torch

    spec, state = state_from_export(model_dir)
    ca = spec.build()
    ca.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in state.items()}, strict=True)
    param_n, state_n = ca.num_parameters, spec.num_parameters()
    if param_n != state_n:
        raise RuntimeError(
            f"{Path(model_dir).name}: reconstructed {param_n} parameters, "
            f"ModelSpec accounts for {state_n} -- nca/spec.py is out of date"
        )
    return ca
