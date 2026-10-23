"""Export a trained Neural CA to ONNX as a single, deterministic CA step.

The exported graph exposes the perception kernel and the stochastic update mask
as inputs, so the browser can drive the loop and the UI can manipulate the
kernel (rotation, custom Sobel filters) without re-exporting.

    uv run python -m nca.export --checkpoint runs/.../ckpt_final.pt \
        --id lizard --verify

Interface::

    x                 float32 [B, C, H, W]
    perception_weight float32 [3C, 1, 3, 3]
    fire_mask         float32 [B, 1, H, W]
    step_size         float32 []
    -> x_next         float32 [B, C, H, W]

Layout: `graph.py` (the traced step + metadata strip), `verify.py` (torch-vs-ONNX
rollout parity), `weights.py` (`weights.bin` for the WGSL engine), `manifest.py`
(`exports/`, targets, `--reindex`), `cli.py` (the argument surface).
"""

from .cli import (
    build_parser,
    export_all,
    export_one,
    main,
    strip_all_metadata,
)
from .graph import (
    DEFAULT_OPSET,
    ExportStep,
    export_step,
    infer_dims,
    strip_export_metadata,
)
from .manifest import PRESENTATION_KEYS, reindex
from .verify import verify
from .weights import _save_weights

__all__ = [
    "DEFAULT_OPSET",
    "ExportStep",
    "PRESENTATION_KEYS",
    "_save_weights",
    "build_parser",
    "export_all",
    "export_one",
    "export_step",
    "infer_dims",
    "main",
    "reindex",
    "strip_all_metadata",
    "strip_export_metadata",
    "verify",
]
