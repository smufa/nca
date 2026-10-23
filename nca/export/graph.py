"""The exported step: the ONNX graph, its nodes and the metadata strip.

`ExportStep` is the only thing torch.onnx ever traces; `export_step` wraps it in
the dynamic-shape logic and reports what the graph looks like. The graph takes the
perception kernel and the fire mask as *inputs*, which is what lets the frontend
rotate the kernel without re-exporting.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import NeuralCA, _circular_pad2d, get_living_mask
from ..spec import ModelSpec, load_model_state

DEFAULT_OPSET = 19


class ExportStep(nn.Module):
    """One CA step with explicit perception kernel and fire mask."""

    def __init__(self, ca: NeuralCA) -> None:
        super().__init__()
        self.ca = ca

    def forward(
        self,
        x: torch.Tensor,
        perception_weight: torch.Tensor,
        fire_mask: torch.Tensor,
        step_size: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        wrap = self.ca.wrap
        pre = get_living_mask(x, wrap)
        xp = _circular_pad2d(x, 1) if wrap else x
        y = F.conv2d(
            xp,
            perception_weight,
            stride=1,
            padding=0 if wrap else 1,
            groups=self.ca.channel_n,
        )
        if self.ca.cond_n:
            y = torch.cat(
                [y, cond.expand(-1, -1, y.shape[-2], y.shape[-1])], dim=1
            )
        dx = self.ca.update(y) * step_size
        x = x + dx * fire_mask
        post = get_living_mask(x, wrap)
        return x * (pre & post).to(x.dtype)


def infer_dims(state_dict: dict) -> tuple[int, int]:
    """Deprecated: return ``(channel_n, hidden_n)`` from a state dict.

    Kept only so external scripts keep working; use ``ModelSpec.from_state_dict``
    instead, which also knows about ``cond_n``, ``hidden_layers`` and
    ``learn_perception``.
    """
    spec = ModelSpec.from_state_dict(state_dict)
    return spec.channel_n, spec.hidden_n


def _load_ca(checkpoint: str | Path, wrap: bool = True) -> NeuralCA:
    """Build a ``NeuralCA`` matching the checkpoint and load its weights."""
    spec = ModelSpec.from_checkpoint(checkpoint, wrap=wrap)
    ca = spec.build()
    ca.load_state_dict(load_model_state(checkpoint))
    ca.eval()
    return ca


def export_step(
    ca: NeuralCA,
    path: str | Path,
    grid: int,
    dynamic: bool = True,
    opset: int = DEFAULT_OPSET,
    cond_n: int = 0,
) -> None:
    from torch.export import Dim

    step = ExportStep(ca).eval()
    c = ca.channel_n
    x = torch.zeros(1, c, grid, grid)
    x[0, 3:, grid // 2, grid // 2] = 1.0
    weight = ca.perception_weight()
    fire_mask = torch.ones(1, 1, grid, grid)
    step_size = torch.tensor(1.0)

    # dynamic_shapes is positional, matching forward()'s args:
    # (x, perception_weight, fire_mask, step_size[, cond]). None = leave static.
    dynamic_shapes = None
    if dynamic:
        batch, height, width = Dim("batch"), Dim("height"), Dim("width")
        dynamic_shapes = (
            {0: batch, 2: height, 3: width},
            None,
            {0: batch, 2: height, 3: width},
            None,
        )

    args = [x, weight, fire_mask, step_size]
    names = ["x", "perception_weight", "fire_mask", "step_size"]
    if cond_n:
        # A conditional model gets one extra input: the [B, cond_n, 1, 1] one-hot
        # selector, broadcast over the grid inside the step.
        args.append(torch.zeros(1, cond_n, 1, 1))
        names.append("cond")
        if dynamic_shapes is not None:
            dynamic_shapes = dynamic_shapes + ({0: batch},)

    torch.onnx.export(
        step,
        tuple(args),
        str(path),
        input_names=names,
        output_names=["x_next"],
        dynamic_shapes=dynamic_shapes,
        opset_version=opset,
        dynamo=True,
        external_data=False,  # keep the .onnx self-contained for browser loading
    )
    strip_export_metadata(path)


def strip_export_metadata(path: str | Path) -> int:
    """Drop ALL exporter metadata (node / graph / model) from a `.onnx` in place.

    None of it is needed to run the graph, and all of it is unstable:

    - `pkg.torch.onnx.stack_trace` embeds the absolute path of `nca/export/graph.py` and
      the *line numbers* of the traced code, so editing anything above those lines
      changes the committed bytes (measured: 41 bytes for a 3-line shift) and ships
      the author's home directory inside a public repo.
    - `pkg.torch.export.ExportedProgram.range_constraints` names the symbolic
      dimensions (`{s0, s10, ...}`), which differs between torch versions even when
      the graph is identical — another spurious `.onnx` diff on re-export.

    What survives is the actual contract: ops, attributes, initializers, input and
    output shapes. Stripping is idempotent, returns how many props it removed, and
    makes a re-export of an unchanged model byte-identical — which is what lets
    `export --all` + `git diff` serve as an exporter-drift guard.
    """
    import onnx

    model = onnx.load(str(path))
    removed = sum(len(node.metadata_props) for node in model.graph.node)
    removed += len(model.graph.metadata_props) + len(model.metadata_props)
    if not removed:
        return 0
    for holder in (model, model.graph, *model.graph.node):
        del holder.metadata_props[:]
    onnx.save(model, str(path))
    return removed


