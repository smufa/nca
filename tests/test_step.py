"""`nca.step` is the shared definition; these tests keep the copies honest.

Four implementations execute one CA step: torch (`nca/model.py`), numpy
(`scripts/shader_bench.py`, now delegating here), WGSL (`frontend/src/wgsl/`) and
TypeScript (`frontend/src/nca.ts`). The Python ones must *call* this module, and
the TS mirror is checked textually -- a rotated kernel or a moved seed cell that
lands in only three of the four is exactly the kind of bug that shows up as a
mysterious 1e-2 parity failure.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

from nca import step
from nca import targets, viz
from nca.model import NeuralCA, _perception_kernels

ROOT = Path(__file__).resolve().parent.parent


def test_canonical_kernels_are_identity_and_sobel_over_eight() -> None:
    ident, dx, dy = step.canonical_kernels()
    assert np.allclose(ident, [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    # A Sobel-x sees a vertical edge; dy is its transpose, so dy == dx.T.
    assert np.allclose(dx, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8)
    assert np.allclose(dy, dx.T)


def test_perception_weight_is_the_depthwise_conv_the_module_uses() -> None:
    for channel_n in (1, 8, 16):
        numpy_weight = step.perception_weight(channel_n)
        torch_weight = _perception_kernels(channel_n).numpy()
        assert numpy_weight.shape == (3 * channel_n, 1, 3, 3)
        assert np.array_equal(numpy_weight, torch_weight)
    # The default angle is the one every export was trained with.
    assert np.array_equal(step.perception_weight(16), step.perception_weight(16, 0.0))


def test_perception_weight_rotation_matches_the_module() -> None:
    angle = 0.37
    ca = NeuralCA(channel_n=4, learn_perception=False)
    assert np.allclose(ca.perception_weight(angle).numpy(), step.perception_weight(4, angle))


def test_seed_matches_the_training_seed() -> None:
    for channel_n, size in ((16, 40), (8, 7), (16, 64)):
        assert np.array_equal(step.make_seed(channel_n, size, size), targets.make_seed(size, size, channel_n).numpy())


def test_ts_mirror_has_the_same_kernels() -> None:
    src = (ROOT / "frontend" / "src" / "nca.ts").read_text()
    block = src.split("CANONICAL_KERNELS", 1)[1].split("];", 1)[0]
    # The kernels are written unnormalised with a `.map((v) => v / 8)` on the
    # Sobel pair; strip it so only the 27 numbers remain.
    block = block.replace(".map((v) => v / 8)", "")
    numbers = [float(n) for n in re.findall(r"-?\d+", block)]
    assert len(numbers) == 27, f"expected 27 kernel numbers in nca.ts, got {numbers}"
    ts = np.array(numbers, dtype=np.float32).reshape(3, 3, 3)
    ts[1:] /= 8.0
    assert np.allclose(ts, step.canonical_kernels()), "nca.ts CANONICAL_KERNELS drifted from nca/step.py"


def test_step_module_never_imports_torch() -> None:
    """The invariant that makes it shared: if it needs torch it is an implementation."""
    import ast

    tree = ast.parse((ROOT / "nca" / "step.py").read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "torch" not in imported, f"nca/step.py must stay torch-free, imports {imported}"


def test_default_mapping_is_rgb_over_white() -> None:
    x = np.zeros((4, 2, 2), np.float32)
    x[0] = 0.5  # premultiplied red
    x[3] = 0.5  # half opaque
    out = step.map_channels(x)
    # out = rgb + (1 - a) * 1 = 0.5 + 0.5 = 1.0 on white.
    assert out.shape == (2, 2, 4)
    assert out[..., 0].tolist() == [[255, 255], [255, 255]]
    assert out[..., 3].tolist() == [[255, 255], [255, 255]]


def test_mapping_selects_arbitrary_channels_and_background() -> None:
    x = np.zeros((4, 1, 1), np.float32)
    x[2] = 0.25  # channel 2 -> R, opaque (a=-1), background ignored
    out = step.map_channels(x, step.ChannelMapping(r=2, g=2, b=2, a=-1, bg=0.0))
    assert out[0, 0].tolist() == [64, 64, 64, 255]

    x[3] = 0.0  # fully transparent over black: rgb + (1 - a) * 0 = rgb
    out = step.map_channels(x, step.ChannelMapping(r=2, g=2, b=2, bg=0.0))
    assert out[0, 0].tolist() == [64, 64, 64, 255]


def test_flat_input_needs_an_explicit_canvas() -> None:
    x = step.make_seed(16, 8, 8)
    flat = x.reshape(-1)
    assert np.array_equal(step.map_channels(flat, h=8, w=8), step.map_channels(x))
    # 16*64 is also 4*256 and 1*1024: guessing a canvas would guess wrong.
    with pytest.raises(ValueError):
        step.map_channels(flat)


def test_viz_render_agrees_with_the_mapping() -> None:
    rng = np.random.default_rng(0)
    states = torch.from_numpy(rng.random((2, 16, 5, 5), dtype=np.float32))
    default = viz.render(states)
    for i in range(2):
        assert np.array_equal(default[i], step.map_channels(states[i].numpy())[..., :3])

    # A hidden-channel view: RGB = channels 5/6/7, no alpha composite.
    hidden = step.ChannelMapping(r=5, g=6, b=7, a=-1, bg=0.0)
    img = viz.render(states, hidden)
    expected = np.rint(np.clip(states[0][[5, 6, 7]].numpy().transpose(1, 2, 0), 0, 1) * 255)
    assert np.array_equal(img[0], expected.astype(np.uint8))
