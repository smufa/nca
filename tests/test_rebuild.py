"""Tests for rebuilding a model from its committed export.

The invariant: an export directory describes its model completely. That is what
lets ``nca.export --rebuild`` re-export a model whose training run is gone (24 of
the 45 committed exports had their checkpoint only in ``/tmp``).

The cheap, checkpoint-free way to prove it: ``state_from_export`` must reproduce
exactly the parameters in the committed ``model.onnx`` -- same layout, same values,
byte for byte. Anything that is in the graph but not in the export is a hole that
would make the model unreconstructable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nca.export.rebuild import model_from_export, state_from_export
from nca.export.weights import save_weights
from nca.spec import ModelSpec

EXPORTS = Path(__file__).resolve().parents[1] / "exports"

META = (
    [(p.parent.name, json.loads(p.read_text())) for p in sorted(EXPORTS.glob("*/meta.json"))]
    if EXPORTS.is_dir()
    else []
)
META_IDS = [model_id for model_id, _ in META]


def _graph_state(model_dir: Path) -> dict:
    """The committed graph's weights as numpy arrays."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(model_dir / "model.onnx")).graph
    return {
        init.name[len("ca.") :]: numpy_helper.to_array(init).copy()
        for init in graph.initializer
        if init.name.startswith("ca.")
    }


@pytest.mark.parametrize(("model_id", "meta"), META, ids=META_IDS)
def test_export_state_reproduces_the_committed_graph(model_id, meta):
    """Every parameter in ``model.onnx`` is recovered exactly, from the export alone."""
    pytest.importorskip("onnx")
    pytest.importorskip("torch")

    spec, state = state_from_export(EXPORTS / model_id)
    graph = _graph_state(EXPORTS / model_id)

    missing = sorted(set(graph) - set(state))
    assert not missing, f"{model_id}: not reconstructable from the export: {missing}"
    # A learned-perception model's Sobel is not a graph initialiser: the ONNX graph
    # takes `perception_weight` as an *input*, so the kernel lives in the meta only.
    allowed_extra = {"sobel"} if spec.learn_perception else set()
    extra = sorted(set(state) - set(graph) - allowed_extra)
    assert not extra, f"{model_id}: state_from_export invented {extra}"
    for key in sorted(set(graph)):
        want = graph[key]
        assert np.array_equal(np.asarray(state[key], dtype=np.float32), want.astype(np.float32)), (
            f"{model_id}: {key} differs from the graph"
        )
    assert spec.num_parameters() == int(meta["num_parameters"])


@pytest.mark.parametrize(("model_id", "meta"), META, ids=META_IDS)
def test_model_from_export_round_trips_the_blob(model_id, meta, tmp_path):
    """Rebuild the module, write it back: identical ``weights.bin``."""
    pytest.importorskip("torch")
    ca = model_from_export(EXPORTS / model_id)
    save_weights(ca, tmp_path, ca.spec)
    assert (tmp_path / "weights.bin").read_bytes() == (
        EXPORTS / model_id / "weights.bin"
    ).read_bytes()


def test_learn_perception_kernel_comes_from_the_meta(tmp_path):
    """A ``--learn-perception`` model needs its Sobel from the meta, not the blob."""
    pytest.importorskip("torch")
    pytest.importorskip("onnx")

    perc = [m for m in META_IDS if json.loads((EXPORTS / m / "meta.json").read_text())
            .get("learn_perception")]
    if not perc:
        pytest.skip("no --learn-perception export committed")
    model_id = perc[0]
    spec, state = state_from_export(EXPORTS / model_id)
    assert spec.learn_perception
    assert state["sobel"].shape == (2, 3, 3)
    kernels = np.asarray(
        json.loads((EXPORTS / model_id / "meta.json").read_text())["perception_kernel"]
    ).reshape(3, 3, 3)
    assert np.array_equal(state["sobel"], kernels[1:])

    ca = model_from_export(EXPORTS / model_id)
    assert np.array_equal(ca.kernels().detach().numpy(), kernels)


def test_state_from_weights_rejects_a_wrong_size(tmp_path):
    _, meta = META[0]
    spec = ModelSpec.from_meta(meta)
    with pytest.raises(ValueError, match="floats, spec says"):
        spec.state_from_weights(np.zeros(spec.weights_count() - 1, dtype="<f4"))


def test_state_from_weights_rejects_non_zero_padding():
    """A blob that is not from ``save_weights`` cannot be decoded (lossy padding)."""
    import torch

    from nca.model import NeuralCA

    spec = ModelSpec(channel_n=6, hidden_n=16, hidden_layers=1, learn_perception=True)
    assert spec.weights_padding > 0, "this test needs a padded perception"
    ca = spec.build()
    with torch.no_grad():
        for p in ca.parameters():
            p.zero_()
    blob = np.zeros(spec.weights_count(), dtype="<f4")
    # Corrupt one padding float of the first W1 row.
    blob[spec.weight_offsets()["w1"] + spec.perception_n] = 1.0
    with pytest.raises(ValueError, match="row padding is not zero"):
        spec.state_from_weights(blob)
