"""Tests for the exporter: the ``weights.bin`` layout and the manifest.

``weights.bin`` is the contract between ``nca/export.py`` and the WGSL engine, so
the strongest cheap test is: rebuild the blob from the checkpoint for every
committed export and require it to be byte-identical to the shipped file. That
covers the plain, conditional (``cond_n``), deep (``hidden_layers``) and padded
perception (``C=6`` -> ``W1`` rows padded 18 -> 20) layouts at once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nca.export import _save_weights, export_all, reindex
from nca.spec import ModelSpec, load_model_state

EXPORTS = Path(__file__).resolve().parents[1] / "exports"


def _export_metas() -> list[tuple[str, dict]]:
    if not EXPORTS.is_dir():
        return []
    return [
        (p.parent.name, json.loads(p.read_text())) for p in sorted(EXPORTS.glob("*/meta.json"))
    ]


META = _export_metas()
META_IDS = [model_id for model_id, _ in META]


def _state_from_onnx(model_dir: Path) -> dict:
    """The committed graph's weights as a module state dict.

    ``model.onnx`` is committed for every export, so this makes the
    ``weights.bin`` layout checkable in a clean clone -- the checkpoint-based test
    below skips there (`runs/` is gitignored), which left CI green while nothing
    verified the blob against the model.
    """
    import onnx
    from onnx import numpy_helper

    torch = pytest.importorskip("torch")
    graph = onnx.load(str(model_dir / "model.onnx")).graph
    return {
        init.name[len("ca.") :]: torch.from_numpy(numpy_helper.to_array(init).copy())
        for init in graph.initializer
        if init.name.startswith("ca.")
    }


@pytest.mark.parametrize(("model_id", "meta"), META, ids=META_IDS)
def test_committed_blob_is_a_repack_of_the_committed_graph(model_id, meta, tmp_path):
    """``weights.bin`` == ``_save_weights(graph weights)``, without the run dir."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")

    spec = ModelSpec.from_meta(meta)
    state = _state_from_onnx(EXPORTS / model_id)
    ca = spec.build()
    missing, unexpected = ca.load_state_dict(state, strict=False)
    # Only the learned perception kernels are absent, and they are not in the blob.
    assert not unexpected, f"{model_id}: unexpected keys {unexpected}"
    assert set(missing) <= {"sobel"}, f"{model_id}: keys missing from the graph: {missing}"

    _save_weights(ca, tmp_path, spec)
    rebuilt = (tmp_path / "weights.bin").read_bytes()
    shipped = (EXPORTS / model_id / "weights.bin").read_bytes()
    assert rebuilt == shipped, f"{model_id}: weights.bin is not a repack of model.onnx"


@pytest.mark.parametrize(("model_id", "meta"), META, ids=META_IDS)
def test_save_weights_reproduces_committed_blob(model_id, meta, tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = Path(meta["source_checkpoint"])
    if not checkpoint.is_file():
        pytest.skip(f"{model_id}: checkpoint {checkpoint} is not available")

    spec = ModelSpec.from_meta(meta)
    ca = spec.build()
    ca.load_state_dict(load_model_state(checkpoint))
    _save_weights(ca, tmp_path, spec)

    rebuilt = (tmp_path / "weights.bin").read_bytes()
    shipped = (EXPORTS / model_id / "weights.bin").read_bytes()
    assert rebuilt == shipped, f"{model_id}: rebuilt weights.bin differs from the shipped one"


def _fake_model(exports: Path, model_id: str, checkpoint: str = "/does/not/exist.pt") -> None:
    model_dir = exports / model_id
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"")
    (model_dir / "meta.json").write_text(
        json.dumps(
            {
                "id": model_id,
                "channel_n": 16,
                "hidden_n": 32,
                "num_parameters": 2096,
                "grid": 64,
                "dynamic_shape": True,
                "source_checkpoint": checkpoint,
                "weights": ModelSpec(channel_n=16, hidden_n=32).weights_meta(),
            }
        )
    )


def test_reindex_rebuilds_the_manifest_from_metas(tmp_path):
    exports = tmp_path / "exports"
    _fake_model(exports, "b_model")
    _fake_model(exports, "a_model")

    models = reindex(exports)
    assert [m["id"] for m in models] == ["a_model", "b_model"]
    assert all(m["path"] == f"{m['id']}/model.onnx" for m in models)
    assert all("meta" in m for m in models)

    manifest = json.loads((exports / "manifest.json").read_text())
    assert manifest["version"] == 1
    assert [m["id"] for m in manifest["models"]] == ["a_model", "b_model"]
    assert "generated" in manifest


def test_export_all_carries_the_presentation_fields(tmp_path, monkeypatch):
    """A re-export must not drop `sweep`/`train` -- the picker groups by them."""
    exports = tmp_path / "exports"
    _fake_model(exports, "crab.growing")
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"not really a checkpoint")
    meta_path = exports / "crab.growing" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(
        sweep="emoji-sweep",
        name="crab.growing",
        train={"iteration": 8000},
        source_checkpoint=str(checkpoint),
    )
    meta_path.write_text(json.dumps(meta))

    seen: dict = {}
    monkeypatch.setattr("nca.export.export_one", lambda **kw: seen.update(kw) or {})
    monkeypatch.setattr("nca.export.reindex", lambda out: [])
    export_all(exports, do_verify=False)

    assert seen["carry"] == {
        "name": "crab.growing",
        "sweep": "emoji-sweep",
        "train": {"iteration": 8000},
    }


def test_committed_metas_carry_a_sweep():
    """Every shipped model must be groupable: the picker reads `meta.sweep`."""
    missing = [model_id for model_id, meta in META if not meta.get("sweep")]
    assert not missing, f"no sweep recorded: {missing}"


def test_export_all_skips_models_without_a_checkpoint(tmp_path, capsys):
    exports = tmp_path / "exports"
    _fake_model(exports, "gone")
    assert export_all(exports, do_verify=False) == []
    out = capsys.readouterr().out
    assert "Skipped 1 model(s)" in out
    assert "gone" in out
    # The manifest is still rebuilt, so the UI keeps listing what exists on disk.
    assert (exports / "manifest.json").exists()
