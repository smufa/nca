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
    monkeypatch.setattr("nca.export.cli.export_one", lambda **kw: seen.update(kw) or {})
    monkeypatch.setattr("nca.export.cli.reindex", lambda out: [])
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


# --- export determinism ------------------------------------------------------


def test_committed_graphs_carry_no_exporter_metadata():
    """`export --all` + `git diff` is the exporter-drift guard, so the graphs must
    be free of the metadata that changes without the graph changing: the stack
    trace embeds `nca/export.py` line numbers (and the author's home path) and
    `range_constraints` names symbolic dims that differ between torch versions."""
    onnx = pytest.importorskip("onnx")
    noisy = []
    for model_id, _ in META:
        model = onnx.load(str(EXPORTS / model_id / "model.onnx"))
        if model.metadata_props or model.graph.metadata_props:
            noisy.append(model_id)
        if any(node.metadata_props for node in model.graph.node):
            noisy.append(model_id)
    assert not noisy, f"run `python -m nca.export --strip-metadata` for: {sorted(set(noisy))}"


def test_strip_export_metadata_is_idempotent(tmp_path):
    onnx = pytest.importorskip("onnx")
    src = EXPORTS / next(iter(META_IDS)) / "model.onnx"
    import shutil

    from nca.export import strip_export_metadata

    target = tmp_path / "model.onnx"
    shutil.copy(src, target)
    assert strip_export_metadata(target) == 0  # committed graphs are already clean
    model = onnx.load(str(target))
    model.graph.node[0].metadata_props.add(key="pkg.torch.onnx.stack_trace", value="x")
    onnx.save(model, str(target))
    assert strip_export_metadata(target) == 1
    assert strip_export_metadata(target) == 0


# --- meta/graph agreement ----------------------------------------------------


@pytest.mark.parametrize(("model_id", "meta"), META, ids=META_IDS)
def test_committed_graph_yields_a_valid_spec(model_id, meta):
    """`check_meta` derives the spec from the graph + blob; that derivation is what
    `--fix` trusts. It had two real bugs (`len(layers)` instead of `len(layers)-1`
    made every 1-layer model read as 2 layers, and the boundary inference ignored
    zero padding carried by a Conv), so it is pinned here for every export."""
    from scripts.check_meta import derive_spec

    spec, why = derive_spec(EXPORTS / model_id, meta)
    assert spec is not None, why
    assert spec.channel_n == meta["channel_n"]
    assert spec.hidden_n == meta["hidden_n"]
    assert spec.hidden_layers == meta.get("hidden_layers", 1)
    assert spec.cond_n == meta.get("cond_n", 0)
    assert spec.wrap == meta.get("wrap", True)


def test_derive_spec_sees_layers_and_zero_padding():
    from scripts.check_meta import derive_spec

    by_id = dict(META)
    deep, _ = derive_spec(EXPORTS / "cross8.mismatch.l=2", by_id["cross8.mismatch.l=2"])
    assert (deep.hidden_layers, deep.cond_n) == (2, 8)
    plain, _ = derive_spec(EXPORTS / "crab.growing", by_id["crab.growing"])
    assert (plain.hidden_layers, plain.cond_n) == (1, 0)
    # Zero padding with no `Pad` node at all: the perception Conv carries
    # `pads=[1,1,1,1]`, which is the only evidence that `wrap` is False.
    onnx = pytest.importorskip("onnx")
    demo, _ = derive_spec(EXPORTS / "smile.demo", by_id["smile.demo"])
    assert not any(node.op_type == "Pad" for node in onnx.load(str(EXPORTS / "smile.demo" / "model.onnx")).graph.node)
    assert demo.wrap is False
