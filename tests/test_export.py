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


def test_export_all_skips_models_without_a_checkpoint(tmp_path, capsys):
    exports = tmp_path / "exports"
    _fake_model(exports, "gone")
    assert export_all(exports, do_verify=False) == []
    out = capsys.readouterr().out
    assert "Skipped 1 model(s)" in out
    assert "gone" in out
    # The manifest is still rebuilt, so the UI keeps listing what exists on disk.
    assert (exports / "manifest.json").exists()
