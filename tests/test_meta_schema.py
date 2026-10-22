"""Every committed ``exports/*/meta.json`` must agree with ``ModelSpec``.

This is the repo-wide shape gate: it re-derives ``channel_n``/``hidden_n``/
``hidden_layers``/``cond_n``/``num_parameters``, the ``weights`` block (offsets,
count, padding) and the ONNX graph inputs from ``nca/spec.py``. The checker is
``scripts/check_meta.py`` -- the same code the CLI runs, so a failure here is
reproducible with ``python -m scripts.check_meta --only <model>``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.check_meta import EXPORTS, check_model  # noqa: E402

if not EXPORTS.is_dir():
    pytest.skip("no exports/ directory", allow_module_level=True)

MODEL_DIRS = sorted(p for p in EXPORTS.iterdir() if p.is_dir() and (p / "meta.json").is_file())
MODEL_IDS = [p.name for p in MODEL_DIRS]


@pytest.mark.parametrize("model_dir", MODEL_DIRS, ids=MODEL_IDS)
def test_meta_matches_spec(model_dir):
    meta, problems = check_model(model_dir)
    errors = [str(p) for p in problems if p.level == "error"]
    assert meta is not None, f"{model_dir.name}: meta.json unreadable"
    assert not errors, f"{model_dir.name}:\n  " + "\n  ".join(errors)


def test_manifest_lists_every_exported_model():
    import json

    manifest = json.loads((EXPORTS / "manifest.json").read_text())
    listed = [m["id"] for m in manifest["models"]]
    assert listed == sorted(MODEL_IDS)
    for entry in manifest["models"]:
        assert entry["path"] == f"{entry['id']}/model.onnx"
        assert entry["meta"]["id"] == entry["id"]
