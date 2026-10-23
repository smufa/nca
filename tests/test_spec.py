"""Tests for ``nca.spec.ModelSpec`` -- the single source of truth for model shape.

The most valuable test here is ``test_weights_layout_matches_existing_exports``:
it re-derives the ``weights.bin`` layout and the ``meta.json`` ``weights`` block
from the spec for **every** committed export and asserts they are identical to
what is on disk. That pins the refactor: if ``ModelSpec`` ever disagrees with the
shipped artefacts (a new layer, a different offset, a changed pad), the export
tests fail instead of the browser silently rendering garbage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nca.spec import ModelSpec, find_checkpoint

EXPORTS = Path(__file__).resolve().parents[1] / "exports"


def _export_metas() -> list[tuple[str, dict]]:
    if not EXPORTS.is_dir():
        return []
    out = []
    for meta_path in sorted(EXPORTS.glob("*/meta.json")):
        out.append((meta_path.parent.name, json.loads(meta_path.read_text())))
    return out


META = _export_metas()
META_IDS = [model_id for model_id, _ in META]


def test_exports_found():
    """Guard against the parametrised tests silently collecting nothing."""
    assert META, f"no exports/*/meta.json found under {EXPORTS}"


@pytest.mark.parametrize(("model_id", "meta"), META, ids=META_IDS)
def test_weights_layout_matches_existing_exports(model_id, meta):
    """The spec must reproduce the shipped ``weights`` block exactly.

    Older exports predate ``hidden_layers`` / ``perception_n`` in the block, so we
    compare the keys that are present and separately require that the new keys
    agree with the module-derived values.
    """
    spec = ModelSpec.from_meta(meta)
    shipped = meta["weights"]
    derived = spec.weights_meta()
    for key, value in shipped.items():
        assert derived[key] == value, f"{model_id}: weights.{key} drifted"
    assert shipped["count"] == spec.weights_count()
    assert shipped["offsets"] == spec.weight_offsets()


@pytest.mark.parametrize(("model_id", "meta"), META, ids=META_IDS)
def test_num_parameters_matches_existing_exports(model_id, meta):
    spec = ModelSpec.from_meta(meta)
    assert spec.num_parameters() == meta["num_parameters"], f"{model_id}: param count drifted"
    # weights.bin holds the MLP minus the learned Sobel pair, plus vec4 padding.
    expected = meta["num_parameters"] - (18 if spec.learn_perception else 0)
    expected += spec.weights_padding
    assert spec.weights_count() == expected


@pytest.mark.parametrize(("model_id", "meta"), META, ids=META_IDS)
def test_spec_roundtrips_through_the_module(model_id, meta):
    """spec -> module -> spec must be the identity (shape, depth, cond, sobel)."""
    torch = pytest.importorskip("torch")
    spec = ModelSpec.from_meta(meta)
    ca = spec.build()
    assert ca.spec == spec
    assert ca.num_parameters == spec.num_parameters() == meta["num_parameters"]


@pytest.mark.parametrize(("model_id", "meta"), META, ids=META_IDS)
def test_committed_weight_blob_has_the_documented_size(model_id, meta):
    blob = EXPORTS / model_id / "weights.bin"
    if not blob.exists():
        pytest.skip(f"{model_id}: no weights.bin")
    spec = ModelSpec.from_meta(meta)
    assert blob.stat().st_size == 4 * spec.weights_count()


@pytest.mark.parametrize(("model_id", "meta"), META, ids=META_IDS)
def test_forward_pass_works_for_every_exported_model(model_id, meta):
    """One step with the right conditioning -- catches a spec the CLI cannot load."""
    torch = pytest.importorskip("torch")
    spec = ModelSpec.from_meta(meta)
    checkpoint = Path(meta["source_checkpoint"])
    if not checkpoint.is_file():
        pytest.skip(f"{model_id}: source checkpoint {checkpoint} is not available")

    ca = spec.build()
    ca.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    ca.eval()

    grid = 16
    x = torch.zeros(1, spec.channel_n, grid, grid)
    x[:, 3] = 1.0
    cond = None
    if spec.cond_n:
        cond = torch.zeros(1, spec.cond_n, 1, 1)
        cond[0, spec.cond_n - 1, 0, 0] = 1.0
    with torch.no_grad():
        out = ca(x, cond=cond)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_from_state_dict_reads_all_variants():
    pytest.importorskip("torch")
    from nca.model import NeuralCA

    cases = [
        dict(channel_n=16, hidden_n=128),
        dict(channel_n=16, hidden_n=128, hidden_layers=2),
        dict(channel_n=16, hidden_n=64, cond_n=4),
        dict(channel_n=12, hidden_n=32, learn_perception=True),
        dict(channel_n=6, hidden_n=8, wrap=False),
    ]
    for kwargs in cases:
        ca = NeuralCA(**kwargs)
        spec = ModelSpec.from_model(ca)
        assert spec.channel_n == kwargs["channel_n"]
        assert spec.hidden_n == kwargs["hidden_n"]
        assert spec.hidden_layers == kwargs.get("hidden_layers", 1)
        assert spec.cond_n == kwargs.get("cond_n", 0)
        assert spec.learn_perception == kwargs.get("learn_perception", False)
        assert spec.wrap == kwargs.get("wrap", True)
        assert spec.num_parameters() == ca.num_parameters


def test_perception_padding():
    assert ModelSpec(channel_n=16).perception_padded_n == 48
    assert ModelSpec(channel_n=6).perception_n == 18
    assert ModelSpec(channel_n=6).perception_padded_n == 20
    assert ModelSpec(channel_n=16, cond_n=8).perception_n == 56
    assert ModelSpec(channel_n=16, cond_n=8).perception_padded_n == 56
    assert ModelSpec(channel_n=16, hidden_n=128).hidden_groups == 32


def test_weight_offsets_are_contiguous():
    for spec in (
        ModelSpec(),
        ModelSpec(hidden_layers=3),
        ModelSpec(channel_n=6, cond_n=2, hidden_n=64),
    ):
        off = spec.weight_offsets()
        cursor = 0
        assert off["w1"] == cursor
        cursor += spec.hidden_n * spec.perception_padded_n
        assert off["b1"] == cursor
        cursor += spec.hidden_n
        if spec.hidden_layers > 1:
            assert off["mid_stride"] == spec.hidden_n**2 + spec.hidden_n
            # wmid/bmid point at the *first* intermediate block, the rest follow it.
            assert off["wmid"] == cursor
            assert off["bmid"] == cursor + spec.hidden_n**2
            cursor += off["mid_stride"] * (spec.hidden_layers - 1)
        assert off["w2"] == cursor
        cursor += spec.channel_n * spec.hidden_n
        assert off["b2"] == cursor
        cursor += spec.channel_n
        assert cursor == spec.weights_count()


def test_invalid_specs_are_rejected():
    with pytest.raises(ValueError):
        ModelSpec(channel_n=3)  # needs RGB + alpha
    with pytest.raises(ValueError):
        ModelSpec(hidden_n=0)
    with pytest.raises(ValueError):
        ModelSpec(hidden_layers=0)
    with pytest.raises(ValueError):
        ModelSpec(cond_n=-1)


def test_json_roundtrip(tmp_path):
    spec = ModelSpec(channel_n=12, hidden_n=64, hidden_layers=2, cond_n=6, learn_perception=True, wrap=False)
    path = tmp_path / "spec.json"
    spec.save(path)
    assert ModelSpec.from_json(json.loads(path.read_text())) == spec
    assert ModelSpec.from_run_dir(tmp_path) == spec


def test_from_meta_accepts_a_minimal_meta():
    spec = ModelSpec.from_meta({"channel_n": 16, "hidden_n": 128, "num_parameters": 8336})
    assert spec == ModelSpec()
    assert spec.wrap is True and spec.cond_n == 0 and spec.hidden_layers == 1


def test_find_checkpoint(tmp_path):
    assert find_checkpoint(tmp_path) is None
    (tmp_path / "ckpt_000100.pt").write_bytes(b"")
    (tmp_path / "ckpt_000200.pt").write_bytes(b"")
    assert find_checkpoint(tmp_path).name == "ckpt_000200.pt"
    (tmp_path / "ckpt_final.pt").write_bytes(b"")
    assert find_checkpoint(tmp_path).name == "ckpt_final.pt"
