"""`exports/`: targets, the manifest and `--reindex`.

A run directory is self-describing, so an export can recover what it needs from
`train.json`; `reindex` is the repair path that rebuilds `manifest.json` (and,
with `--refresh-targets`, re-renders `target.png`/`target.npy`) from the metas.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..targets import prepare_targets
from ..viz import render, save_image

def _git_rev() -> str | None:
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return None


def _run_train_json(checkpoint: Path) -> dict:
    """The run's ``train.json`` as a dict (empty if it is missing or corrupt)."""
    train_json = checkpoint.parent / "train.json"
    if not train_json.is_file():
        return {}
    try:
        return json.loads(train_json.read_text())
    except json.JSONDecodeError:
        return {}


def derive_targets(
    target: str, targets: list[str] | None, train_block: dict
) -> tuple[str, list[str] | None]:
    """Resolve the target label(s): explicit flags win, else the run's record.

    ``--target`` defaults to ``"unknown"``, and a run that trained on eight
    targets records all eight in ``train.json``. Exporting such a run with only
    ``--checkpoint``/``--id`` therefore used to label it ``unknown`` -- which
    ``scripts/check_meta.py`` rejects (``targets has 1 entries, cond_n is 8``) and
    which leaves the picker with no thumbnail and no selector labels.
    """
    if targets is None and train_block.get("targets"):
        targets = list(train_block["targets"])
    if target == "unknown" and targets:
        target = targets[0]
    return target, targets


def _train_block(run: dict, overrides: dict | None) -> dict:
    """What ``meta.json``'s ``train`` block records about the run.

    Only the keys that describe the *experiment* -- the regime and the numbers a
    model id encodes (``iterations``, ``mismatch_p``) -- so an exported model is
    reproducible from ``exports/`` alone. ``--train k=v`` overrides win.
    """
    cfg = run.get("experiment_config") or {}
    keys = (
        "regime", "iterations", "targets", "target_size", "padding", "use_pool",
        "damage_n", "noise_n", "noise_std", "mismatch_p", "fire_rate", "unroll",
        "lr", "seed", "sweep",
    )
    block = {k: cfg[k] for k in keys if k in cfg}
    block.update(overrides or {})
    return block


def _run_layout(checkpoint: Path) -> tuple[int | None, int | None]:
    """``(target_size, padding)`` the run was trained with, from its ``train.json``.

    The only trustworthy source once ``target.npy`` is gone: the two numbers
    together decide the canvas (``grid == target_size + 2*padding``), and neither
    can be recovered from the checkpoint or the grid alone. The old fallback here
    (40/16) disagreed with the training CLI's (32/8), which silently re-rendered
    ``target.png`` on a different canvas than the model was grown on.
    """
    train_json = checkpoint.parent / "train.json"
    if not train_json.is_file():
        return None, None
    try:
        data = json.loads(train_json.read_text())
    except json.JSONDecodeError:
        return None, None
    size, pad = data.get("target_size"), data.get("padding")
    return (int(size) if size is not None else None, int(pad) if pad is not None else None)


def _find_target(
    checkpoint: Path,
    target: str | None = None,
    max_size: int | None = None,
    padding: int | None = None,
) -> np.ndarray | None:
    """The ``[N, 4, H, W]`` target the run was trained on.

    Preferred source is the run's own ``target.npy`` (written by ``nca/cli.py``).
    Older runs predate it, so fall back to re-loading the ``--target`` spec --
    which is why ``target_size``/``padding`` are recorded in ``meta.json``: without
    them the same spec does not reproduce the same canvas.
    """
    candidate = checkpoint.parent / "target.npy"
    if candidate.exists():
        return np.load(candidate)
    if not target or target == "unknown":
        return None
    if max_size is None or padding is None:
        print(
            "  no target.npy and no target_size/padding on record; skipping target.png "
            "(pass --target-size/--padding, or export from the run directory)"
        )
        return None
    try:
        return prepare_targets([target], max_size=max_size, padding=padding).numpy()
    except (OSError, ValueError) as exc:  # offline cache miss, unknown spec, ...
        print(f"  no target.npy and {target!r} could not be reloaded ({exc}); skipping target.png")
        return None


def _update_manifest(out_dir: Path, entry: dict) -> None:
    manifest_path = out_dir / "manifest.json"
    manifest = {"version": 1, "models": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest["models"] = [m for m in manifest["models"] if m["id"] != entry["id"]]
    manifest["models"].append(entry)
    manifest["models"].sort(key=lambda m: m["id"])
    manifest["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _refresh_target_artefacts(out_dir: Path, meta: dict) -> str | None:
    """Re-render ``target.png`` (and ``target.npy``) from the meta alone.

    ``target_size`` + ``padding`` decide the canvas, so a meta that records both
    can rebuild the picture the run was grown on -- the only way to repair
    ``target.png`` for the runs whose ``target.npy`` is gone. Returns an error
    string, or ``None`` on success.
    """
    size, pad = meta.get("target_size"), meta.get("padding")
    specs = meta.get("targets") or [meta.get("target")]
    if size is None or pad is None or not specs or specs == ["unknown"]:
        return "no target_size/padding on record"
    try:
        target = prepare_targets(list(specs), max_size=int(size), padding=int(pad))
    except (OSError, ValueError) as exc:
        return f"could not reload target: {exc}"
    grid = int(meta.get("grid") or target.shape[-1])
    if target.shape[-1] != grid:
        return f"rebuilt canvas {target.shape[-1]} != meta grid {grid}"
    model_dir = out_dir / meta["id"]
    model_dir.mkdir(parents=True, exist_ok=True)
    # Same framing as `export_one` writes at export time (one frame per model),
    # so a refresh of an already-correct export is a byte-identical no-op.
    save_image(render(target)[0], model_dir / "target.png")
    np.save(model_dir / "target.npy", target.numpy())
    return None


def reindex(out_dir: Path, *, fix_targets: bool = False, fix_loss: bool = False) -> list[dict]:
    """Rebuild ``manifest.json`` from ``exports/*/meta.json``.

    ``meta.json`` is the source of truth and the manifest is a cache of it, so the
    manifest can always be regenerated: that is what makes the append-only
    ``_update_manifest`` crash-safe and what removes manifest/meta drift after a
    re-export or a manual edit.

    ``fix_loss`` backfills the ``loss`` block into metas exported before it was
    recorded, from the run dir beside ``source_checkpoint`` (silently skipped when
    that run is gone -- a curve cannot be invented).
    """
    out_dir = Path(out_dir)
    models: list[dict] = []
    for meta_path in sorted(out_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        model_id = meta.get("id") or meta_path.parent.name
        if fix_targets:
            error = _refresh_target_artefacts(out_dir, {**meta, "id": model_id})
            print(f"  {'ok  ' if error is None else 'skip'} {model_id}" + (f": {error}" if error else ""))
        if fix_loss and "loss" not in meta:
            curve = _loss_from_source(meta)
            if curve:
                meta["loss"] = curve
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        models.append({"id": model_id, "path": f"{model_id}/model.onnx", "meta": meta})
    manifest = {
        "version": 1,
        "models": sorted(models, key=lambda m: m["id"]),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest["models"]


PRESENTATION_KEYS = ("name", "sweep", "train", "picker", "loss")


def loss_curve(run_dir: str | Path, points: int = 64) -> dict | None:
    """A compact loss curve from the run's ``loss.json``.

    ``loss.json`` is one float per ``log_every`` iterations (tens to thousands of
    entries); the UI wants a shape, not a dataset, so bucket-mean it down to
    ``points`` samples. Bucket mean rather than subsample: the per-log loss is
    noisy, and a spike drawn from the noise looks like a training failure.

    Returns ``None`` when the run did not log anything (a legacy run, or a
    rebuild from an export that never had a curve).
    """
    path = Path(run_dir) / "loss.json"
    if not path.is_file():
        return None
    try:
        values = [float(v) for v in json.loads(path.read_text())]
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return None

    n = len(values)
    if n > points:
        edges = [round(i * n / points) for i in range(points + 1)]
        buckets = [values[a:b] for a, b in zip(edges, edges[1:]) if b > a]
        sampled = [sum(b) / len(b) for b in buckets]
    else:
        sampled = values

    lo = min(values)
    return {
        "points": [round(v, 6) for v in sampled],
        "logged": n,
        "min": round(lo, 6),
        "final": round(values[-1], 6),
        # Where the minimum was, as a fraction of the run: the UI marks it so a
        # model that bottomed out at 20% and then drifted is visible as such.
        "min_at": round(values.index(lo) / max(n - 1, 1), 4),
    }


def _carry_over(meta: dict) -> dict:
    """The presentation fields of an existing meta, for ``export_one(carry=...)``."""
    return {k: meta[k] for k in PRESENTATION_KEYS if k in meta}


def _loss_from_source(meta: dict) -> dict | None:
    """The recorded loss curve, from the run dir beside ``source_checkpoint``."""
    source = str(meta.get("source_checkpoint", ""))
    return loss_curve(Path(source).parent) if source else None


