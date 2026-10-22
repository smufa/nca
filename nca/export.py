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
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import NeuralCA, _circular_pad2d, _perception_kernels, get_living_mask
from .spec import ModelSpec, find_checkpoint, load_model_state
from .targets import prepare_targets
from .viz import render, save_image

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


def _make_fire_masks(steps: int, grid: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [(rng.random((1, 1, grid, grid)) < 0.5).astype(np.float32) for _ in range(steps)]


@torch.no_grad()
def _torch_rollout(
    ca: NeuralCA,
    grid: int,
    masks: list[np.ndarray],
    channel_n: int,
    cond: torch.Tensor | None = None,
) -> np.ndarray:
    step = ExportStep(ca).eval()
    x = torch.zeros(1, channel_n, grid, grid)
    x[0, 3:, grid // 2, grid // 2] = 1.0
    weight = ca.perception_weight()
    s = torch.tensor(1.0)
    for m in masks:
        x = step(x, weight, torch.from_numpy(m), s, cond)
    return x.numpy()


def _onnx_rollout(
    onnx_path: str | Path,
    grid: int,
    masks: list[np.ndarray],
    channel_n: int,
    weight: np.ndarray,
    cond: np.ndarray | None = None,
) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.zeros((1, channel_n, grid, grid), np.float32)
    x[0, 3:, grid // 2, grid // 2] = 1.0
    for m in masks:
        feeds = {
            "x": x,
            "perception_weight": weight,
            "fire_mask": m,
            "step_size": np.array(1.0, dtype=np.float32),
        }
        if cond is not None:
            feeds["cond"] = cond
        x = sess.run(["x_next"], feeds)[0]
    return x


def verify(
    ca: NeuralCA,
    onnx_path: str | Path,
    grid: int,
    steps: int,
    dynamic: bool,
) -> dict:
    """Compare PyTorch and ONNX rollouts at one or two grid sizes."""
    channel_n = ca.channel_n
    rng = np.random.default_rng(0)
    weight = ca.perception_weight().detach().numpy()
    # Parity only needs *some* consistent selector; use the first target.
    cond = None
    if ca.cond_n:
        cond = torch.zeros(1, ca.cond_n, 1, 1)
        cond[0, 0] = 1.0
    grids = [grid, grid + 8] if dynamic else [grid]
    results = {}
    for g in grids:
        masks = _make_fire_masks(steps, g, rng)
        ref = _torch_rollout(ca, g, masks, channel_n, cond)
        out = _onnx_rollout(
            onnx_path, g, masks, channel_n, weight,
            cond.numpy() if cond is not None else None,
        )
        diff = float(np.max(np.abs(ref - out)))
        results[str(g)] = diff
        status = "ok" if diff < 1e-4 else "FAIL"
        print(f"  parity @ {g}x{g}: max|diff| = {diff:.2e}  [{status}]")
        if diff >= 1e-4:
            raise RuntimeError(f"ONNX parity check failed at {g}x{g}: {diff}")
    return results


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


def reindex(out_dir: Path) -> list[dict]:
    """Rebuild ``manifest.json`` from ``exports/*/meta.json``.

    ``meta.json`` is the source of truth and the manifest is a cache of it, so the
    manifest can always be regenerated: that is what makes the append-only
    ``_update_manifest`` crash-safe and what removes manifest/meta drift after a
    re-export or a manual edit.
    """
    out_dir = Path(out_dir)
    models: list[dict] = []
    for meta_path in sorted(out_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        model_id = meta.get("id") or meta_path.parent.name
        models.append({"id": model_id, "path": f"{model_id}/model.onnx", "meta": meta})
    manifest = {
        "version": 1,
        "models": sorted(models, key=lambda m: m["id"]),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest["models"]


def _save_weights(ca: NeuralCA, model_dir: Path, spec: ModelSpec) -> dict:
    """Dump the update-rule MLP as a flat float32 blob for the WGSL engine.

    The layout is owned by ``ModelSpec.weight_offsets()`` (see there for the block
    order). The only transformation applied here is padding each ``W1`` row from
    ``perception_n`` to ``perception_padded_n`` so the shader can read rows as
    ``vec4``. At ``C=16`` without conditioning that pad is a no-op, so existing
    blobs stay byte-identical.
    """
    h, p, p4 = spec.hidden_n, spec.perception_n, spec.perception_padded_n

    def f32(t: torch.Tensor) -> np.ndarray:
        return t.detach().reshape(-1).cpu().numpy().astype("<f4")

    w1 = ca.update[0].weight.detach().reshape(h, p).cpu().numpy().astype("<f4")
    if p4 != p:
        w1 = np.pad(w1, ((0, 0), (0, p4 - p)))
    parts: list[np.ndarray] = [w1.reshape(-1), f32(ca.update[0].bias)]
    # Intermediate hidden -> hidden blocks, in update order (conv 2..hidden_layers).
    for li in range(1, spec.hidden_layers):
        parts.append(f32(ca.update[2 * li].weight))
        parts.append(f32(ca.update[2 * li].bias))
    parts.append(f32(ca.update[-1].weight))
    parts.append(f32(ca.update[-1].bias))

    blob = np.concatenate(parts)
    if blob.size != spec.weights_count():
        raise RuntimeError(
            f"weights.bin layout mismatch: {blob.size} floats written, spec says "
            f"{spec.weights_count()} (offsets {spec.weight_offsets()})"
        )
    (model_dir / "weights.bin").write_bytes(blob.tobytes())
    return spec.weights_meta()


def export_one(
    checkpoint: str | Path,
    model_id: str,
    out_dir: Path,
    grid: int | None,
    dynamic: bool,
    opset: int,
    experiment: str,
    target: str,
    do_verify: bool,
    verify_steps: int,
    wrap: bool = True,
    targets: list[str] | None = None,
    max_size: int | None = None,
    padding: int | None = None,
) -> dict:
    import onnx
    import onnxruntime as ort

    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)
    model_dir = out_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Explicit flag > the run's own train.json > unknown. Deliberately no default:
    # inventing one is what put target.png on a different canvas than the run.
    run_size, run_padding = _run_layout(checkpoint)
    max_size = max_size if max_size is not None else run_size
    padding = padding if padding is not None else run_padding

    ca = _load_ca(checkpoint, wrap=wrap)
    spec = ca.spec
    if spec.num_parameters() != ca.num_parameters:
        raise RuntimeError(
            f"{model_id}: ModelSpec accounts for {spec.num_parameters()} parameters "
            f"but the module has {ca.num_parameters}. nca/spec.py is out of date with "
            f"nca/model.py (a new layer or parameter was added)."
        )
    target_arr = _find_target(checkpoint, target, max_size, padding)
    if grid is None:
        grid = int(target_arr.shape[-1]) if target_arr is not None else 64

    onnx_path = model_dir / "model.onnx"
    print(f"Exporting {model_id} (grid {grid}, dynamic={dynamic}, opset {opset})")
    export_step(
        ca,
        onnx_path,
        grid,
        dynamic=dynamic,
        opset=opset,
        cond_n=spec.cond_n,
    )

    parity = None
    if do_verify:
        parity = verify(ca, onnx_path, grid, verify_steps, dynamic)

    if target_arr is not None:
        save_image(render(torch.from_numpy(target_arr))[0], model_dir / "target.png")

    weight_layout = _save_weights(ca, model_dir, spec)

    model_proto = onnx.load(str(onnx_path))
    meta = {
        "id": model_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "channel_n": spec.channel_n,
        "cond_n": spec.cond_n,
        # 1 = the paper's single hidden block. The WGSL step implements 1..n.
        "hidden_layers": spec.hidden_layers,
        "hidden_n": spec.hidden_n,
        "fire_rate": spec.fire_rate,
        "learn_perception": spec.learn_perception,
        # The [3,3,3] (identity, d/dx, d/dy) kernels in use, flattened. For a
        # --learn-perception model these are the trained ones, so the frontend
        # must use them (and not the canonical Sobel).
        "perception_kernel": [round(float(v), 8) for v in ca.kernels().reshape(-1)],
        "num_parameters": spec.num_parameters(),
        "grid": grid,
        "dynamic_shape": dynamic,
        # How the target was placed on the grid: grid == target_size + 2*padding.
        # Recorded so target.png (and a future re-export) can be reproduced from
        # the meta alone -- 9 of the capacity-sweep runs predate target.npy.
        "target_size": max_size,
        "padding": padding,
        "wrap": spec.wrap,
        "opset": opset,
        "experiment": experiment,
        "target": target,
        # Full spec list for a conditional (--targets) run, so the playground can
        # label its target selector. Falls back to the single target.
        "targets": list(targets) if targets else [target],
        "source_checkpoint": str(checkpoint),
        "torch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
        "onnx_ir_version": model_proto.ir_version,
        "parity_max_abs_diff": parity,
        "weights": weight_layout,
        "size_bytes": onnx_path.stat().st_size,
        "git_rev": _git_rev(),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    entry = {
        "id": model_id,
        "path": f"{model_id}/model.onnx",
        "meta": meta,
    }
    _update_manifest(out_dir, entry)
    print(f"  wrote {onnx_path} ({meta['size_bytes']} bytes) and meta.json")
    return meta


def export_all(
    out_dir: Path,
    *,
    opset: int = DEFAULT_OPSET,
    do_verify: bool = True,
    verify_steps: int = 8,
    only: list[str] | None = None,
) -> list[dict]:
    """Re-export every model in ``out_dir`` that still has its source checkpoint.

    Grid / dynamic shape / experiment / target are taken from the existing
    ``meta.json`` so a re-export reproduces the same artefacts from the
    checkpoint. This is the command to run after any change to ``nca/model.py``
    or ``nca/export.py``; it then rewrites ``manifest.json`` via ``reindex`` so the
    manifest cannot drift.
    """
    out_dir = Path(out_dir)
    wanted = set(only) if only else None
    exported: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for meta_path in sorted(out_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        model_id = meta.get("id") or meta_path.parent.name
        if wanted is not None and model_id not in wanted:
            continue
        checkpoint = Path(meta.get("source_checkpoint") or "")
        if not checkpoint.is_file():
            skipped.append((model_id, str(checkpoint)))
            continue
        spec = ModelSpec.from_meta(meta)
        exported.append(
            export_one(
                checkpoint=checkpoint,
                model_id=model_id,
                out_dir=out_dir,
                grid=int(meta["grid"]) if meta.get("grid") else None,
                dynamic=bool(meta.get("dynamic_shape", True)),
                opset=opset,
                experiment=meta.get("experiment", "unknown"),
                target=meta.get("target", "unknown"),
                do_verify=do_verify,
                verify_steps=verify_steps,
                wrap=spec.wrap,
                targets=meta.get("targets"),
                # Preserve the *recorded* canvas (None stays None: a legacy meta
                # without target_size/padding must not silently gain a default).
                max_size=meta.get("target_size"),
                padding=meta.get("padding"),
            )
        )

    if skipped:
        print(f"\nSkipped {len(skipped)} model(s) whose source checkpoint is gone:")
        for model_id, path in skipped:
            print(f"  {model_id}: {path or '(not recorded)'}")

    reindex(out_dir)
    print(f"\nExported {len(exported)} model(s) into {out_dir}")
    return exported


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a trained NCA to ONNX.")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="path to a .pt checkpoint (required unless --all/--reindex)",
    )
    p.add_argument("--id", default=None, help="model id (default: run dir name)")
    p.add_argument("--out", default="exports", help="output directory")
    p.add_argument(
        "--all",
        action="store_true",
        help="re-export every model in --out whose source checkpoint still exists",
    )
    p.add_argument(
        "--only",
        nargs="+",
        metavar="ID",
        default=None,
        help="with --all: restrict to these model ids",
    )
    p.add_argument(
        "--reindex",
        action="store_true",
        help="rebuild manifest.json from */meta.json and exit",
    )
    p.add_argument("--grid-size", type=int, default=None, help="dummy grid size")
    p.add_argument(
        "--static",
        action="store_true",
        help="fix H/W instead of exporting dynamic spatial axes",
    )
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    p.add_argument("--experiment", default="unknown")
    p.add_argument("--target", default="unknown")
    p.add_argument(
        "--targets",
        nargs="+",
        default=None,
        metavar="SPEC",
        help="target spec list for a conditional model (metadata only)",
    )
    p.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="target canvas this run was trained with (default: the run's train.json; "
        "needed only when the run has no target.npy and --target must be reloaded)",
    )
    p.add_argument(
        "--padding",
        type=int,
        default=None,
        help="border this run was trained with; grid == target-size + 2*padding "
        "(default: the run's train.json)",
    )
    p.add_argument(
        "--no-wrap",
        action="store_true",
        help="use zero (non-periodic) boundaries instead of circular wrapping",
    )
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--verify-steps", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.reindex:
        models = reindex(Path(args.out))
        print(f"Reindexed {len(models)} model(s) into {Path(args.out) / 'manifest.json'}")
        return

    if args.all:
        export_all(
            Path(args.out),
            opset=args.opset,
            do_verify=not args.no_verify,
            verify_steps=args.verify_steps,
            only=args.only,
        )
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --all or --reindex is given")
    checkpoint = Path(args.checkpoint)
    model_id = args.id or checkpoint.parent.name or checkpoint.stem

    export_one(
        checkpoint=checkpoint,
        model_id=model_id,
        out_dir=Path(args.out),
        grid=args.grid_size,
        dynamic=not args.static,
        opset=args.opset,
        experiment=args.experiment,
        target=args.target,
        do_verify=not args.no_verify,
        verify_steps=args.verify_steps,
        wrap=not args.no_wrap,
        targets=args.targets,
        max_size=args.target_size,
        padding=args.padding,
    )


if __name__ == "__main__":
    main()
