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
        dx = self.ca.update(y) * step_size
        x = x + dx * fire_mask
        post = get_living_mask(x, wrap)
        return x * (pre & post).to(x.dtype)


def infer_dims(state_dict: dict) -> tuple[int, int]:
    weight = state_dict["update.0.weight"]
    return weight.shape[1] // 3, weight.shape[0]


def _load_ca(checkpoint: str | Path, wrap: bool = True) -> NeuralCA:
    state = torch.load(checkpoint, map_location="cpu")
    channel_n, hidden_n = infer_dims(state)
    ca = NeuralCA(channel_n=channel_n, hidden_n=hidden_n, wrap=wrap)
    ca.load_state_dict(state)
    ca.eval()
    return ca


def export_step(
    ca: NeuralCA,
    path: str | Path,
    grid: int,
    dynamic: bool = True,
    opset: int = DEFAULT_OPSET,
) -> None:
    from torch.export import Dim

    step = ExportStep(ca).eval()
    c = ca.channel_n
    x = torch.zeros(1, c, grid, grid)
    x[0, 3:, grid // 2, grid // 2] = 1.0
    weight = _perception_kernels(c)
    fire_mask = torch.ones(1, 1, grid, grid)
    step_size = torch.tensor(1.0)

    # dynamic_shapes is positional, matching forward()'s args:
    # (x, perception_weight, fire_mask, step_size). None = leave static.
    dynamic_shapes = None
    if dynamic:
        batch, height, width = Dim("batch"), Dim("height"), Dim("width")
        dynamic_shapes = (
            {0: batch, 2: height, 3: width},
            None,
            {0: batch, 2: height, 3: width},
            None,
        )

    torch.onnx.export(
        step,
        (x, weight, fire_mask, step_size),
        str(path),
        input_names=["x", "perception_weight", "fire_mask", "step_size"],
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
    ca: NeuralCA, grid: int, masks: list[np.ndarray], channel_n: int
) -> np.ndarray:
    step = ExportStep(ca).eval()
    x = torch.zeros(1, channel_n, grid, grid)
    x[0, 3:, grid // 2, grid // 2] = 1.0
    weight = _perception_kernels(channel_n)
    s = torch.tensor(1.0)
    for m in masks:
        x = step(x, weight, torch.from_numpy(m), s)
    return x.numpy()


def _onnx_rollout(
    onnx_path: str | Path, grid: int, masks: list[np.ndarray], channel_n: int
) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.zeros((1, channel_n, grid, grid), np.float32)
    x[0, 3:, grid // 2, grid // 2] = 1.0
    weight = _perception_kernels(channel_n).numpy()
    for m in masks:
        x = sess.run(
            ["x_next"],
            {
                "x": x,
                "perception_weight": weight,
                "fire_mask": m,
                "step_size": np.array(1.0, dtype=np.float32),
            },
        )[0]
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
    grids = [grid, grid + 8] if dynamic else [grid]
    results = {}
    for g in grids:
        masks = _make_fire_masks(steps, g, rng)
        ref = _torch_rollout(ca, g, masks, channel_n)
        out = _onnx_rollout(onnx_path, g, masks, channel_n)
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


def _find_target(checkpoint: Path) -> np.ndarray | None:
    for candidate in (checkpoint.parent / "target.npy",):
        if candidate.exists():
            return np.load(candidate)
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


def _save_weights(ca: NeuralCA, model_dir: Path) -> dict:
    """Dump the update-rule MLP as a flat float32 blob for the WGSL engine.

    Layout in floats: ``W1 [hidden, 3C]``, ``b1 [hidden]``, ``W2 [C, hidden]``,
    ``b2 [C]`` - row-major, matching ``ca.update``.
    """
    w1 = ca.update[0].weight.detach().reshape(-1).cpu().numpy().astype("<f4")
    b1 = ca.update[0].bias.detach().reshape(-1).cpu().numpy().astype("<f4")
    w2 = ca.update[2].weight.detach().reshape(-1).cpu().numpy().astype("<f4")
    b2 = ca.update[2].bias.detach().reshape(-1).cpu().numpy().astype("<f4")
    blob = np.concatenate([w1, b1, w2, b2])
    (model_dir / "weights.bin").write_bytes(blob.tobytes())
    return {
        "file": "weights.bin",
        "channel_n": int(ca.channel_n),
        "hidden_n": int(w1.size // (3 * ca.channel_n)),
        "offsets": {
            "w1": 0,
            "b1": int(w1.size),
            "w2": int(w1.size + b1.size),
            "b2": int(w1.size + b1.size + w2.size),
        },
        "count": int(blob.size),
    }


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
) -> dict:
    import onnx
    import onnxruntime as ort

    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)
    model_dir = out_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    ca = _load_ca(checkpoint, wrap=wrap)
    target_arr = _find_target(checkpoint)
    if grid is None:
        grid = int(target_arr.shape[-1]) if target_arr is not None else 64

    onnx_path = model_dir / "model.onnx"
    print(f"Exporting {model_id} (grid {grid}, dynamic={dynamic}, opset {opset})")
    export_step(ca, onnx_path, grid, dynamic=dynamic, opset=opset)

    parity = None
    if do_verify:
        parity = verify(ca, onnx_path, grid, verify_steps, dynamic)

    if target_arr is not None:
        save_image(render(torch.from_numpy(target_arr))[0], model_dir / "target.png")

    weight_layout = _save_weights(ca, model_dir)

    model_proto = onnx.load(str(onnx_path))
    meta = {
        "id": model_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "channel_n": ca.channel_n,
        "hidden_n": infer_dims(ca.state_dict())[1],
        "fire_rate": ca.fire_rate,
        "num_parameters": ca.num_parameters,
        "grid": grid,
        "dynamic_shape": dynamic,
        "wrap": wrap,
        "opset": opset,
        "experiment": experiment,
        "target": target,
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a trained NCA to ONNX.")
    p.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    p.add_argument("--id", default=None, help="model id (default: run dir name)")
    p.add_argument("--out", default="exports", help="output directory")
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
        "--no-wrap",
        action="store_true",
        help="use zero (non-periodic) boundaries instead of circular wrapping",
    )
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--verify-steps", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
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
    )


if __name__ == "__main__":
    main()
