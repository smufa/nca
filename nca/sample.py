"""Run a trained Neural CA: grow a pattern and optionally damage it.

Everything about the model's shape (channels, hidden width, depth, conditioning,
learned perception, boundary mode) is read from the run's ``spec.json`` or the
checkpoint itself, so this works for every model in ``exports/`` -- including the
conditional (``cross*`` / ``multi*``) and deeper (``cross8_deep``) ones.

Examples::

    # grow for 300 steps and save a gif
    uv run python -m nca.sample --checkpoint runs/.../ckpt_final.pt --steps 300

    # same, but let the run directory tell us the shape
    uv run python -m nca.sample --run runs/crab_regenerating --steps 300

    # a conditional model: grow target index 3
    uv run python -m nca.sample --run runs/cross8 --cond 3 --steps 300

    # grow, then cut out the middle at step 150 and watch it regenerate
    uv run python -m nca.sample --checkpoint runs/.../ckpt_final.pt \
        --steps 400 --damage-at 150 --damage-frac 0.25
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .spec import ModelSpec, find_checkpoint, load_model_state
from .targets import make_seed
from .viz import render, save_gif, save_image, tile


def _damage(x: torch.Tensor, frac: float) -> torch.Tensor:
    _, _, h, w = x.shape
    yy = torch.arange(h, device=x.device, dtype=torch.float32)[:, None]
    xx = torch.arange(w, device=x.device, dtype=torch.float32)[None, :]
    cy, cx = h / 2, w / 2
    r = frac * min(h, w)
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) > r * r
    return x * mask


def _select_cond(spec: ModelSpec, index: int, device: torch.device) -> torch.Tensor | None:
    """The ``[1, cond_n, 1, 1]`` one-hot selector for ``index`` (or ``None``)."""
    if not spec.cond_n:
        return None
    if not 0 <= index < spec.cond_n:
        raise SystemExit(f"--cond must be in [0, {spec.cond_n}) for this model, got {index}")
    cond = torch.zeros(1, spec.cond_n, 1, 1, device=device)
    cond[0, index, 0, 0] = 1.0
    return cond


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample / animate a trained Neural CA.")
    p.add_argument("--checkpoint", default=None, help="path to a .pt checkpoint")
    p.add_argument(
        "--run",
        default=None,
        help="run directory: reads spec.json and finds the checkpoint (preferred)",
    )
    p.add_argument("--out", default="sample.gif")
    p.add_argument("--size", type=int, default=64, help="square grid size")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument(
        "--fire-rate",
        type=float,
        default=None,
        help="override the fire rate (default: the model's own)",
    )
    p.add_argument(
        "--no-wrap",
        action="store_true",
        help="override the model's boundary mode to zero (non-periodic) padding",
    )
    p.add_argument("--cond", type=int, default=0, help="select target index (conditional models)")
    p.add_argument("--frame-every", type=int, default=2)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--damage-at", type=int, default=None, help="step to apply damage")
    p.add_argument("--damage-frac", type=float, default=0.25)
    p.add_argument("--device", default="auto")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.run:
        spec = ModelSpec.from_run_dir(args.run)
        checkpoint = Path(args.checkpoint) if args.checkpoint else find_checkpoint(Path(args.run))
    elif args.checkpoint:
        checkpoint = Path(args.checkpoint)
        spec = ModelSpec.from_checkpoint(checkpoint)
    else:
        parser.error("one of --checkpoint or --run is required")
        return  # unreachable, keeps type checkers happy
    if checkpoint is None:
        parser.error(f"no checkpoint found in {args.run}")

    if args.no_wrap:
        spec = replace(spec, wrap=False)
    if args.fire_rate is not None:
        spec = replace(spec, fire_rate=args.fire_rate)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ca = spec.build()
    ca.load_state_dict(load_model_state(checkpoint))
    ca.to(device).eval()
    cond = _select_cond(spec, args.cond, device)
    print(f"loaded {checkpoint} ({spec}, {ca.num_parameters} params)")

    x = make_seed(args.size, args.size, spec.channel_n).to(device).unsqueeze(0)

    frames: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(args.steps + 1):
            if args.damage_at is not None and i == args.damage_at:
                x = _damage(x, args.damage_frac)
                print(f"applied damage at step {i}")
            if i % args.frame_every == 0:
                frames.append(render(x)[0])
            if i < args.steps:
                x = ca(x, cond=cond)

    out = Path(args.out)
    save_gif(frames, out)
    save_image(
        tile(np.stack([frames[0], frames[-1]]), ncols=2, gap=4),
        out.with_name(out.stem + "_final.png"),
    )
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
