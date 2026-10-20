"""Run a trained Neural CA: grow a pattern and optionally damage it.

Examples::

    # grow for 300 steps and save a gif
    uv run python -m nca.sample --checkpoint runs/.../ckpt_final.pt --steps 300

    # grow, then cut out the middle at step 150 and watch it regenerate
    uv run python -m nca.sample --checkpoint runs/.../ckpt_final.pt \
        --steps 400 --damage-at 150 --damage-frac 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import NeuralCA
from .targets import make_seed
from .viz import render, save_gif, save_image, tile


def _infer_dims(state_dict: dict) -> tuple[int, int]:
    weight = state_dict["update.0.weight"]
    hidden_n, in_channels = weight.shape[0], weight.shape[1]
    return in_channels // 3, hidden_n


def _damage(x: torch.Tensor, frac: float) -> torch.Tensor:
    _, _, h, w = x.shape
    yy = torch.arange(h, device=x.device, dtype=torch.float32)[:, None]
    xx = torch.arange(w, device=x.device, dtype=torch.float32)[None, :]
    cy, cx = h / 2, w / 2
    r = frac * min(h, w)
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) > r * r
    return x * mask


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample / animate a trained Neural CA.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="sample.gif")
    p.add_argument("--size", type=int, default=64, help="square grid size")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--fire-rate", type=float, default=0.5)
    p.add_argument("--frame-every", type=int, default=2)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--damage-at", type=int, default=None, help="step to apply damage")
    p.add_argument("--damage-frac", type=float, default=0.25)
    p.add_argument("--device", default="auto")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    state = torch.load(args.checkpoint, map_location="cpu")
    channel_n, hidden_n = _infer_dims(state)
    ca = NeuralCA(channel_n=channel_n, hidden_n=hidden_n, fire_rate=args.fire_rate)
    ca.load_state_dict(state)
    ca.to(device).eval()
    print(f"loaded {args.checkpoint} ({channel_n} channels, {ca.num_parameters} params)")

    seed = make_seed(args.size, args.size, channel_n).to(device)
    x = seed.unsqueeze(0)

    frames: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(args.steps + 1):
            if args.damage_at is not None and i == args.damage_at:
                x = _damage(x, args.damage_frac)
                print(f"applied damage at step {i}")
            if i % args.frame_every == 0:
                frames.append(render(x)[0])
            if i < args.steps:
                x = ca(x)

    out = Path(args.out)
    save_gif(frames, out)
    save_image(tile(np.stack([frames[0], frames[-1]]), ncols=2, gap=4), out.with_name(out.stem + "_final.png"))
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
