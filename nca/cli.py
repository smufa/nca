"""Command line interface for training the Neural CA."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .model import NeuralCA
from .targets import load_target, make_seed, prepare_target
from .train import TrainConfig, train
from .viz import render, save_image

EXPERIMENTS = {
    "growing": dict(use_pool=False, damage_n=0),
    "persistent": dict(use_pool=True, damage_n=0),
    "regenerating": dict(use_pool=True, damage_n=3),
}


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a Neural Cellular Automaton to grow a target pattern."
    )
    p.add_argument(
        "--target",
        default="emoji:🦎",
        help="target spec: emoji:🦎 | emoji:1f98e | image:path.png | shape:heart",
    )
    p.add_argument("--target-size", type=int, default=32, help="target image size")
    p.add_argument("--padding", type=int, default=8, help="border padding around target")

    p.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default="regenerating",
        help="training regime (sets pool/damage defaults)",
    )
    p.add_argument("--no-pool", action="store_true", help="disable the sample pool")
    p.add_argument(
        "--damage-n", type=int, default=None, help="patterns to damage per batch"
    )
    p.add_argument("--pool-size", type=int, default=1024)
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="patterns per step; GPU memory scales with batch_size x --train-steps",
    )

    p.add_argument("--channels", type=int, default=16, help="number of state channels")
    p.add_argument("--hidden", type=int, default=128, help="update-rule hidden width")
    p.add_argument("--fire-rate", type=float, default=0.5)
    p.add_argument(
        "--learn-perception",
        action="store_true",
        help="learn the Sobel dx/dy perception kernels (identity stays fixed)",
    )
    p.add_argument(
        "--no-wrap",
        action="store_true",
        help="use zero (non-periodic) boundaries instead of circular wrapping",
    )

    p.add_argument("--iterations", type=int, default=1200)
    p.add_argument(
        "--train-steps",
        type=int,
        nargs=2,
        default=[48, 64],
        metavar=("MIN", "MAX"),
        help="random number of CA steps unrolled per training step",
    )
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument(
        "--lr-milestone",
        type=int,
        nargs="*",
        default=None,
        help="iteration(s) at which to decay the LR (default: 25%% of iterations)",
    )
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--no-grad-norm", action="store_true")
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the CA step (faster unroll on GPU)",
    )
    p.add_argument(
        "--amp",
        choices=["off", "bf16"],
        default="off",
        help="autocast dtype for the unroll (bf16 may be ~2x on compute-bound GPUs)",
    )

    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--run-name",
        default=None,
        help="deterministic run directory (runs/<name>)",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="resume training from a state.pt written by a previous run",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="render a batch montage every N iterations (0 = never)",
    )
    p.add_argument(
        "--save-pool",
        action="store_true",
        help="include the large sample pool in state.pt (exact resume; ~340 MB/ckpt)",
    )
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--gif-steps", type=int, default=120)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = _resolve_device(args.device)
    preset = EXPERIMENTS[args.experiment]
    use_pool = preset["use_pool"] and not args.no_pool
    damage_n = preset["damage_n"] if args.damage_n is None else args.damage_n
    if not args.lr_milestone:
        args.lr_milestone = [max(1, int(0.25 * args.iterations))]

    # --- target ---------------------------------------------------------
    target_rgba = load_target(args.target, max_size=args.target_size)
    target = prepare_target(target_rgba, args.padding)
    _, _, h, w = target.shape
    seed = make_seed(h, w, args.channels)

    # --- output directory ----------------------------------------------
    if args.run_name:
        out_dir = Path("runs") / args.run_name
    elif args.out_dir is None:
        safe = args.target.replace(":", "_").replace("/", "_")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = Path("runs") / f"{safe}_{args.experiment}_{stamp}"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save a preview of the padded target and the seed.
    target_preview = render(target)[0]
    seed_preview = render(seed.unsqueeze(0))[0]
    save_image(target_preview, out_dir / "target.png")
    save_image(seed_preview, out_dir / "seed.png")
    np.save(out_dir / "target.npy", target.numpy())

    cfg = TrainConfig(
        iterations=args.iterations,
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        use_pool=use_pool,
        damage_n=damage_n,
        train_steps_min=args.train_steps[0],
        train_steps_max=args.train_steps[1],
        lr=args.lr,
        lr_milestones=tuple(args.lr_milestone),
        lr_gamma=args.lr_gamma,
        normalize_grads=not args.no_grad_norm,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        progress_every=args.progress_every,
        save_pool=args.save_pool,
        make_gif=not args.no_gif,
        gif_steps=args.gif_steps,
        seed=args.seed,
        compile=args.compile,
        amp=args.amp,
    )

    ca = NeuralCA(
        channel_n=args.channels,
        fire_rate=args.fire_rate,
        hidden_n=args.hidden,
        wrap=not args.no_wrap,
        learn_perception=args.learn_perception,
    )

    print(f"Target: {args.target}  ({args.target_size}px, padded to {h}x{w})")
    train(ca, target, seed, cfg, device, out_dir, resume=args.resume)


if __name__ == "__main__":
    main()
