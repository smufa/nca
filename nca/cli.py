"""Command line interface for training the Neural CA.

Two ways in:

* ad hoc -- every knob is a flag, defaults come from :class:`nca.experiments.Experiment`;
* from a sweep file -- ``--config slurm/sweeps/x.toml --run <name>``, with flags
  and ``--set k=v`` as overrides of that run.

The second form is what the cluster uses, and it is why the defaults live in one
place: a flag whose default differs from the experiment's default *is* the drift
this CLI used to have (``--target-size 32`` here, ``40`` in the exporter).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from .experiments import REGIMES, Experiment, experiment_from_dict, load_sweep
from .model import NeuralCA
from .targets import make_seed, prepare_targets
from .train import train
from .viz import render, save_image, tile

# Backwards-compatible name: the regime table used to live here.
EXPERIMENTS = REGIMES

_TUPLE_FIELDS = {"targets", "unroll", "lr_milestones"}


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _coerce(key: str, raw: str):
    """Best-effort typing for ``--set key=value`` (numbers, bools, tuples)."""
    if key in _TUPLE_FIELDS:
        parts = [p for p in raw.replace(" ", "").split(",") if p]
        return tuple(int(p) if p.lstrip("-").isdigit() else p for p in parts)
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return {"true": True, "false": False, "none": None}.get(raw.lower(), raw)


def build_parser() -> argparse.ArgumentParser:
    d = Experiment()  # the single source of defaults; help text quotes it
    p = argparse.ArgumentParser(
        description="Train a Neural Cellular Automaton to grow a target pattern."
    )

    src = p.add_argument_group("experiment source")
    src.add_argument("--config", default=None, metavar="TOML", help="sweep file (slurm/sweeps/*.toml)")
    src.add_argument("--run", default=None, metavar="NAME", help="run name inside --config")
    src.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="override any experiment field (e.g. --set lr=1e-3 --set targets=emoji:1f980,emoji:1f419)",
    )

    target = p.add_argument_group("target")
    target.add_argument(
        "--target", default=None, help=f"emoji:🦎 | emoji:1f98e | image:path.png | shape:heart (default {d.targets[0]})"
    )
    target.add_argument(
        "--targets",
        nargs="+",
        default=None,
        metavar="SPEC",
        help="several targets -> a conditional model with a cond_n-way one-hot selector",
    )
    target.add_argument("--target-size", type=int, default=None, help=f"target image size (default {d.target_size})")
    target.add_argument("--padding", type=int, default=None, help=f"border around the target (default {d.padding})")

    train_ = p.add_argument_group("training")
    train_.add_argument(
        "--experiment",
        choices=sorted(REGIMES),
        default=None,
        help=f"training regime (default {d.regime}): "
        + "; ".join(f"{r.name}={r.help.split(':')[0]}" for r in REGIMES.values()),
    )
    train_.add_argument("--no-pool", action="store_true", help="disable the sample pool")
    train_.add_argument("--damage-n", type=int, default=None, help="patterns to damage per batch")
    train_.add_argument(
        "--noise-n", type=int, default=None, help="pool samples per batch corrupted with state noise"
    )
    train_.add_argument("--noise-std", type=float, default=None, help="std of that noise (0.1-0.3 works)")
    train_.add_argument(
        "--mismatch-p",
        type=float,
        default=None,
        help="fraction of each batch asked to convert a grown target into another one (needs --targets)",
    )
    train_.add_argument("--pool-size", type=int, default=None, help=f"sample pool size (default {d.pool_size})")
    train_.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"patterns per step (default {d.batch_size}); memory scales with batch_size x --train-steps",
    )
    train_.add_argument("--iterations", type=int, default=None, help=f"training iterations (default {d.iterations})")
    train_.add_argument(
        "--train-steps",
        type=int,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help=f"CA steps unrolled per training step (default {d.unroll[0]} {d.unroll[1]})",
    )
    train_.add_argument("--lr", type=float, default=None, help=f"learning rate (default {d.lr})")
    train_.add_argument(
        "--lr-milestone",
        type=int,
        nargs="*",
        default=None,
        help="iteration(s) at which to decay the LR (default: 25% of iterations)",
    )
    train_.add_argument("--lr-gamma", type=float, default=None, help=f"LR decay factor (default {d.lr_gamma})")
    train_.add_argument("--no-grad-norm", action="store_true")
    train_.add_argument(
        "--amp",
        choices=["off", "bf16"],
        default=None,
        help="autocast dtype for the unroll (bf16 may be ~2x on compute-bound GPUs)",
    )
    train_.add_argument("--seed", type=int, default=None, help=f"RNG seed (default {d.seed})")

    model = p.add_argument_group("model")
    model.add_argument("--channels", type=int, default=None, help=f"state channels (default {d.channels})")
    model.add_argument("--hidden", type=int, default=None, help=f"update-rule hidden width (default {d.hidden})")
    model.add_argument(
        "--hidden-layers",
        type=int,
        default=None,
        help=f"hidden (ReLU) blocks / MLP depth (default {d.hidden_layers})",
    )
    model.add_argument("--fire-rate", type=float, default=None, help=f"stochastic update rate (default {d.fire_rate})")
    model.add_argument(
        "--learn-perception",
        action="store_true",
        help="learn the Sobel dx/dy perception kernels (identity stays fixed)",
    )
    model.add_argument("--no-wrap", action="store_true", help="zero (non-periodic) boundaries")

    run = p.add_argument_group("run control")
    run.add_argument("--device", default="auto")
    run.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    run.add_argument("--out-dir", default=None)
    run.add_argument("--run-name", default=None, help="deterministic run directory (runs/<name>)")
    run.add_argument("--resume", default=None, help="resume from a state.pt written by a previous run")
    run.add_argument("--log-every", type=int, default=None, help=f"log interval (default {d.log_every})")
    run.add_argument("--ckpt-every", type=int, default=None, help=f"checkpoint interval (default {d.ckpt_every})")
    run.add_argument(
        "--progress-every", type=int, default=None, help="render a batch montage every N iterations (0 = never)"
    )
    run.add_argument(
        "--save-pool",
        action="store_true",
        help="include the large sample pool in state.pt (exact resume; ~340 MB/ckpt)",
    )
    run.add_argument("--no-gif", action="store_true")
    run.add_argument("--gif-steps", type=int, default=None)
    return p


def experiment_for(args: argparse.Namespace) -> Experiment:
    """The experiment these arguments describe, in precedence order.

    ``--config/--run`` (the file) < explicit flags < ``--set k=v``.
    """
    if args.config:
        sweep = load_sweep(args.config)
        if args.run:
            base = sweep.by_name(args.run)
        elif len(sweep.runs) == 1:
            base = sweep.runs[0]
        else:
            raise SystemExit(f"{args.config} has {len(sweep.runs)} runs; pass --run <name>")
    else:
        if args.run:
            raise SystemExit("--run selects a run inside --config; give --config too")
        base = Experiment()

    overrides = {
        "targets": tuple(args.targets) if args.targets else None,
        "regime": args.experiment,
        "target_size": args.target_size,
        "padding": args.padding,
        "channels": args.channels,
        "hidden": args.hidden,
        "hidden_layers": args.hidden_layers,
        "fire_rate": args.fire_rate,
        "learn_perception": True if args.learn_perception else None,
        "wrap": False if args.no_wrap else None,
        "use_pool": False if args.no_pool else None,
        "damage_n": args.damage_n,
        "noise_n": args.noise_n,
        "noise_std": args.noise_std,
        "mismatch_p": args.mismatch_p,
        "pool_size": args.pool_size,
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "unroll": tuple(args.train_steps) if args.train_steps else None,
        "lr": args.lr,
        "lr_milestones": tuple(args.lr_milestone) if args.lr_milestone else None,
        "lr_gamma": args.lr_gamma,
        "normalize_grads": False if args.no_grad_norm else None,
        "amp": args.amp,
        "seed": args.seed,
        "log_every": args.log_every,
        "ckpt_every": args.ckpt_every,
    }
    if args.target:  # --target is a shorthand for a one-element --targets
        overrides["targets"] = (args.target,)
    exp = base.with_overrides(**overrides)

    if args.set:
        known = {f.name for f in fields(Experiment)}
        patch = {}
        for item in args.set:
            if "=" not in item:
                raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
            key, raw = item.split("=", 1)
            if key not in known:
                raise SystemExit(f"--set: unknown field {key!r}; known: {sorted(known)}")
            patch[key] = _coerce(key, raw)
        exp = experiment_from_dict({**exp.as_dict(), **patch})

    if args.run_name:
        exp = replace(exp, name=args.run_name)
    return exp


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = _resolve_device(args.device)
    exp = experiment_for(args)

    # --- targets ---------------------------------------------------------
    specs = list(exp.targets)
    target = prepare_targets(specs, max_size=exp.target_size, padding=exp.padding)
    _, _, h, w = target.shape
    seed = make_seed(h, w, exp.channels)

    # --- output directory ----------------------------------------------
    # An explicit --out-dir wins over the run's name (it is the more specific
    # instruction, and the tests/tools use it to keep runs/ clean).
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif exp.name:
        out_dir = Path("runs") / exp.name
    else:
        safe = specs[0].replace(":", "_").replace("/", "_")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = Path("runs") / f"{safe}_{exp.regime}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save a preview of the padded target(s) and the seed.
    save_image(tile(render(target), ncols=min(len(specs), 4)), out_dir / "target.png")
    save_image(render(seed.unsqueeze(0))[0], out_dir / "seed.png")
    np.save(out_dir / "target.npy", target.numpy())

    cfg = exp.train_config()
    if args.progress_every is not None:
        cfg = replace(cfg, progress_every=args.progress_every)
    if args.save_pool:
        cfg = replace(cfg, save_pool=True)
    if args.no_gif:
        cfg = replace(cfg, make_gif=False)
    if args.gif_steps is not None:
        cfg = replace(cfg, gif_steps=args.gif_steps)

    ca = NeuralCA(
        channel_n=exp.channels,
        fire_rate=exp.fire_rate,
        hidden_n=exp.hidden,
        wrap=exp.wrap,
        learn_perception=exp.learn_perception,
        cond_n=exp.cond_n,
        hidden_layers=exp.hidden_layers,
    )

    print(f"Targets: {', '.join(specs)}  ({exp.target_size}px, padded to {h}x{w})")
    if exp.cond_n:
        print(f"Conditional model: cond_n={exp.cond_n} one-hot selector")
    print(f"Experiment: {exp.name or '(unnamed)'}  regime={exp.regime}  iterations={exp.iterations}")
    train(
        ca,
        target,
        seed,
        cfg,
        device,
        out_dir,
        resume=args.resume,
        provenance={
            # The regime as a string (what meta.json has always recorded) plus the
            # resolved experiment, so an export or a paper can be rebuilt from runs/.
            "experiment": exp.regime,
            "experiment_config": exp.as_dict(),
            "target": specs[0],
            "targets": specs,
            "grid": [int(h), int(w)],
            # The canvas split, so an export can rebuild target.png on the same
            # grid even after target.npy is gone (grid = target_size + 2*padding).
            "target_size": exp.target_size,
            "padding": exp.padding,
        },
    )


if __name__ == "__main__":
    main()
