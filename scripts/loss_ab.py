#!/usr/bin/env python
"""A/B two loss configurations on one target: does the extra term earn it?

The question "is the perceptual loss better?" cannot be answered by the training
curve, because the curve *is* the loss being changed -- adding a term raises it by
construction. This trains N configurations with the same target/seed/schedule and
scores the result on measures none of them optimises directly:

``pixel``    MSE over RGBA (the historical objective; lower is better)
``psnr``     peak signal-to-noise of the composited RGB in dB (higher is better)
``ssim``     structural similarity of the composited RGB (higher is better)
``edge``     L1 between Sobel gradients, relative to the target's own edge energy
             (0 = identical edges; the perceptual terms optimise this, so it is the
             one most likely to improve for the wrong reason)
``alpha``    IoU of the silhouette rounded at 0.5 (structure, not colour)

Two modes, because they answer different questions and one of them is cheap:

``--mode repair`` (default)
    Start from the target with a disc erased, optimise the CA to restore it, then
    score the restoration. Cheap and discriminating: the CA begins next to a
    solution, so a few hundred CPU iterations say a lot about the *attractor* the
    loss builds (a sharp repaired pattern, or a blurred average?).

``--mode grow``
    The real regime: grow from a seed with a sample pool and damage. Needs a
    converged run (hours on CPU), otherwise every configuration looks equally bad
    and the comparison says nothing -- it is here for GPU use.

Usage::

    python -m scripts.loss_ab --target shape:heart --compare pixel \\
        --compare "laplacian=0.5,ssim=0.1,edge=0.05,start=0.2,ramp=0.2"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from nca.experiments import Experiment
from nca.losses import Loss, LossConfig
from nca.model import NeuralCA
from nca.step import make_seed
from nca.targets import prepare_targets
from nca.train import train


def _metrics(state: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Score a state against its target, on measures the loss may not use."""
    loss = Loss(LossConfig())  # metric-only: the terms are read, never optimised
    pred_rgb = loss.composite(state[:, :4])
    tgt_rgb = loss.composite(target)
    mask = loss._alpha_mask(state, target)

    pixel = float((state[:, :4] - target).pow(2).mean())
    mse = float((pred_rgb - tgt_rgb).pow(2).mean())
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
    ssim = float(1.0 - loss._ssim(pred_rgb * mask, tgt_rgb * mask).mean())
    edge = float(loss._edge(pred_rgb, tgt_rgb, mask).mean())
    a, b = state[:, 3] > 0.5, target[:, 3] > 0.5
    iou = float((a & b).sum() / max((a | b).sum(), 1))
    return {"pixel": pixel, "psnr": psnr, "ssim": ssim, "edge": edge, "alpha": iou}


def _damage(target: torch.Tensor, radius: float = 6.0) -> torch.Tensor:
    """``target`` with a disc erased: the repair starting point."""
    _, _, h, w = target.shape
    yy, xx = torch.meshgrid(
        torch.arange(h).float() - h / 2 + 0.5,
        torch.arange(w).float() - w / 2 + 0.5,
        indexing="ij",
    )
    hole = (xx**2 + yy**2) < radius**2
    out = target.clone()
    out[..., hole] = 0.0
    return out


def _parse(text: str) -> dict | None:
    """``pixel`` -> None (the historical objective), ``k=v,...`` -> a loss table."""
    if text.strip().lower() == "pixel":
        return None
    out: dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, raw = item.partition("=")
        out[key.strip()] = float(raw)
    LossConfig.from_dict(out)  # fail here, not an hour into a sweep
    return out


def _unroll(ca: NeuralCA, x: torch.Tensor, steps: int, fire: torch.Tensor) -> torch.Tensor:
    for k in range(steps):
        x = ca(x, update_mask=fire[k])
    return x


def _repair(
    ca: NeuralCA,
    x0: torch.Tensor,
    target: torch.Tensor,
    cfg,
    device: torch.device,
    iterations: int,
    unroll: tuple[int, int],
    log_every: int,
) -> torch.Tensor:
    """Optimise the CA to restore ``x0``; return the grown state after ``x0``."""
    criterion = Loss(cfg.loss) if cfg.loss else None
    if criterion is not None:
        criterion.calibrate(x0, target)
    optimizer = torch.optim.Adam(ca.parameters(), lr=cfg.lr)
    generator = torch.Generator(device=device.type).manual_seed(cfg.seed)
    start = time.time()
    for step in range(iterations):
        steps = int(torch.randint(unroll[0], unroll[1] + 1, (1,), generator=generator))
        fire = torch.rand((steps, 1, 1, *x0.shape[-2:]), device=device, generator=generator) <= ca.fire_rate
        x = _unroll(ca, x0, steps, fire)
        if criterion is None:
            from nca.train import loss_fn

            loss = loss_fn(x, target)
        else:
            loss, _ = criterion(x, target, step / max(iterations - 1, 1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.normalize_grads:
            grads = [p.grad for p in ca.parameters() if p.grad is not None]
            if grads:
                norms = torch._foreach_norm(grads)
                torch._foreach_div_(grads, [n + 1e-8 for n in norms])
        optimizer.step()
        if log_every and (step % log_every == 0 or step == iterations - 1):
            rate = (step + 1) / max(time.time() - start, 1e-9)
            print(f"    step {step + 1:5d}/{iterations} | loss {loss.item():.5f} | {rate:.1f} it/s", flush=True)
    # Recover: no gradient, no fire-mask randomness for the comparison itself.
    ca.eval()
    with torch.no_grad():
        x = x0.clone()
        for _ in range(max(unroll)):
            x = ca(x)
    ca.train()
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default="shape:heart")
    ap.add_argument("--compare", action="append", required=True, metavar="LOSS")
    ap.add_argument("--mode", choices=("repair", "grow"), default="repair")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--target-size", type=int, default=32)
    ap.add_argument("--padding", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="/tmp/loss_ab")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_root = Path(args.out)
    target = prepare_targets([args.target], max_size=args.target_size, padding=args.padding)[0].unsqueeze(0)
    grid = target.shape[-1]
    print(f"{args.target} on {grid}x{grid}, mode={args.mode}, {args.iterations} iters, device={device}\n")

    rows = []
    for index, spec in enumerate(args.compare):
        loss_table = _parse(spec)
        label = "pixel" if loss_table is None else ",".join(f"{k}={v}" for k, v in loss_table.items())
        run_dir = out_root / f"run{index}"
        exp = Experiment(
            name=run_dir.name,
            targets=(args.target,),
            target_size=args.target_size,
            padding=args.padding,
            channels=args.channels,
            hidden=args.hidden,
            iterations=args.iterations,
            batch_size=args.batch_size,
            unroll=(48, 64),
            seed=args.seed,
            loss=loss_table,
        )
        ca = NeuralCA.from_spec(exp.spec()).to(device)
        torch.manual_seed(args.seed)
        if args.mode == "repair":
            # A state, not a target: channels 0:4 are the target, the rest is hidden
            # state the CA must learn to build. The disc is erased from all channels.
            x0 = torch.zeros(1, args.channels, grid, grid, device=device)
            x0[:, :4] = target.to(device)
            x0 = _damage(x0)
            state = _repair(
                ca, x0, target.to(device), exp.train_config(), device,
                args.iterations, exp.unroll, max(1, args.iterations // 5),
            )
        else:
            state = None
            train(
                ca,
                target,
                make_seed(grid, grid, args.channels),
                exp.train_config(),
                device,
                run_dir,
                provenance={"target": args.target, "experiment": exp.regime, "loss": loss_table},
            )
            ca.eval()
            with torch.no_grad():
                x = torch.from_numpy(make_seed(args.channels, grid, grid)).to(device)
                for _ in range(200):
                    x = ca(x)
            state = x

        m = _metrics(state.cpu(), target)
        rows.append((label, m))
        (run_dir / "ab_metrics.json").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "ab_metrics.json").write_text(json.dumps({**m, "loss": loss_table}, indent=2))
        print(f"  {label:44s} psnr {m['psnr']:5.2f} dB  ssim {m['ssim']:.4f}  edge {m['edge']:.3f}  alpha {m['alpha']:.4f}")

    print("\nmetric       " + "".join(f"{label[:18]:>20s}" for label, _ in rows))
    for key in ("pixel", "psnr", "ssim", "edge", "alpha"):
        print(f"{key:12s} " + "".join(f"{m[key]:20.4f}" for _, m in rows))


if __name__ == "__main__":
    main()
