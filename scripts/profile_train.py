"""Micro-benchmark for the training loop.

Splits a training iteration into its pieces (pool sample/rank/damage, the
N-step unroll forward/backward, grad-norm, commit) and benchmarks a few
variants, so we can see where the time actually goes.

    uv run python scripts/profile_train.py --breakdown
    uv run python scripts/profile_train.py --variants --steps 80

Run it on a GPU node (the local box is CPU-only):

    srun -p mig-priority --constraint=mig --gres=gpu:1g.10gb:1 \
         --time=00:20:00 --mem=16G --cpus-per-task=4 \
         bash -lc 'cd ~/nca && uv run python scripts/profile_train.py --breakdown --variants'
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nca.model import NeuralCA
from nca.pool import SamplePool
from nca.targets import load_target, make_seed, prepare_target
from nca.train import TrainConfig, _train_step, make_circle_masks, per_sample_loss


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, dev: torch.device, n: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / n * 1000  # ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--pool-size", type=int, default=1024)
    ap.add_argument("--grid", type=int, default=72)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--variants", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    gen = torch.Generator(device="cpu").manual_seed(0)

    target_rgba = load_target("emoji:1f980", max_size=40)
    target = prepare_target(target_rgba, 16).to(dev)
    seed = make_seed(*target.shape[-2:], 16).to(dev)
    h = w = target.shape[-1]
    print(f"device={dev} grid={h}x{w} batch={args.batch} steps={args.steps}")

    ca = NeuralCA().to(dev)
    optimizer = torch.optim.Adam(ca.parameters(), lr=2e-3)
    cfg = TrainConfig(batch_size=args.batch, amp="off")
    pool = SamplePool(seed.unsqueeze(0).repeat(args.pool_size, 1, 1, 1).contiguous().to(dev))

    x0 = seed.unsqueeze(0).repeat(args.batch, 1, 1, 1).contiguous()
    n = args.steps

    if args.breakdown:
        print("\n== breakdown (ms) ==")

        def sample_rank():
            idx, xb = pool.sample(args.batch, gen)
            with torch.no_grad():
                rank = per_sample_loss(xb, target).argsort(descending=True)
            xb = xb[rank]
            idx = idx[rank]
            xb[0] = seed
            dmg = min(3, max(args.batch - 1, 0))
            if dmg:
                masks = make_circle_masks(dmg, h, w, gen).to(dev)
                xb[-dmg:] *= (1.0 - masks).unsqueeze(1)
            return idx, xb

        t_sample = timeit(sample_rank, dev, args.iters)
        print(f"pool.sample + rank + damage   {t_sample:8.2f}")

        idx, xb = sample_rank()
        t_fwd = timeit(lambda: forward_n(ca, xb, n), dev, args.iters)
        print(f"unroll forward  ({n} steps)    {t_fwd:8.2f}")

        def fwd_bwd():
            optimizer.zero_grad(set_to_none=True)
            x = forward_n(ca, xb, n)
            per_sample_loss(x, target).mean().backward()

        t_fb = timeit(fwd_bwd, dev, args.iters)
        print(f"unroll fwd+bwd  ({n} steps)    {t_fb:8.2f}")

        def gradnorm():
            for p in ca.parameters():
                if p.grad is not None:
                    p.grad /= p.grad.norm() + 1e-8

        t_gn = timeit(gradnorm, dev, args.iters)
        print(f"grad norm (6 params)          {t_gn:8.2f}")

        t_commit = timeit(lambda: pool.commit(idx, xb), dev, args.iters)
        print(f"pool.commit                   {t_commit:8.2f}")

        t_item = timeit(lambda: float(per_sample_loss(xb, target).mean().item()), dev, args.iters)
        print(f"loss.item() sync              {t_item:8.2f}")

    if args.variants:
        print("\n== variants (full _train_step, ms) ==")
        variants = []
        for batch in (16, 32, 64):
            for amp in ("off", "bf16"):
                variants.append((batch, amp, "eager"))
            variants.append((batch, "off", "reduce-overhead"))  # CUDA-graph hypothesis
        rows = []
        for batch, amp, mode in variants:

            def raw(x, nn, _ca=ca):
                for _ in range(nn):
                    x = _ca(x)
                return x

            fwd = raw if mode == "eager" else torch.compile(raw, mode=mode, dynamic=False)
            c = TrainConfig(batch_size=batch, amp=amp, train_steps_min=n, train_steps_max=n + 1)
            b = seed.unsqueeze(0).repeat(batch, 1, 1, 1).contiguous()
            try:
                ms = timeit(
                    lambda: _train_step(ca, fwd, optimizer, b, target, c, gen),
                    dev,
                    n=max(5, args.iters // 2),
                    warmup=3,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  batch={batch:3d} amp={amp:4s} {mode:15s} ERROR {e}")
                continue
            rows.append((batch, amp, mode, ms))
            print(
                f"  batch={batch:3d} amp={amp:4s} {mode:15s} {ms:8.2f} ms  "
                f"({1000 / ms:6.2f} it/s, {1000 / ms * batch:7.0f} samples/s)"
            )
        if rows:
            best = min(rows, key=lambda r: r[3])
            print(f"\nbest: batch={best[0]} amp={best[1]} {best[2]} -> {best[3]:.2f} ms")


def forward_n(ca: NeuralCA, x: torch.Tensor, n: int) -> torch.Tensor:
    for _ in range(n):
        x = ca(x)
    return x


if __name__ == "__main__":
    main()
