"""Micro-benchmark for the training loop: where does one iteration go?

Splits an iteration into its pieces (pool sample/rank/damage, the N-step unroll
forward and backward, grad-norm, optimiser step, commit) and prints milliseconds
per iteration plus a per-CA-step view. Defaults mirror the perceptual sweep:
`hidden=128`, 8 conditional targets, 72x72 canvas, `batch=16`, unroll 64-96.

    python scripts/profile_train.py --breakdown
    python scripts/profile_train.py --breakdown --unroll 96 --batch 32

The local box is CPU-only; for the numbers that matter, run it on a GPU node:

    srun -p mig-priority --constraint=mig --gres=gpu:1g.10gb:1 \
         --time=00:20:00 --mem=16G --cpus-per-task=4 \
         bash -lc 'cd ~/nca && uv run python scripts/profile_train.py --breakdown'

A note on what the ratio means: one CA step at this size is ~8 MFLOP per sample,
i.e. ~1.3 us of GPU arithmetic, so each of the ~80 launches per iteration costs
more in dispatch than in maths. If `fwd` here is far above `batch x steps x
1.3 us`, the loop is launch-bound, not FLOP-bound.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nca.model import NeuralCA
from nca.pool import SamplePool
from nca.spec import ModelSpec
from nca.step import make_seed
from nca.targets import prepare_targets
from nca.train import TrainConfig, make_circle_masks, per_sample_loss

TARGETS = [
    "emoji:1f980", "emoji:1f419", "emoji:1f996", "emoji:1f422",
    "emoji:1f335", "emoji:1f344", "emoji:1f41d", "emoji:1f9a9",
]


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
    ap.add_argument("--unroll", type=int, default=80, help="CA steps per iteration")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--cond", type=int, default=8, help="conditional targets (0 = unconditional)")
    ap.add_argument("--target-size", type=int, default=40)
    ap.add_argument("--padding", type=int, default=16)
    ap.add_argument("--pool-size", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument(
        "--variants", action="store_true", help="also time batching / unroll / amp variants"
    )
    ap.add_argument("--compile", action="store_true", help="also time torch.compile over the unroll")
    ap.add_argument("--amp", action="store_true", help="also time bf16 autocast")
    ap.add_argument("--recipe", action="store_true", help="time the combined knob settings")
    ap.add_argument(
        "--batch-sweep", action="store_true",
        help="time whole iterations at several batch sizes (fixed unroll)",
    )
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    gen = torch.Generator(device="cpu").manual_seed(0)

    specs = TARGETS[: args.cond] if args.cond else TARGETS[:1]
    targets = prepare_targets(specs, max_size=args.target_size, padding=args.padding).to(dev)
    target = targets[:1]  # per-sample loss against one target keeps it comparable
    spec = ModelSpec(hidden_n=args.hidden, cond_n=args.cond)
    ca = spec.build().to(dev)
    h, w = targets.shape[-2:]
    # `ca(x, cond=...)` broadcasts the conditioning rows over the batch dimension,
    # so one hot vector per sample is enough; cache them per batch size.
    _conds: dict[int, torch.Tensor] = {}

    def cond_for(batch: int):
        if not spec.cond_n:
            return None
        if batch not in _conds:
            _conds[batch] = (
                torch.eye(spec.cond_n)[:1].reshape(1, spec.cond_n, 1, 1).repeat(batch, 1, 1, 1).to(dev)
            )
        return _conds[batch]

    optimizer = torch.optim.Adam(ca.parameters(), lr=2e-3)
    seed = torch.from_numpy(make_seed(spec.channel_n, h, w)).float().to(dev).unsqueeze(0)
    pool = SamplePool(seed.repeat(args.pool_size, 1, 1, 1).contiguous().to(dev))
    x0 = seed.repeat(args.batch, 1, 1, 1).contiguous()
    n = args.unroll
    dmg = 3
    noise = 1

    print(
        f"device={dev} grid={h}x{w} batch={args.batch} unroll={n} hidden={args.hidden} "
        f"cond={args.cond} params={ca.num_parameters}"
    )
    # multicast-accumulates per cell: perception taps + W1 + W2 (perception is
    # `3C` offsets, but only `C` taps because the kernel is depth-shared).
    macs = spec.channel_n * 9 + (3 * spec.channel_n + spec.cond_n) * spec.hidden_n
    macs += spec.hidden_n * spec.channel_n
    print(
        f"arithmetic per CA step per sample: {macs * h * w / 1e6:.1f} MMAC "
        f"({2 * macs * h * w / 1e9:.2f} GFLOP/sample)"
    )

    def fwd(x=x0):
        with torch.no_grad():
            for _ in range(n):
                x = ca(x, cond=cond_for(x.shape[0]))
        return x

    def fwd_bwd():
        optimizer.zero_grad(set_to_none=True)
        x = x0
        for _ in range(n):
            x = ca(x, cond=cond_for(x.shape[0]))
        per_sample_loss(x, target).mean().backward()

    def sample_rank(batch: int, src=None):
        idx, xb, _pool_cond = (src or pool).sample(batch, gen)
        with torch.no_grad():
            rank = per_sample_loss(xb, target).argsort(descending=True)
        xb = xb[rank].clone()
        idx = idx[rank]
        xb[0] = seed
        if dmg:
            masks = make_circle_masks(min(dmg, batch - 1), h, w, gen).to(dev)
            xb[-(dmg):] *= (1.0 - masks).unsqueeze(1)
        if noise:
            xb[-(dmg + noise) : -dmg] += torch.randn(
                noise, *xb.shape[1:], generator=gen
            ).to(dev) * 0.1
        return idx, xb

    def one_iteration():
        idx, xb = sample_rank()
        optimizer.zero_grad(set_to_none=True)
        x = xb
        for _ in range(n):
            x = ca(x, cond=cond_for(x.shape[0]))
        loss = per_sample_loss(x, target).mean()
        loss.backward()
        for p in ca.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        optimizer.step()
        pool.commit(idx, xb)
        return float(loss.item())

    if args.breakdown:
        print("\n== breakdown (ms per iteration) ==")
        total = timeit(one_iteration, dev, args.iters)
        parts = {
            "pool.sample + rank + damage": timeit(sample_rank, dev, args.iters),
            f"unroll forward ({n} steps)": timeit(fwd, dev, args.iters),
            f"unroll fwd+bwd ({n} steps)": timeit(fwd_bwd, dev, args.iters),
        }
        parts["backward (derived)"] = parts[f"unroll fwd+bwd ({n} steps)"] - parts[f"unroll forward ({n} steps)"]

        def gradnorm():
            for p in ca.parameters():
                if p.grad is not None:
                    p.grad /= p.grad.norm() + 1e-8

        fwd_bwd()  # leave gradients in place for the timed pieces below
        parts["grad norm"] = timeit(gradnorm, dev, args.iters)
        parts["optimizer.step"] = timeit(lambda: optimizer.step(), dev, args.iters)
        idx, xb = sample_rank()
        parts["pool.commit"] = timeit(lambda: pool.commit(idx, xb), dev, args.iters)
        parts["loss.item() (sync)"] = timeit(
            lambda: float(per_sample_loss(xb, target).mean().item()), dev, args.iters
        )
        parts["unrolled step (fwd only)"] = parts[f"unroll forward ({n} steps)"] / n
        parts["unrolled step (fwd+bwd)"] = parts[f"unroll fwd+bwd ({n} steps)"] / n

        width = max(len(k) for k in parts)
        for k, v in parts.items():
            share = "" if "step (" in k else f"  {100 * v / total:5.1f}% of {total:.0f}"
            print(f"  {k:{width}s}  {v:8.2f}{share}")
        print(f"  {'whole iteration':{width}s}  {total:8.2f}")
        print(f"  {'=> iterations/s':{width}s}  {1000 / total:8.2f}")
        if dev.type == "cuda":
            print(f"  {'peak memory':{width}s}  {torch.cuda.max_memory_allocated() / 2**30:8.2f} GiB")

    if args.compile or args.amp:
        print("\n== forwards: eager vs compiled vs bf16 ==")
        eager = timeit(fwd, dev, max(args.iters // 2, 5))
        print(f"  {'eager':12s} {eager:8.2f} ms  {eager / n:6.2f} ms/step")
        model = ca
        if args.compile:
            try:
                model_c = torch.compile(ca, mode="default")
                # One warm-up call per shape compiles the graph; the timed runs replay it.
                t0 = time.perf_counter()
                with torch.no_grad():
                    model_c(x0, cond=cond_for(args.batch))
                print(f"  {'compile+warmup':12s} {1000 * (time.perf_counter() - t0):8.0f} ms (one-off)")

                def fwd_c():
                    with torch.no_grad():
                        x = x0
                        for _ in range(n):
                            x = model_c(x, cond=cond_for(x.shape[0]))
                    return x

                ms = timeit(fwd_c, dev, max(args.iters // 2, 5))
                print(f"  {'compile':12s} {ms:8.2f} ms  {ms / n:6.2f} ms/step  ({eager / ms:.2f}x)")
            except Exception as exc:  # pragma: no cover - dynamic (noqa: BLE001)
                print(f"  {'compile':12s} FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")
        if args.amp and dev.type == "cuda":
            def fwd_amp():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    x = x0
                    for _ in range(n):
                        x = model(x, cond=cond_for(x.shape[0]))
                return x

            ms = timeit(fwd_amp, dev, max(args.iters // 2, 5))
            print(f"  {'bf16 autocast':12s} {ms:8.2f} ms  {ms / n:6.2f} ms/step  ({eager / ms:.2f}x)")
            print(f"  {'peak memory':12s} {torch.cuda.max_memory_allocated() / 2**30:8.2f} GiB")

    if args.batch_sweep:
        print("\n== whole iteration vs batch (is the loop dispatch-bound?) ==")
        model = torch.compile(ca, mode="default") if args.compile else ca
        for batch in (8, 16, 32, 64):
            this_pool = SamplePool(seed.repeat(args.pool_size, 1, 1, 1).contiguous().to(dev))

            def it(batch=batch, this_pool=this_pool):
                idx, xb2 = sample_rank(batch, this_pool)
                optimizer.zero_grad(set_to_none=True)
                x = xb2
                for _ in range(n):
                    x = model(x, cond=cond_for(x.shape[0]))
                per_sample_loss(x, target).mean().backward()
                optimizer.step()
                this_pool.commit(idx, xb2)
                return None

            try:
                with torch.no_grad():
                    model(seed.repeat(batch, 1, 1, 1), cond=cond_for(batch))  # warm the graph
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                ms = timeit(it, dev, max(args.iters // 2, 5))
                print(
                    f"  batch={batch:3d}  {ms:8.2f} ms  {1000 / ms:6.2f} it/s"
                    f"  {batch / (ms / 1000):7.1f} samples/s"
                    f"  {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB"
                )
            except RuntimeError as exc:
                print(f"  batch={batch:3d}  FAILED: {str(exc).splitlines()[0][:70]}")

    if args.recipe:
        print("\n== recipes: batch x unroll x compile x bf16 (whole iteration) ==")
        print(f"  {'config':34s} {'ms':>8s} {'it/s':>7s} {'sample-steps/s':>15s} {'GiB':>6s}")
        for batch, unroll, comp, amp in [
            (16, 80, False, False),   # what the sweep ran
            (16, 80, True, False),
            (16, 80, False, True),
            (32, 48, False, False),
            (32, 48, True, True),
            (64, 24, True, True),
            (64, 48, True, True),
        ]:
            n = unroll
            this_pool = SamplePool(seed.repeat(args.pool_size, 1, 1, 1).contiguous().to(dev))
            model = torch.compile(ca, mode="default") if comp else ca
            ctx = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if amp and dev.type == "cuda"
                else contextlib.nullcontext()
            )

            def it(batch=batch, this_pool=this_pool, n=n, model=model, ctx=ctx):
                idx, xb2 = sample_rank(batch, this_pool)
                optimizer.zero_grad(set_to_none=True)
                with ctx:
                    x = xb2
                    for _ in range(n):
                        x = model(x, cond=cond_for(x.shape[0]))
                per_sample_loss(x.float(), target).mean().backward()
                optimizer.step()
                return None

            label = f"b{batch} u{unroll}{' compile' if comp else ''}{' bf16' if amp else ''}"
            try:
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                ms = timeit(it, dev, 5)
                gib = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
                print(
                    f"  {label:34s} {ms:8.1f} {1000 / ms:7.2f} {batch * unroll / (ms / 1000):15.0f} {gib:6.2f}"
                )
            except RuntimeError as exc:
                print(f"  {label:34s} FAILED: {str(exc).splitlines()[0][:40]}")

    if args.variants:
        print("\n== variants (whole iteration) ==")
        for batch in (8, 16, 32, 64):
            this_pool = SamplePool(seed.repeat(args.pool_size, 1, 1, 1).contiguous().to(dev))

            def variant_iter(batch=batch, this_pool=this_pool):
                idx, xb2 = sample_rank(batch, this_pool)
                optimizer.zero_grad(set_to_none=True)
                x = xb2
                for _ in range(n):
                    x = ca(x, cond=cond_for(x.shape[0]))
                per_sample_loss(x, target).mean().backward()
                optimizer.step()
                this_pool.commit(idx, xb2)
                return None

            try:
                ms = timeit(variant_iter, dev, max(args.iters // 2, 5))
                print(f"  batch={batch:3d} unroll={n:3d}  {ms:8.2f} ms  {1000 / ms:6.2f} it/s"
                      f"  {batch / (ms / 1000):7.1f} sample-steps/s")
            except RuntimeError as exc:
                print(f"  batch={batch:3d} unroll={n:3d}  FAILED: {str(exc).splitlines()[0][:60]}")


if __name__ == "__main__":
    main()
