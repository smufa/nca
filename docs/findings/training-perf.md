# Training performance

Measured at 72×72, hidden 128, `cond_n=8` (the perceptual sweep's shape) on a
`1g.10gb` MIG slice of an H100. Regenerate with
`python scripts/profile_train.py --breakdown --recipe` (local, CPU-only) or the
same script under `srun` for GPU numbers.

## The loop is 99.8% unroll — there is nothing else to win

Per iteration, batch 16, unroll 64/96:

| piece | u64 | u96 | share |
|---|---|---|---|
| pool.sample + rank + damage | 0.62 ms | 0.62 ms | 0.2% |
| unroll forward | 152.4 | 228.5 | 41% |
| backward | 216.6 | 325.7 | 58% |
| grad norm / optimiser / commit / `.item()` | 0.33 | 0.33 | 0.1% |
| **iteration** | **370.8** | **556.5** | 2.70 / 1.80 it/s |

So: no amount of cleverness in the pool, the ranking or the damage masks is worth
anything; only the CA step matters. Per CA step a *single sample* is 48.5 MMAC
(~97 MFLOP), so an iteration at unroll 80 is ~124 GFLOP for ~460 ms — the GPU is
running at roughly 8% of the 9.5 TFLOPS of its fp32 pipe.

## Why: 31 small kernels in series

Forward, batch 16, per CA step (`--breakdown` op table):

| kernel | ms/step | share |
|---|---|---|
| cudnn conv ×2 (the `W1`/`W2` 1×1 convs) | 0.50 | 21% |
| perception depthwise conv | 0.46 | 19% |
| 2× bias `add_` (from those convs) | 0.45 | 19% |
| relu | 0.36 | 15% |
| gemm (`W2`, tf32) | 0.28 | 12% |
| `cat` (appending the one-hot cond) | 0.22 | 9% |
| circular `pad` ×3 (1 perception + 2 living masks) | 0.10 | 4% |
| `mul` ×3, `add` | 0.22 | 9% |
| **total** | **2.37** | |

Almost half a step is plumbing (cat/bias/relu/pad), and every tensor is only
~40 MB, so each kernel is short-lived: the slice is at ~8% of its fp32 peak and
~10% of its bandwidth. It is **latency-bound on a chain of small kernels**, but
with too little work in each to fill the GPU — which is why batching helps only
until the SMs are busy, and why fusion helps.

## What actually buys time

Whole iteration, MIG slice:

| config | ms | sample-CA-steps/s | peak GiB |
|---|---|---|---|
| b16 u80 eager fp32 | 462.9 | 2765 | 5.71 |
| b16 u80 `torch.compile` | 455.1 | 2812 | 5.71 |
| b16 u80 **bf16** | 379.6 | 3372 | **3.20** |
| b32 u48 eager fp32 | 552.4 | 2781 | 6.85 |
| b32 u48 **bf16 + compile** | **280.5** | 5475 | 3.75 |
| b64 u24 bf16 + compile | 247.3 | 6211 | 3.81 |

- **bf16 is the cheap win**: 1.22× alone and it halves memory (5.71 → 3.20 GiB),
  which is what makes `batch × unroll` fit at all. It is now the default on CUDA
  (`amp="auto"`); `--amp off` restores fp32.
- **`torch.compile` alone is neutral** (455 vs 463, forward 7% slower); with bf16
  it compounds to **1.65–1.97×**. It is opt-in (`--compile`), because it costs
  ~10 s of warmup per shape and buys nothing without bf16.
- **Batch and unroll are not throughput knobs.** Eager throughput saturates at
  batch 16: 79 → 87 → 87.3 → 87.5 samples/s for batch 8/16/32/64 at unroll 32.
  Bigger batches cost wall time and buy only credit assignment and memory use.

Peak GPU memory at batch 16: **5.7 GB fp32 / 3.2 GB bf16**, linear in
`batch_size × train_steps` (autograd keeps every step's activations), so a
`1g.10gb` slice caps out near batch 16 × unroll 80 in fp32, ~b32 × u48 in bf16.

## Consequences

- **A full H100 is ~8× a MIG slice** at the same batch and schedule. Use an H100
  for a single run; use the 14 MIG slices for a *sweep*.
- **The step-change would be a fused kernel.** One dispatch for
  perception+MLP+masks, as `frontend/src/wgsl/pass1.wgsl` already does — the
  shader is roughly an order of magnitude cheaper per cell-step than this eager
  loop, because it never materialises the 9-offset concat or the intermediate
  bias tensors, and it skips non-firing cells entirely. Fusing in Triton is the
  only path to 5-10×; it means hand-writing forward *and* backward for the MLP.
- Small, safe shavings left if ever needed: fold the two conv biases (0.45 ms),
  and the cond `cat` (0.22 ms) by pre-adding `W_cond·cond` into the first conv's
  bias once per iteration (~9% of a step, but it changes the graph the shader and
  exporter assume, `3C + cond_n` fan-in, so not worth it while exports exist).

## What the loop does to stay out of the way

- Draws the whole unroll's fire mask up front (`torch.rand([n, B, 1, H, W])`)
  instead of one `rand_like` per step.
- Keeps losses as device scalars until a log/checkpoint needs the number, so there
  is no per-step `.item()` sync.
- Saves `state.pt` (model + optimizer + scheduler + RNG, optionally the pool with
  `--save-pool`) for exact resume; the pool is ~340 MB at the paper size, so it is
  opt-in and a resume without it restarts the pool from the seed (printed).

## History

The old version of this file claimed "the unroll is throughput-bound, not
latency-bound: its time is exactly linear in batch size", and concluded that
`torch.compile` was irrelevant. Both came from a profiling script with a bug: the
per-iteration helper read `args.batch` instead of the loop variable, so every
"batch size" ran the same workload. With the script fixed, time is linear in batch
only *below* saturation (batch ≤ 16), and compile+bf16 is worth ~2×.
