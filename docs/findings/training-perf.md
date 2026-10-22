# Training performance

Measured at 72×72 with an 80-step unroll. Regenerate the table with
`sbatch slurm/profile.sbatch` (`scripts/profile_train.py` is the local version).

## The unroll is throughput-bound, not latency-bound

Its time is *exactly* linear in batch size, so a larger batch buys nothing on a
1g MIG slice (and only ~1.8× on an under-utilised H100). Non-unroll code (pool
sample/rank/damage, grad norm, commit, `.item()`) is ~0.3 ms of a 54 ms (H100) /
429 ms (MIG) iteration — that is what makes `torch.compile` and CUDA graphs
irrelevant here.

| batch | MIG fp32 / bf16 | H100 fp32 / bf16 |
|---:|---|---|
| 8 | 235 / 150 ms | — |
| 16 | 429 / 331 ms (2.3 it/s) | 54 / 55 ms (18.6 it/s) |
| 24 | 642 / 534 ms | — |
| 32 | — | 85 / 74 ms |
| 64 | — | 156 / 126 ms |
| 128 | — | 302 / 244 ms |

Samples/s: MIG 34 → 37 → 37 (8 → 24); H100 298 → 432 → 510 → 525 (16 → 128).

Peak GPU memory at batch 16: **5.6 GB fp32 / 3.0 GB bf16**, linear in
`batch_size × train_steps` (autograd keeps every step's activations), so a
`1g.10gb` slice caps out near batch 24.

## Consequences

- **A full H100 is ~8× a MIG slice** at the same batch and schedule (8000 iters at
  batch 16: ~7 min vs ~54 min). Use an H100 for a single run; use the 14 MIG slices
  for a *sweep*, which parallelises better than one GPU serially.
- **`--amp bf16`**: ~1.3–1.55× on MIG and halves memory; neutral on H100 at batch
  16, ~1.2–1.3× at batch ≥ 32.
- **`torch.compile` does not help**: neutral on the module, slower on the whole
  sequence (warmup + per-`n` specialisation). CUDA graphs are untested and would
  only matter if the step were latency-bound, which it is not.

## What the loop does to stay out of the way

- Draws the whole unroll's fire mask up front (`torch.rand([n, B, 1, H, W])`)
  instead of one `rand_like` per step.
- Keeps losses as device scalars until a log/checkpoint needs the number, so there
  is no per-step `.item()` sync.
- Saves `state.pt` (model + optimizer + scheduler + RNG, optionally the pool with
  `--save-pool`) for exact resume; the pool is ~340 MB at the paper size, so it is
  opt-in and a resume without it restarts the pool from the seed (printed).
