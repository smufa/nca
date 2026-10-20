# Training on the ARNES SLURM cluster

Plan for training the Neural CA "properly" on `ssh arnes`.

## Cluster facts

- Login node `hpc-login4.arnes.si`, user `es6878`, AlmaLinux 9.8, Lmod modules.
- SLURM partitions: `gpu` (4-day limit), `long` (7-day), `all` (2-day).
- **GPU nodes are mixed:**
  - H100 (Hopper, sm_90): `gwn[01-06]` (2 GPU), `gwn[08-10]` (4 GPU, `sxm`)
  - V100S (Volta, sm_70): `wn[201-212,217-221]` (2 GPU)
- Driver **580.178.04 / CUDA 13.0**.
- Login node: 48 cores, 62 GB RAM, **internet works**, `uv 0.11.21` already
  installed at `~/.local/bin/uv`, Apptainer available, no conda.
- `$HOME=/d/hpc/home/es6878` on a large `/d` filesystem.

### Critical gotcha: pin H100

A plain `srun -p gpu --gres=gpu:1` can land on a **V100S**. CUDA 13 dropped
Volta, so the cu13 torch wheels we get from PyPI **will not run there**.

> Always add `--constraint=h100` (equivalently `-C h100`) to GPU jobs.

H100 is sm_90 and matches driver 580 / cu13, so the default PyPI torch build is
exactly right for it.

## Environment

`requires-python` was relaxed to `>=3.12` so the cluster can use stable Python
3.13 (uv would otherwise pull a 3.14 release candidate).

Local env is unchanged (CPU torch via `[tool.uv.sources]`). On the cluster:

```bash
cd ~/nca
uv sync --no-sources --python 3.13 --group dev
```

- `--no-sources` ignores the local CPU torch pin → PyPI `torch==2.14.1` + cu13.
- This rewrites the **cluster's** `uv.lock`; keep it there and never sync it back
  to the laptop (the laptop lock stays CPU-pinned).
- Run it once on the login node; the `.venv` is on `/d` and shared with compute
  nodes.

Fallback if we want zero lock divergence: `uv venv --python 3.13` +
`uv pip install --torch-backend cu130 -r requirements.txt`.

## Code changes needed before the first real run

1. **Device / perf knobs** (`nca/cli.py`, `nca/train.py`)
   - `--device cuda` already resolves; add `--amp {off,bf16,fp16}` (default
     `off` — the model is tiny and launch-bound, not FLOP-bound).
   - `torch.backends.cudnn.benchmark = True`, `torch.set_float32_matmul_precision('high')`.
   - Optional `--compile` to wrap the step in `torch.compile` (the 64–96-step
     Python loop is the bottleneck, not the GPU).
2. **Resume** — save/load optimizer + step (`--resume ckpt.pt`); long jobs and
   any preemption.
3. **Logging** — write `loss.json` (currently only an in-memory list + PNG), and
   flush stdout so `squeue`/`tail` monitoring works.
4. **Offline targets** — compute nodes have no internet. Pre-download emoji PNGs
   on the login node (or commit `assets/targets/`) and support `image:` specs.
5. **Run naming** — deterministic `--run-name` so sweeps produce predictable dirs.

## SLURM scripts

`slurm/train.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=nca
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python main.py "$@"
```

`slurm/submit_sweep.sh` submits one job per (target, experiment):

```bash
#!/bin/bash
set -euo pipefail
mkdir -p logs
for target in "emoji:🦎" "emoji:😀" "emoji:🦋" "emoji:🐠"; do
  for exp in growing persistent regenerating; do
    id="${target#emoji:}_${exp}"
    sbatch slurm/train.sbatch \
      --target "$target" --experiment "$exp" \
      --target-size 40 --padding 16 \
      --batch-size 16 --iterations 8000 --train-steps 64 96 \
      --run-name "$id" --device cuda
  done
done
```

## Suggested H100 config (paper settings, scaled)

| knob | value | why |
| --- | --- | --- |
| target-size / padding | 40 / 16 (72×72 grid) | paper setting |
| batch-size | 16–32 | H100 is underutilised at batch 8 |
| iterations | 8000 | paper |
| train-steps | 64–96 | paper |
| pool-size / damage | 1024 / 0 or 3 | persistent / regenerating |
| lr | 2e-3, milestone 2000 | paper |
| checkpoint-every | 250 | cheap, safe |

Expect roughly tens of minutes per model (the CA is tiny; the Python unroll
dominates), so a full sweep of 12 models is an afternoon.

## Transfer

Repo is not committed/pushed yet. Either:

- **rsync**: `rsync -az --exclude .venv --exclude runs --exclude node_modules \
  ~/Projects/nca/ arnes:~/nca/`
- or commit and `git clone` on the cluster.

Export ONNX on the **login node** after training (CPU, uses the `dev` group),
then rsync `exports/` back and commit it for the frontend.

## Workflow

1. rsync repo → cluster.
2. `uv sync --no-sources --python 3.13 --group dev` on the login node.
3. **Smoke test**: one short job (100 iters, small grid) with `srun` to confirm
   H100 + cu13 torch import and a training step.
4. Launch the sweep with `submit_sweep.sh`.
5. Monitor: `squeue -u $USER`, `tail -f logs/*.out`, `sacct -j <id>`.
6. Export the best checkpoints to ONNX on the login node.
7. rsync `exports/` back, commit, open the frontend.

## Milestones

1. **T1** — env on cluster + H100 smoke test (imports, 100 iters).
2. **T2** — one high-quality run (lizard, regenerating) + ONNX export.
3. **T3** — sweep experiments 1–3 across targets.
4. **T4** — compare/curate models, ship to the frontend.

## Open questions

- Transfer via rsync or push to a git remote?
- Keep all checkpoints, or only the final + best-by-loss?
- Any preference for a specific emoji/target set for the first sweep?
