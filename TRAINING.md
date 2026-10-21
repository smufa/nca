# Training on the ARNES SLURM cluster

How to train the Neural CA on `ssh arnes` (H100 / MIG): environment, sweep
scripts, and the gotchas that cost time to find.

## Cluster facts

- Access is **`ssh arnes`** (user `es6878`). It multiplexes over a TOTP-authenticated
  session (`ControlMaster` in the user's `~/.ssh/config`); which login node it lands
  on varies (`hpc-login1/3/4`). **When the master session expires, fresh
  connections fail** with `Permission denied (gssapi-with-mic,keyboard-interactive)`
  until it is re-established. Don't force a fresh connection (`-o ControlPath=none`)
  — it cannot authenticate.
- The cluster has **no GitHub SSH key** — clone over **HTTPS**.
- SLURM partitions: `gpu` (4-day), `long` (7-day), `all` (2-day), plus
  **`mig-priority` / `mig-preempt`** (see below).
- **GPU nodes are mixed:**
  - H100 (Hopper, sm_90): `gwn[01-06]` (2 GPU), `gwn[08-10]` (4 GPU, `sxm`)
  - V100S (Volta, sm_70): `wn[201-212,217-221]` (2 GPU)
  - MIG slices (H100): `gwn07` — 14 × `gpu:1g.10gb`
- Driver **580.178.04 / CUDA 13.0**.
- Login node: 48 cores, 62 GB RAM, **internet works**, `uv 0.11.21` at
  `~/.local/bin/uv`, Apptainer available, no conda.
- `$HOME=/d/hpc/home/es6878` on a large `/d` filesystem.

### Critical gotcha: pin H100

A plain `srun -p gpu --gres=gpu:1` can land on a **V100S**. CUDA 13 dropped
Volta, so the cu13 torch wheels we get from PyPI **will not run there**.

> Always add `--constraint=h100` (equivalently `-C h100`) to GPU jobs.

H100 is sm_90 and matches driver 580 / cu13, so the default PyPI torch build is
exactly right for it. `slurm/train.sbatch` already pins it.

### MIG slices (when the H100s are busy)

The `gpu` partition is usually fully allocated. `gwn07` exposes **14 H100 MIG
slices** via `mig-priority` / `mig-preempt` (1-day limit, all accounts allowed):

```bash
sbatch -p mig-priority --constraint=mig --gres=gpu:1g.10gb:1 \
       --mem=16G --cpus-per-task=4 slurm/train.sbatch ...
```

- `gwn07` has **no `h100` feature** — pass `--constraint=mig`, not `h100`.
- It is still a real H100 (sm_90), so the cu13 torch build works.
- Measured: **~2.5 it/s** at 72×72 / batch 16 → **~54 min per 8000-iteration run**.
- The node has 250 GB RAM, so `train.sbatch`'s default `--mem=64G` fits only **3**
  jobs. `slurm/submit_sweep.sh` overrides `MEM=16G CPUS=4` so all 14 slices run at
  once (an 18-job sweep then takes ~2 h).

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

### Gotchas found

- The Lmod environment exports `PYTHONPATH` pointing at a Python 3.12
  site-packages tree on cvmfs, which shadows the venv's `numpy`/`torch`
  (`ImportError: you should not try to import numpy from its source directory`).
  `slurm/train.sbatch` does `unset PYTHONPATH PYTHONHOME`.
- The sample pool generated `randperm` indices on the CPU while the pool lived on
  CUDA; fixed in `nca/pool.py`.

## Code changes (done)

- **Periodic boundaries**: the CA now wraps (circular padding in perception and in
  the living mask), so cells no longer "bounce off" the edges. Controlled by
  `NeuralCA(wrap=True)`; `--no-wrap` reproduces the original zero-padded
  behaviour. ONNX export uses a single `Pad(mode="wrap")` (opset 19) — see
  `AGENTS.md` — with parity ~1e-6.
- **Resume**: `--resume runs/<name>/state.pt` restores model + optimizer +
  scheduler + RNG + sample pool.
- **Logging**: `loss.json` plus flushed stdout; `state.pt` and `ckpt_*.pt` are
  written every `--ckpt-every` steps. **All** intermediate checkpoints are kept
  (they are tiny).
- **Perf**: `cudnn.benchmark` + TF32 enabled automatically on CUDA.
- **Run naming**: `--run-name` gives a deterministic `runs/<name>/`.
- Targets are pre-cached in `assets/targets/` (committed), so compute nodes need
  no internet.
- Still optional/untested: `torch.compile`, AMP. The bottleneck is the Python
  unroll, so try `--compile` if iteration time is high.

## SLURM scripts

- `slurm/train.sbatch` — one job; sources `.venv`, `unset PYTHONPATH PYTHONHOME`,
  forwards its args to `main.py --device cuda`. Pins `--constraint=h100`.
- `slurm/submit_sweep.sh` — submits experiments 1–3 for each target, with
  `PARTITION` / `CONSTRAINT` / `GRES` / `TIME` / `MEM` / `CPUS` overrides:

  ```bash
  bash slurm/submit_sweep.sh                       # gpu partition (H100)
  PARTITION=mig-priority CONSTRAINT=mig GRES=gpu:1g.10gb:1 TIME=02:00:00 \
    MEM=16G CPUS=4 bash slurm/submit_sweep.sh      # all MIG slices
  ```

  Targets use **codepoint** specs (`emoji:1f980`) so ssh/locale can't mangle emoji.

New emoji set (distinct from the paper's):

| name | emoji | name | emoji |
| --- | --- | --- | --- |
| crab | 🦀 | mushroom | 🍄 |
| octopus | 🐙 | cactus | 🌵 |
| trex | 🦖 | turtle | 🐢 |

```bash
bash slurm/submit_sweep.sh              # all 6 targets x 3 experiments = 18 jobs
bash slurm/submit_sweep.sh crab         # just one target
```

## Suggested H100 config (paper settings)

| knob | value | why |
| --- | --- | --- |
| target-size / padding | 40 / 16 (72×72 grid) | paper setting |
| batch-size | 16 | H100 is underutilised at batch 8 |
| iterations | 8000 | paper |
| train-steps | 64–96 | paper |
| pool-size / damage | 1024 / 0 or 3 | persistent / regenerating |
| lr | 2e-3, milestone 2000 | paper |
| checkpoint-every | 250 | cheap, safe |

Measured: **~54 min per model on a MIG slice** (~2.5 it/s), faster on a full H100.
An 18-model sweep is ~2 h on the 14 MIG slices. Each run's `state.pt` is ~340 MB,
so 18 runs ≈ 6 GB (fine on `/d`).

## Transfer (git)

Remote: `https://github.com/smufa/nca.git` (branch `master`).

```bash
# laptop
git push -u origin master

# cluster
git clone https://github.com/smufa/nca.git ~/nca
cd ~/nca && uv sync --no-sources --python 3.13 --group dev
```

## Workflow

1. Push the repo, clone on the cluster, `uv sync`.
2. **Smoke test**: `srun -p gpu --constraint=h100 --gres=gpu:1 --time=00:10:00 \
   --pty bash` then a 100-iteration run to confirm H100 + cu13 torch.
3. `bash slurm/submit_sweep.sh`.
4. Monitor: `squeue -u $USER`, `tail -f logs/*.out`, `sacct -j <id>`.
5. Export ONNX. Either on the login node (`uv run python -m nca.export ...`, same
   `unset PYTHONPATH` caveat) or pull the small `ckpt_final.pt` + `target.npy`
   back with rsync and export locally (the exporter needs the `dev` group).
6. Commit `exports/`, open the frontend.

## Milestones

1. **T1** — env on cluster + H100 smoke test (imports, 100 iters). ✅ done
   (validated on an H100 MIG slice: `torch 2.14.1+cu130`, CUDA ok, loss drops)
2. **T2** — one high-quality run (crab, regenerating) + ONNX export. ✅ done
3. **T3** — sweep experiments 1–3 across the 6 targets. 🚧 14/18 exported;
   `mushroom_{growing,persistent,regenerating}` + `turtle_regenerating` remain.
4. **T4** — compare/curate models, ship to the frontend. 🚧 20 models in `exports/`.

## Open questions

- Keep all checkpoints (current behaviour) or prune old ones after each run?
- Export every checkpoint, or only `ckpt_final` + best-by-loss?
