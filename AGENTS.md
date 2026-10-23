# AGENTS.md — orientation for coding agents

Read this first: the map, the invariants, the commands. Measurements and their
reasoning live in [`docs/findings/`](docs/findings/README.md) — not repeated here.
Other docs: `README.md` (usage), `PLAN.md` (ONNX/frontend design), `TRAINING.md`
(cluster), `docs/HOUSEKEEPING.md` (refactor plan + known drift).

## What this is

A from-scratch PyTorch reimplementation of **Growing Neural Cellular Automata**
(Mordvintsev et al., Distill 2020): `nca/` (the CA, training, ONNX export),
`frontend/` (Vite + TS playground running a trained CA live) and `slurm/`
(training on the ARNES H100s). The architecture is one idea: **one trained model =
a single `step(x) -> x'` function**, everything else is plumbing around it.

## Core invariants (do not break these)

- **State layout is channels-first `[B, C, H, W]`** everywhere (PyTorch, ONNX,
  WGSL, JS). `C = 16`: channels `0:3` = RGB, `3` = alpha, `4:16` = hidden.
- **RGB is premultiplied by alpha**: rendering composites `out = rgb + (1-a)*bg`.
  `nca.ts:mapChannels` is the reference, `frontend/src/wgsl/draw.wgsl` mirrors it.
- **Boundaries are periodic (toroidal)**: one ONNX `Pad(mode="wrap")` (opset 19) /
  `F.pad(mode="circular")`; `--no-wrap` restores the paper's zero padding.
- **The perception kernel is a model *input*, not a constant** — that is what makes
  the kernel lab (rotation, editable Sobel) possible without re-export.
- **Training and inference must agree**: any change to `nca/model.py` changes the
  graph, so re-export and read the parity line (`--no-verify` skips it).
- **Read a model's shape from `nca/spec.py`, never re-derive it.** `ModelSpec` is
  the one source of truth for `C` / hidden / hidden_layers / `cond_n` / boundaries,
  and `tests/test_spec.py` + `scripts/check_meta.py` prove every committed export
  agrees with it (`tests/test_export.py` repacks each `model.onnx` into
  `weights.bin` and diffs it). `update[0]`'s fan-in is `3C + cond_n`; deriving `C`
  from it is wrong (that bug lived in `nca/sample.py`).
- **Read the step's *constants* from `nca/step.py`, never re-type them.** Perception
  basis, depthwise weight, seed cell and channel->pixel mapping are shared by torch,
  numpy (`scripts/shader_bench.py`), WGSL and `frontend/src/nca.ts`; `nca/step.py`
  may not import torch, and `tests/test_step.py` fails when `nca.ts` drifts.

## Repo layout

```
nca/spec.py         ModelSpec: the shape of a model (channels, hidden, hidden_layers, cond_n)
nca/step.py         torch-free primitives: perception kernels/weight, seed, channel->pixel
nca/model.py        NeuralCA (perception, update rule, fire + living masks)
nca/experiments.py  Experiment + REGIMES: every training default, in one place
nca/sweep.py        `python -m nca.sweep <toml>`: render one sbatch call per run
nca/train.py        training loop (sample pool, damage + noise, grad normalisation, resume)
nca/targets.py      target images: emoji / image file / procedural shapes
nca/export.py       export a checkpoint to ONNX + parity check + weights.bin
nca/cli.py          `python main.py` entry point (train)
nca/sample.py       load a checkpoint, grow / regenerate (standalone)
nca/viz.py          rendering, tiling, gifs, loss plots

frontend/src/
  engine.ts         CaEngine + Capabilities: the interface both backends implement
  runner.ts         CaRunner (onnxruntime-web WASM) · shader.ts ShaderRunner (WGSL)
  wgsl/*.wgsl       the shaders, shared with the Python bench
  nca.ts            pure helpers: perception kernels, seed, channel mapping
  main.ts           UI: views, brush, kernel lab, engine/grid switches
  models.ts         manifest + meta types + picker grouping

scripts/shader_bench.py   headless GPU correctness/benchmark harness (wgpu-py)
scripts/check_meta.py     validate exports/*/meta.json against ModelSpec (--fix repairs)
scripts/variants/*.wgsl   shader experiments (see docs/findings/shader-perf.md)
tests/                    pytest: spec/weights, exporter, meta, experiments, training smoke
slurm/train.sbatch        one job (sources .venv, unsets PYTHONPATH, pins --constraint=h100)
slurm/submit_sweep.sh     render a study's jobs: `bash … slurm/sweeps/<study>.toml`
slurm/sweeps/*.toml       the studies: emoji-sweep, capacity, multi, cross, experiments, demo
exports/                  exported models, committed (small); served to the UI
assets/targets/           cached emoji PNGs (offline training)
```

## ONNX graph interface

`x[B,C,H,W]`, `perception_weight[3C,1,3,3]`, `fire_mask[B,1,H,W]`, `step_size[]`,
optional `cond[B,cond_n,1,1]` → `x_next[B,C,H,W]`; ~19 nodes (`Pad(wrap)`, `Conv`,
`MaxPool`, `Relu`, `Mul`, `Add`, `Greater`, `And`, `Cast`, `Slice`). `export.py` also
writes `weights.bin` (offsets: `ModelSpec.weight_offsets()`). Design: `PLAN.md`.

## Commands

```bash
uv sync --group dev                     # Python env (CPU torch)
uv run python main.py --target shape:heart --iterations 300        # train
uv run python main.py --config slurm/sweeps/demo.toml --device cpu # a study's run, locally
uv run python main.py --config slurm/sweeps/capacity.toml \
    --run crab.regenerating.c=8.h=128 --set lr=1e-3                # file < flags < --set
uv run python -m nca.export --checkpoint runs/.../ckpt_final.pt \
    --id NAME --target "emoji:1f980"                             # verifies parity
uv run python -m nca.export --all            # re-export every model that has its checkpoint
uv run python -m nca.export --reindex        # rebuild manifest.json from exports/*/meta.json
uv run python -m nca.export --reindex --refresh-targets  # also re-render target.png/npy
uv run python -m nca.sample --run runs/NAME --steps 300         # shape from spec.json
uv run python -m nca.sample --run runs/cross8.mismatch --cond 3 # conditional model, target 3
uv run python -m nca.sweep slurm/sweeps/cross.toml --dry-run    # what would sbatch
uv run python -m pytest -q tests              # the suite (CPU, seconds)
uv run python -m scripts.check_meta           # exports/*/meta.json vs nca/spec.py

cd frontend && npm install && npm run dev     # http://localhost:5173
npm run build                                 # tsc + vite build

bash scripts/verify.sh                        # compile + tests + meta + build + parity
bash scripts/bench.sh crab.regenerating --reference   # headless GPU correctness/perf

bash slurm/submit_sweep.sh slurm/sweeps/emoji-sweep.toml   # cluster: 18 jobs
```

## Frontend engines

| engine | runtime | notes |
|---|---|---|
| `wasm` | onnxruntime-web WASM | compatibility fallback |
| `shader` | native WGSL | performance path; 2 dispatches, zero-copy render + GPU brush |

Both advertise a `Capabilities` object (`frontend/src/engine.ts`) and `checkSpec()`
is the only place a spec is tested against one — no `?.` feature probing. The ORT
**WebGPU EP was removed** (per-node dispatch lost to both); do not reintroduce it.
Details: [`docs/findings/frontend-engines.md`](docs/findings/frontend-engines.md).

## Gotchas

- **NixOS + Vulkan**: `scripts/bench.sh` sets up the loader/ICDs; skill: `headless-webgpu`.
- **Cluster (ARNES)**: `gpu` is **mixed H100 + V100S** — pin `--constraint=h100`
  (CUDA 13 dropped Volta). Lmod exports a 3.12 `PYTHONPATH` that shadows the venv
  (`train.sbatch` unsets it). Clone over HTTPS (no GitHub SSH key).
- **Dynamo ONNX export** writes weights as *external data* by default (the exporter
  passes `external_data=False`), and **`dynamic_shapes=`** takes *positional/forward
  arg* names, not ONNX input names.
- `frontend`'s Vite `publicDir` is `../exports`, which Vite ignores for a dir
  outside the root; a small plugin in `vite.config.ts` serves `/models` instead.
- A run directory is self-describing (`spec.json`, `train.json`, `target.npy`,
  `loss.json`, `state.pt`): read it with `ModelSpec.from_run_dir`.

## Workflows

**Add a model**: train → `nca.export` (writes `exports/<id>/{model.onnx,meta.json,
weights.bin,target.png}`, updates `manifest.json`) → `scripts/check_meta.py` → commit.

**Change the shader**: edit `frontend/src/wgsl/pass1.wgsl` →
`bash scripts/bench.sh <model> --reference` (correctness + perf) → `npm run build`.

**Change the model**: edit `nca/model.py` → `nca.export --all` **and re-verify**
(parity ~1e-6) → `npm run build`.

## Status

- Training works; `regenerating` produces crisp, self-healing patterns. Frontend:
  `shader` is the fast path, `wasm` the compatibility default.
- 45 exported models (18 from the cluster sweeps). Next likely work: texture
  targets, eval-per-checkpoint, and the shader ideas in `docs/findings/shader-perf.md`.
