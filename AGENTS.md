# AGENTS.md — orientation for coding agents

Read this first. It is the map of the repo, the non-obvious invariants, and the
commands you'll actually need. Measurements and the reasoning behind them live in
[`docs/findings/`](docs/findings/README.md) — these numbers are not repeated here.
Other docs: `README.md` (usage + model), `PLAN.md` (ONNX/frontend design),
`TRAINING.md` (cluster), `docs/HOUSEKEEPING.md` (refactor plan + known drift).

## What this is

A from-scratch PyTorch reimplementation of **Growing Neural Cellular Automata**
(Mordvintsev et al., Distill 2020) with three parts:

1. **`nca/`** — the 16-channel CA, training (sample pool + damage), ONNX export.
2. **`frontend/`** — Vite + TypeScript playground that runs a trained CA live.
3. **`slurm/`** — training on the ARNES HPC cluster (H100s).

The whole point of the architecture is that **one trained model = a single
`step(x) -> x'` function**, everything else is plumbing around it.

## Core invariants (do not break these)

- **State layout is channels-first `[B, C, H, W]`** everywhere (PyTorch, ONNX,
  WGSL, JS). `C = 16`: channels `0:3` = RGB, `3` = alpha, `4:16` = hidden.
- **RGB is premultiplied by alpha**; rendering composites `out = rgb + (1-a)*bg`
  over white (`bg=1`) or black (`bg=0`). `nca.ts:mapChannels` is the reference;
  `frontend/src/wgsl/draw.wgsl` mirrors it on the GPU.
- **Boundaries are periodic (toroidal)** — one ONNX `Pad(mode="wrap")` (opset 19) /
  `F.pad(mode="circular")`. `--no-wrap` restores the paper's zero padding.
- **The perception kernel is a model *input*, not a constant.** That is what makes
  the kernel lab (rotation, editable Sobel) possible without re-export.
- **Training and inference must agree.** Any change to `nca/model.py` changes the
  graph; re-export and check the parity line (`--no-verify` skips it).
- **Read a model's shape from `nca/spec.py`, never re-derive it.** `ModelSpec` is
  the one source of truth for `C` / hidden / hidden_layers / `cond_n` / boundaries: the
  module, `export.py`, the `weights.bin` offsets and the frontend's `meta.json` all
  agree with it, and `tests/test_spec.py` + `scripts/check_meta.py` prove it for
  every committed export (`tests/test_export.py` repacks each committed
  `model.onnx` into `weights.bin` and diffs it, so the blob layout is proven
  without the `runs/` checkout; the checkpoint-based checks skip in a clean clone).
  `update[0]`'s fan-in is `3C + cond_n`; deriving `C` from
  it is wrong (that bug lived in `nca/sample.py`).

## Repo layout

```
nca/spec.py         ModelSpec: the shape of a model (channels, hidden, hidden_layers, cond_n)
nca/model.py        NeuralCA (perception, update rule, fire + living masks)
nca/train.py        training loop (sample pool, damage + noise, grad normalisation, resume)
nca/targets.py      target images: emoji / image file / procedural shapes
nca/export.py       export a checkpoint to ONNX + parity check + weights.bin
nca/cli.py          `python main.py` entry point (train)
nca/sample.py       load a checkpoint, grow / regenerate (standalone)
nca/viz.py          rendering, tiling, gifs, loss plots

frontend/src/
  engine.ts         CaEngine interface shared by the two web backends
  runner.ts         CaRunner: onnxruntime-web WASM backend (CPU compatibility path)
  shader.ts         ShaderRunner: native WGSL backend (2 compute passes)
  wgsl/*.wgsl       the actual shaders (shared with the Python bench)
  nca.ts            pure helpers: perception kernels, seed, channel mapping
  main.ts           UI: views, brush, kernel lab, engine/grid switches
  models.ts         manifest + meta types + picker grouping (modelGroup)

scripts/shader_bench.py   headless GPU correctness/benchmark harness (wgpu-py)
scripts/check_meta.py     validate exports/*/meta.json against ModelSpec (--fix repairs)
scripts/variants/*.wgsl   shader experiments (see docs/findings/shader-perf.md)
tests/                    pytest: spec/weights layout, exporter, meta schema, training smoke
slurm/                    cluster submission
exports/                  exported models, committed (small); served to the UI
assets/targets/           cached emoji PNGs (offline training)
```

## ONNX graph interface

```
Inputs
  x                 float32 [B, C, H, W]
  perception_weight float32 [3C, 1, 3, 3]   # identity, dx, dy per channel
  fire_mask         float32 [B, 1, H, W]    # stochastic update (0/1)
  step_size         float32 []
  cond              float32 [B, cond_n, 1, 1]  # only for --targets models
Output
  x_next            float32 [B, C, H, W]
```

Ops: `Pad(wrap)`, `Conv`, `MaxPool`, `Relu`, `Mul`, `Add`, `Greater`, `And`,
`Cast`, `Slice` — ~19 nodes. `nca/export.py` also writes `weights.bin` (the MLP as
flat f32: `W1[hidden,P4]`, `b1`, optional hidden→hidden blocks, `W_out[C,hidden]`,
`b_out[C]`) for the WGSL engine. Exact offsets: `ModelSpec.weight_offsets()`; at the
paper size `W1[128,48]`, `b1[128]`, `W2[16,128]`, `b2[16]`.

## Commands

```bash
uv sync --group dev                     # Python env (CPU torch)
uv run python main.py --target shape:heart --iterations 300        # train
uv run python main.py --run-name demo --target emoji:1f980 --iterations 4000
uv run python -m nca.export --checkpoint runs/.../ckpt_final.pt \
    --id NAME --experiment regenerating --target "emoji:1f980"   # verifies parity
uv run python -m nca.export --all            # re-export every model that has its checkpoint
uv run python -m nca.export --reindex        # rebuild manifest.json from exports/*/meta.json
uv run python -m nca.sample --run runs/NAME --steps 300         # shape from spec.json, else the checkpoint
uv run python -m nca.sample --run runs/cross8 --cond 3          # conditional model, target 3
uv run python -m pytest -q tests              # 329 tests, ~5 s on CPU
uv run python -m scripts.check_meta           # exports/*/meta.json vs nca/spec.py

cd frontend && npm install && npm run dev     # http://localhost:5173
npm run build                                 # tsc + vite build

bash scripts/verify.sh                        # compile + tests + meta + build + parity
bash scripts/bench.sh crab.regenerating --reference   # headless GPU correctness/perf
```

## Frontend engines

| engine | runtime | notes |
|---|---|---|
| `wasm` | onnxruntime-web WASM | compatibility fallback; runs anything the shader cannot |
| `shader` | native WGSL | performance path; 2 dispatches, zero-copy render + GPU brush |

The ORT **WebGPU EP was removed** (per-node dispatch lost to both engines) — do not
reintroduce it. URL params: `?ep=wasm|shader`, `?log=1`; the choice persists in
`localStorage["nca.ep"]`. Details: [`docs/findings/frontend-engines.md`](docs/findings/frontend-engines.md).

## Gotchas

- **NixOS + Vulkan** (for `scripts/shader_bench.py`): loader/ICDs are not on the
  default path; `scripts/bench.sh` sets them up. Without it wgpu reports "No
  suitable graphics adapter found".
- **Cluster (ARNES)**: `gpu` is **mixed H100 + V100S** — pin with `--constraint=h100`
  (CUDA 13 dropped Volta). Lmod exports a Python 3.12 `PYTHONPATH` that shadows the
  venv; `slurm/train.sbatch` unsets it. Clone over HTTPS (no GitHub SSH key).
- **Dynamo ONNX export** writes weights as *external data* by default; the exporter
  passes `external_data=False` so the `.onnx` is self-contained.
- **`torch.onnx.export(dynamic_shapes=...)`** takes **positional / forward arg
  names**, not ONNX input names.
- `frontend`'s Vite `publicDir` is `../exports`, which Vite ignores for a dir
  outside the root; a small plugin in `vite.config.ts` serves `/models` instead.
- A run directory is self-describing (`spec.json`, `train.json`, `target.npy`,
  `loss.json`, `state.pt`). Read it with `ModelSpec.from_run_dir`; do not pass
  shapes on the command line unless you are overriding on purpose.

## Workflows

**Add a model**: train → `nca.export` (writes `exports/<id>/{model.onnx,meta.json,
weights.bin,target.png}` and updates `manifest.json`) → `scripts/check_meta.py` →
commit `exports/`.

**Change the shader**: edit `frontend/src/wgsl/pass1.wgsl` →
`bash scripts/bench.sh <model> --reference` for correctness + perf → `npm run build`.

**Change the model**: edit `nca/model.py` → re-export **and re-verify** (parity
~1e-6) → `nca.export --all` → `npm run build`.

## Status

- Training: working; `regenerating` regime produces crisp, self-healing patterns.
- Frontend: `shader` is the fast path, `wasm` is the compatibility default.
- Cluster: all 18 sweep models exported (45 models in `exports/`).
- Next likely work: larger-grid/texture targets, per-channel stats / timeline UI,
  and the remaining shader ideas in `docs/findings/shader-perf.md`.
