# AGENTS.md — orientation for coding agents

Read this first. It is the map of the repo, the non-obvious invariants, and the
commands you'll actually need. Deeper detail lives in `README.md` (training +
model), `PLAN.md` (ONNX/frontend design), `TRAINING.md` (cluster).

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
  over white (`bg=1`) or black (`bg=0`). `nca/nca.ts:mapChannels` is the
  reference; `frontend/src/wgsl/draw.wgsl` mirrors it on the GPU.
- **Boundaries are periodic (toroidal).** Implemented as a single ONNX
  `Pad(mode="wrap")` (opset 19) / `F.pad(mode="circular")`. `--no-wrap` restores
  the paper's zero padding but is not used by default.
- **The perception kernel is a model *input*, not a constant.** This is what
  makes the kernel lab (rotation, editable Sobel) possible without re-export.
- **Training and inference must agree.** Any change to `nca/model.py` changes the
  graph; always re-export and check `--verify` parity.

## Repo layout

```
nca/model.py        NeuralCA (perception, update rule, fire + living masks)
nca/train.py        training loop (sample pool, damage + noise, grad normalisation, resume)
nca/targets.py      target images: emoji / image file / procedural shapes
nca/export.py       export a checkpoint to ONNX + parity check + weights.bin
nca/cli.py          `python main.py` entry point (train)
nca/sample.py       load a checkpoint, grow / regenerate (standalone)
nca/viz.py          rendering, tiling, gifs, loss plots

frontend/src/
  engine.ts         CaEngine interface shared by the two web backends
  runner.ts         CaRunner: onnxruntime-web backend (wasm or webgpu EP)
  shader.ts         ShaderRunner: native WGSL backend (2 compute passes)
  wgsl/*.wgsl       the actual shaders (shared with the Python bench)
  nca.ts            pure helpers: perception kernels, seed, channel mapping
  main.ts           UI: views, brush, kernel lab, engine/grid switches
  models.ts         manifest + meta types + picker grouping (modelGroup)

scripts/shader_bench.py   headless GPU correctness/benchmark harness (wgpu-py)
scripts/variants/*.wgsl   shader experiments (see "Shader performance")
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
Output
  x_next            float32 [B, C, H, W]
```

Ops: `Pad(wrap)`, `Conv`, `MaxPool`, `Relu`, `Mul`, `Add`, `Greater`, `And`,
`Cast`, `Slice` — ~19 nodes. `nca/export.py` also writes `weights.bin` (the MLP
as flat f32: `W1[128,48]`, `b1[128]`, `W2[16,128]`, `b2[16]`) for the WGSL engine.

## Commands

```bash
uv sync --group dev                     # Python env (CPU torch)
uv run python main.py --target shape:heart --iterations 300   # train
uv run python -m nca.export --checkpoint runs/.../ckpt_final.pt \
    --id NAME --experiment regenerating --target "emoji:1f980" --verify
uv run python -m nca.sample --checkpoint runs/.../ckpt_final.pt --steps 300

cd frontend && npm install && npm run dev     # http://localhost:5173
npm run build                                 # tsc + vite build

bash scripts/verify.sh                        # compile + build + parity, one shot
bash scripts/bench.sh crab_regenerating --reference   # headless GPU correctness/perf
```

Frontend URL params: `?ep=wasm|webgpu|shader`, `?log=1` (ORT logs).
Engine choice persists in `localStorage` (`nca.ep`).

## Frontend engines (Engine dropdown)

| engine | runtime | notes |
|---|---|---|
| `wasm` | onnxruntime-web WASM | **default**; fastest on weak GPUs (~120 steps/s @72²) |
| `webgpu` | onnxruntime-web WebGPU EP | ~1–12 ms **per graph node** — slow on iGPUs |
| `shader` | native WGSL | 2 dispatches; zero-copy render + GPU brush |

`shader` is the performance path. It renders each view straight from the GPU
state buffer (`renderTo`) and applies brush strokes with a compute shader
(`brush`), so nothing is downloaded per frame. `wasm`/`webgpu` render through
Canvas2D and call `sync()` to refresh the CPU mirror `x`.

## Shader performance (important)

`scripts/shader_bench.py` runs the real shaders headlessly on the GPU via Vulkan
and diffs against onnxruntime — **use it instead of guessing**:

```bash
bash scripts/bench.sh crab_regenerating --steps 150 --reference
bash scripts/bench.sh crab_regenerating --grid 256 --shader scripts/variants/foo.wgsl
```

Findings so far (Intel UHD 620):
- The **MLP is ~98%** of the step; perception is ~0.06 ms.
- It is bound by **weight loads**, not FLOPs.
- `vec4` weight loads + 4-hidden-at-a-time: **3.2–3.6×** (see `pass1.wgsl`).
  Current: 72² 1.54 ms · 128² 4.40 ms · 256² 16.8 ms.
- Tried and **rejected** (slower): `perc` in workgroup shared memory
  (added LDS + cost occupancy; the private array was in registers already),
  explicit 4-way ILP (compiler already does it), a 16×16 workgroup, and 2-cell
  register tiling (register spill → 5.2 ms and incorrect).
  The experiments are kept in `scripts/variants/`.
- Remaining ideas (not tried):
  1. **Tiled GEMM for the MLP** (tile cells into shared memory, stream the weights
     once per tile) — the proper fix at 512²+, but a real WGSL matmul project.
  2. **fp16** weights/activations — the iGPU exposes `shader-f16`; maybe ~2×, but
     real precision risk (the bench settles it immediately).
  3. **Split hidden units across workgroups** and reduce `dx` with atomics to cut
     per-thread weight reads.
- The kernel is **size-generic**: `C`, `NP4`, `W1S4`, `HG` are WGSL `override`
  constants set per pipeline. They must stay *compile-time* — dynamic loop bounds
  cost ~1.9×. Export pads each `W1` row to `3C → ceil(3C/4)*4` floats so `vec4`
  loads stay aligned; at `C=16` that is a no-op, so old `weights.bin` is unchanged.
  Supported: `4 ≤ C ≤ 16`, `hidden % 4 == 0` (`ShaderRunner.supports`), else the
  frontend falls back to WASM.
- The bench's `--reference` reports a **1-step** diff as the correctness gate; the
  N-step diff is larger because the CA is chaotic.

## Training performance (measured, 72×72, 80-step unroll)

The unroll is **throughput-bound, not latency-bound**: its time is *exactly*
linear in batch size, so a larger batch buys nothing on MIG (and only ~1.8× on an
under-utilised H100). Non-unroll code (pool sample/rank/damage, grad norm,
commit, `.item()`) is ~0.3 ms of a 54 ms (H100) / 429 ms (MIG) iteration.

| batch | MIG fp32 / bf16 | H100 fp32 / bf16 |
|---:|---|---|
| 8 | 235 / 150 ms | — |
| 16 | 429 / 331 ms (2.3 it/s) | 54 / 55 ms (18.6 it/s) |
| 24 | 642 / 534 ms | — |
| 32 | — | 85 / 74 ms |
| 64 | — | 156 / 126 ms |
| 128 | — | 302 / 244 ms |

Samples/s: MIG 34→37→37 (8→24); H100 298→432→510→525 (16→128). Peak GPU memory
at batch 16: **5.6 GB fp32 / 3.0 GB bf16**, linear in `batch_size x train_steps`
(autograd keeps every step's activations), so a `1g.10gb` slice caps out near
batch 24.

- **A full H100 is ~8× a MIG slice at the same batch and schedule** (8000 iters
  at batch 16: ~7 min vs ~54 min). Use an H100 for a single run; use the 14 MIG
  slices for a *sweep*, which parallelises better than one GPU serially.
- **`--amp bf16`**: ~1.3–1.55× on MIG and halves memory; neutral on H100 at batch
  16, ~1.2–1.3× at batch ≥ 32.
- **`torch.compile` does not help** (module: neutral; whole sequence: slower via
  warmup + per-`n` specialisation). CUDA graphs are untested and only matter if
  the step were latency-bound, which it is not.
- Training draws the whole unroll's fire mask up front (`torch.rand([n,B,1,H,W])`)
  instead of one `rand_like` per step, and keeps losses as device scalars until a
  log/checkpoint needs the number (no per-step `.item()` sync). Regenerate this
  table with `sbatch slurm/profile.sbatch`.

## Model capacity (crab / regenerating, 8k iters)

The paper size (C=16, hidden=128, **8336 params**) reaches min loss **2.0e-4**. The
sweep models are in `exports/crab_c*_h*`:

| hidden | C | params | min loss | ×base |
|---:|---:|---:|---:|---:|
| 128 | 16 | 8336 | 2.0e-4 | 1 × |
| 128 | 12 | 6284 | 2.4e-3 | 12 × |
| 128 | 8 | 4232 | 5.7e-3 | 28 × |
| 64 | 16 | 4176 | 8.8e-3 | 44 × |
| 128 | 6 | 3206 | 9.2e-3 | 46 × |
| 32 | 16 | 2096 | 1.3e-2 | 63 × |
| 16 | 16 | 1056 | 2.0e-2 | 99 × |
| 8 | 16 | 536 | 1.9e-2 | 96 × |

Channels are the cheaper axis to cut at equal params (C=12 beats hidden=64); below
~4k params quality falls off a cliff.

## Robustness / the kernel lab is a probe, not a toy

Perturbing the perception kernels in the playground's kernel lab gives a cell a
scrambled view of its neighbourhood *without* changing its state. A model trained
only on clean perception + circle damage has no return path from the states that
produces: it falls into a noisy attractor that is absorbing and never recovers.
That is an out-of-distribution perturbation the training regime never covered.

Training with state noise (`--experiment noisy`, or `--noise-n` / `--noise-std`)
fixes it: the update rule is then asked to map a whole neighbourhood of corrupted
states back to the target, which widens the target's basin. It costs nothing in
fidelity — `crab_noisy` reaches min loss **2.45e-4** vs **2.58e-4** for the plain
fp32 `regenerating` baseline, with 4 of every 16 pool samples corrupted.

Takeaway: robustness tracks the *breadth of perturbation seen during training*,
not the amount of damage. Circle damage only ever zeroes contiguous blobs, so it
does not cover the space the kernel lab can reach. Neither is captured by the
training loss, which is not comparable across regimes — the failure was found by
playing with the kernel lab, not by a metric.

## Gotchas

- **NixOS + Vulkan (for `scripts/shader_bench.py`)**: the loader/ICDs aren't on
  the default path. `scripts/bench.sh` sets them up. Without it wgpu reports
  "No suitable graphics adapter found".
- **Cluster (ARNES)**: `gpu` partition is **mixed H100 + V100S**; pin the H100s
  with `--constraint=h100` (CUDA 13 dropped Volta). Lmod also exports a Python
  3.12 `PYTHONPATH` that shadows the venv — `slurm/train.sbatch` unsets it.
  Clone over HTTPS (the cluster has no GitHub SSH key).
- **Dynamo ONNX export** writes weights as *external data* by default — the
  exporter passes `external_data=False` so the `.onnx` is self-contained.
- **`torch.onnx.export(dynamic_shapes=...)`** takes **positional / forward arg
  names**, not ONNX input names.
- **`ort.env.webgpu.device` is a promise.**
- The WebGPU EP provider options (`validationMode`, buffer cache modes) were
  tried and **removed** — they abort with `InternalError: out of memory` in the
  current ORT-Web build. Don't reintroduce them.
- `frontend`'s Vite `publicDir` is `../exports`, which Vite ignores for a dir
  outside the root; a small plugin in `vite.config.ts` serves `/models` instead.
- The brush edits via `CaEngine.brush` (shader) or `edit` (ORT), which queues
  edits while a step is in flight; a direct write during `session.run` is lost.

## Workflows

**Add a model**: train → `nca.export` (writes `exports/<id>/{model.onnx,meta.json,weights.bin,target.png}` and updates `exports/manifest.json`) → commit `exports/`.

**Change the shader**: edit `frontend/src/wgsl/pass1.wgsl` → `bash scripts/bench.sh <model> --reference` for correctness + perf → `npm run build`.

**Change the model**: edit `nca/model.py` → re-export **and re-verify** (parity should be ~1e-6) → re-export all models if the graph changed.

## Status

- Training: working; `regenerating` regime produces crisp, self-healing patterns.
- Frontend: `shader` engine is the fast path; `wasm` is the compatibility default.
- Cluster: **all 18 sweep models exported** (24 models in `exports/`).
- Next likely work: larger-grid/texture targets, per-channel stats / timeline UI,
  and the remaining shader ideas listed above.
