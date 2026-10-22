# Neural Cellular Automata

A from-scratch PyTorch reimplementation of **Growing Neural Cellular Automata**
(Mordvintsev, Randazzo, Niklasson & Levin, Distill 2020,
<https://distill.pub/2020/growing-ca/>), plus a browser playground and a cluster
training pipeline.

> **Agents: start with [`AGENTS.md`](AGENTS.md)** — repo map, invariants, commands
> and gotchas.

Starting from a single seed cell, a tiny neural network (8,336 parameters)
learns a local update rule that grows a target image, and — with the right
training regime — persists it and regenerates it after damage.

## Setup

```bash
uv sync
```

PyTorch is pinned to the **CPU** wheel index in `pyproject.toml` because this
project is being developed on a machine without an NVIDIA GPU. To use CUDA,
remove the `[tool.uv.sources]` / `[[tool.uv.index]]` blocks and re-run
`uv sync`.

## Quick start

```bash
# Train a "regenerating" model to grow a lizard emoji (CPU-friendly settings).
uv run python main.py --target "emoji:🦎"

# Faster smoke test on a simple shape.
uv run python main.py --target shape:heart --target-size 20 --padding 4 \
    --iterations 400 --batch-size 4 --train-steps 32 48
```

Outputs land in `runs/<target>_<experiment>_<timestamp>/`:

| file | contents |
| --- | --- |
| `target.png`, `seed.png` | the (padded) target and initial seed |
| `progress/batch_*.png` | batch states before/after each logged step |
| `ckpt_*.pt`, `ckpt_final.pt` | model weights |
| `loss.png` | training loss curve |
| `growth.gif` | the final model growing from a seed |

Run a trained model, optionally with damage to show regeneration:

```bash
# grow for 300 steps
uv run python -m nca.sample --checkpoint runs/.../ckpt_final.pt --steps 300

# grow, cut a hole at step 150, watch it heal
uv run python -m nca.sample --checkpoint runs/.../ckpt_final.pt \
    --size 64 --steps 400 --damage-at 150 --damage-frac 0.25
```

Targets can be an emoji (`emoji:🦎`, `emoji:1f98e`), a local image
(`image:path/to.png`) or a built-in shape (`shape:heart`, `shape:star`,
`shape:circle`, `shape:ring`, `shape:plus`). Emojis are downloaded from the
Twemoji CDN and cached under `assets/targets/`.

## Export to ONNX

Trained models can be exported to ONNX for browser/WebGPU inference. The exported
graph is a single deterministic CA step that takes the perception kernel and the
fire mask as inputs, so a frontend can drive the loop and manipulate the kernel
(e.g. rotate the perception field) without re-exporting.

```bash
uv run python -m nca.export --checkpoint runs/.../ckpt_final.pt \
    --id lizard --experiment regenerating --target "emoji:🦎" --verify
```

Outputs land in `exports/` (`model.onnx`, `meta.json`, `target.png`) and a
`manifest.json` index. `--verify` runs the PyTorch and ONNX rollouts with
identical masks at two grid sizes and asserts they agree (typically ~1e-7).

See [`PLAN.md`](PLAN.md) for the ONNX interface, the WebGPU frontend design and
performance notes, and [`TRAINING.md`](TRAINING.md) for training on the ARNES
SLURM cluster (H100, mixed-precision env, sweeps).

## Web playground

A Vite + TypeScript app runs the exported models live in the browser. See
[`AGENTS.md`](AGENTS.md) for the architecture, commands and gotchas.

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
```

It reads `exports/manifest.json` and offers three inference engines:

| engine | runtime | notes |
| --- | --- | --- |
| `WASM (CPU)` | `onnxruntime-web` WASM | default; fastest on weak GPUs |
| `WebGPU (ORT)` | `onnxruntime-web` WebGPU EP | ~1–12 ms per graph node |
| `WebGPU (shader)` | native WGSL | 2 compute passes, zero-copy render |

Controls: model / engine / grid pickers, play-pause (<kbd>space</kbd>), step
(<kbd>s</kbd>), reset (<kbd>r</kbd>), speed, canvas size, add channel view
(<kbd>v</kbd>).

- **Channel views** — any number of views, each mapping any three state channels
  to R/G/B (plus optional alpha and background); a `1ch` tick shows one channel as
  grayscale.
- **Brush** — View / Set 1 / Set 0 / × (multiply) with a radius and factor; edits
  the channels the window shows (tick `edit α` to also write channel 3).
- **Kernel lab** — angle slider (Experiment 4: rotate the perception field with no
  retraining), auto-rotate, three editable 3×3 kernels, and reset.
- **Grid** — 48…256; the model has dynamic `H`/`W` and periodic boundaries, so it
  runs on a larger torus without re-exporting.

URL params: `?ep=wasm|webgpu|shader`, `?log=1` (ORT logs).

Models are served from `../exports` at `/models` by a small Vite plugin (Vite
ignores a `publicDir` outside the project root), so run
`uv run python -m nca.export ...` first.

## The model

Each cell has a **16-dimensional state vector** `[B, 16, H, W]`:

* channels `0:3` — RGB colour,
* channel `3` — alpha (how "alive" the cell is),
* channels `4:16` — hidden signalling channels with no predefined meaning.

One update step (`nca/model.py`) has four phases:

1. **Perception.** Each cell concatenates its own state with the `x`/`y`
   spatial gradients of every channel, estimated with fixed Sobel filters
   (`depthwise_conv2d`). This produces a 48-dimensional perception vector
   (`16 state + 16 d/dx + 16 d/dy`).
2. **Update rule.** A small MLP (1×1 conv `48→128`, ReLU, 1×1 conv `128→16`)
   turns the perception vector into an *increment* `dx`. The final layer is
   zero-initialised so the CA initially "does nothing".
3. **Stochastic / asynchronous update.** A random per-cell mask zeroes `dx`
   with probability `1 - fire_rate` (default `0.5`), modelling cells that
   update at independent random times instead of on a global clock.
4. **Living-cell masking.** Cells are zeroed unless a cell with `alpha > 0.1`
   exists in their 3×3 neighbourhood. Empty cells therefore carry no state
   and cannot grow on their own.

Perception and the living mask use **circular (periodic) padding**, so the grid
is a torus: patterns wrap around the edges instead of the CA learning to
"bounce off" walls. Pass `--no-wrap` for the paper's zero-padded behaviour.

Total parameters: **8,336** (`48*128 + 128 + 128*16 + 16`), matching the
reference implementation.

## Training

The loss is the mean-squared error between the first four channels (RGBA) of
the state grid and the padded target. Gradients are backpropagated through a
random number of unrolled CA steps (default 48–64). Each variable's gradient
is L2-normalised (`g / (||g|| + 1e-8)`) as in the paper's footnote, which
stabilises long unrolls.

Three regimes are available via `--experiment`:

| experiment | sample pool | damage | noise | idea |
| --- | --- | --- | --- | --- |
| `growing` | off | 0 | 0 | grow the target from the seed (Experiment 1) |
| `persistent` | on | 0 | 0 | learn an attractor so the pattern persists (Experiment 2) |
| `regenerating` | on | 3 | 0 | also damage pool states so the model heals (Experiment 3) |
| `noisy` | on | 3 | 4×0.2 | also corrupt states with Gaussian noise, so the model reconstructs |

`noisy` is `regenerating` plus additive Gaussian state noise (`--noise-n` samples
per batch, `--noise-std` the stddev; also settable directly). Circle damage only
ever zeroes contiguous blobs, so the noise is what forces the model to pull an
*arbitrary* corrupted state back to the target. The first slot of every batch
stays the clean seed.

With the sample pool, each training step samples a batch of previously seen
states, replaces the highest-loss sample with the fresh seed (so the model
never forgets how to grow), damages the lowest-loss samples with random
circles, trains, and writes the outputs back into the pool. This teaches the
model to *stay* at the target rather than just reach it once.

## Notes on hardware

The default settings (`--target-size 32 --padding 8 --batch-size 4
--iterations 1200 --train-steps 48 64`) are tuned so a run finishes in roughly
15–25 minutes on a 4-core CPU. The paper's settings
(`--target-size 40 --padding 16 --batch-size 8 --iterations 8000
--train-steps 64 96`) produce much sharper patterns but take many hours here.
On a GPU, use the paper settings.

## Layout

```
main.py              CLI entry point
nca/model.py         NeuralCA module: perception, update rule, masking
nca/targets.py       target loading / preprocessing / seed
nca/pool.py          sample pool
nca/train.py         training loop, damage masks, gradient normalisation
nca/viz.py           rendering, tiling, gifs, loss plots
nca/cli.py           argument parsing and wiring
nca/sample.py        run a checkpoint, grow / regenerate
nca/export.py        export a checkpoint to ONNX + parity check
exports/             exported models + manifest.json
frontend/            Vite + TypeScript + onnxruntime-web playground
```

## Possible next steps

* **Experiment 4 — rotated perception.** `NeuralCA.perceive` and `forward`
  already accept an `angle` argument; rotating the Sobel kernels rotates the
  grown pattern without retraining.
* **Longer training / GPU** for crisp patterns.
* **Port to JAX / TF** for `jax.lax.scan`-based unrolling.
* **Quantisation + WebGL/TF.js export** for the interactive demo.

## Reference

Mordvintsev, A., Randazzo, E., Niklasson, E., & Levin, M. (2020).
*Growing Neural Cellular Automata.* Distill. https://doi.org/10.23915/distill.00023
