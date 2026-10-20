# Plan: NCA → ONNX → WebGPU analysis playground

Reimplementation of *Growing Neural Cellular Automata* (Mordvintsev et al., Distill 2020).
This document covers the **inference/export pipeline and web frontend**. Training stays in
Python and is expected to move to a GPU cluster.

## Goals

- Export trained NCA checkpoints to ONNX as a **single CA step**.
- Run them in the browser on **WebGPU** with `onnxruntime-web`.
- Build a **general playground** for **deep analysis of a single model**
  (not a guided narrative, not multi-model comparison).

## Decisions (agreed)

| # | Decision |
|---|---|
| 1 | **Dynamic perception kernels** passed from JS (identity / sobel_x / sobel_y), so the UI can manipulate them. Static/baked kernels deferred. |
| 2 | Perception + update live in **one** ONNX model. |
| 3 | **WebGPU only** for now; other execution providers are a later fallback. |
| 4 | General **playground**, single-model deep analysis. |
| 5 | Training is separate (GPU cluster); the exporter is **standalone** and takes a `.pt` path. |
| 6 | **Commit** exported `.onnx` models (they are small, ~10–50 KB). |
| 7 | Dynamic `H`/`W` shapes (verified safe on the WebGPU/WASM path, see below). |

## ONNX graph interface

One step of the CA, fully deterministic (randomness lives in JS):

```
Inputs
  x                   float32 [B, C, H, W]     state grid
  perception_weight   float32 [3C, 1, 3, 3]    JS expands the 3 base kernels per channel
  fire_mask           float32 [B, 1, H, W]     per-cell update mask (0/1)
  step_size           float32 []               optional
Output
  x_next              float32 [B, C, H, W]
```

Graph = depthwise Sobel conv (`groups=C`) → 1×1 conv `3C→hidden` → ReLU →
1×1 conv `hidden→C` → `x + dx*fire_mask*step_size` → living-cell masking
(`MaxPool` → `Greater` → `And` → `Mul`).

The UI edits three 3×3 matrices; JS expands them with `w[3*ci+cm] = kernel[cm]` so the
conv output stays `[state, grad_x, grad_y]` per input channel, preserving the trained
update rule's semantics.

### Dynamic shapes — findings

- The current WebGPU/WASM path does **not** validate input dims against the graph:
  `InferenceSession.run` (`js/common/lib/inference-session-impl.ts`) only checks feed
  presence/names; `session-handler-inference.ts` and `prepareInputOutputTensor`
  (`wasm-core-impl.ts`) build tensors from the supplied dims directly.
- The well-known "expected shape `[,,,]`" error originates in the **legacy onnxjs/WebGL
  backend** (`js/web/lib/onnxjs/session.ts`, `validateInputTensorDims`), which does
  static dim validation. That backend is not our target.
- Conclusion: dynamic `H`/`W` are safe for WebGPU. `--verify` runs at a second shape to
  prove it.

## Performance notes: static vs dynamic kernels

| Aspect | Static kernels (initializers) | Dynamic kernels (graph inputs) — *current choice* |
|---|---|---|
| Conv FLOPs | identical | identical |
| Weight layout | **prepacked once** at session init (HWIO) | runtime transpose every call |
| Weight upload | none | ~1.7 KB/step (negligible) |
| **Graph capture** | ✅ possible (needs static shapes + all-GPU) | ❌ not possible |
| UI control | requires re-export / session rebuild | instant |

**Takeaway:** dynamic kernels cost almost nothing per step, but forfeit graph capture.
If profiling shows per-step `session.run` overhead dominating, revisit by baking kernels
as initializers and rebuilding the ~8 KB session on change.

Additional WebGPU notes:
- WebGPU EP **prefers NHWC**; ORT inserts layout transforms for our NCHW graph. An
  NHWC-exported variant may be faster — defer.
- JSEP is deprecated in favor of the native WebGPU EP; pin an `onnxruntime-web` version
  and test both builds (`.webgpu.mjs` vs `.bundle`).

## Export pipeline

`nca/export.py`:

```bash
uv run python -m nca.export --checkpoint ckpt.pt --id lizard --grid-size 64 \
    --experiment regenerating --target "emoji:🦎" --verify --out exports/
```

Emits:

```
exports/
  manifest.json
  <id>/
    model.onnx
    meta.json        # channels, hidden, fire_rate, grid, dynamic, opset, experiment,
                     # target, source checkpoint, versions, parity result
    target.png       # padded target for the UI (if available)
```

- `--verify` runs N steps in PyTorch and `onnxruntime` (CPU) with identical masks at
  **two different shapes** and asserts `max_abs_diff < 1e-4`.
- `--grid-size` sets the dummy shape; spatial axes are dynamic by default and
  `--static` fixes them. `--all <run-dir>` (future) exports every checkpoint.
- The dynamo exporter writes weights as **external data** by default
  (`model.onnx.data`), which ORT-Web will not fetch automatically — the exporter
  passes `external_data=False` so the `.onnx` is self-contained.

## Frontend architecture

Vite + TypeScript + `onnxruntime-web`, minimal deps. WebGPU only, with a clear
"WebGPU unavailable" gate.

```
frontend/
  vite.config.ts        # dev server; publicDir -> ../exports
  index.html
  src/
    main.ts             # bootstrap
    runner.ts           # CaRunner: session, ping-pong buffers, step(), seeded fire mask
    render.ts           # state -> ImageData (premultiplied RGB over bg), Canvas2D
    models.ts           # manifest types + loading
    analysis/           # per-channel views, gradient views, stats, timeline
    ui/                 # controls, canvas grid, kernel editor, timeline scrubber
```

- **CaRunner**: `InferenceSession` with `['webgpu']`, warm-up run on load, ping-pong
  `Float32Array`s, optional IO binding to keep tensors on GPU, seeded PRNG for masks.
- **Render**: Canvas2D `putImageData` at grid resolution, scaled with
  `image-rendering: pixelated`. Upgrade path: WebGL/WebGPU texture compositing.
- **Loop**: `requestAnimationFrame`, steps/frame from a speed control.

### Analysis features (single-model deep dive)

- **Channel-mapping views**: any number of panels; each maps any three state
  channels to R/G/B, with an optional alpha channel and background for
  compositing. The first view defaults to channels 0/1/2 with alpha 3 over white
  (the familiar RGB view); new views default to the next hidden channel as a
  grayscale map.
- Perception gradient views (gx, gy) and update magnitude.
- Fire-mask overlay.
- Timeline: ring-buffer of snapshots, scrub / step-back / replay.
- Kernel lab: identity / sobel_x / sobel_y editors + rotation slider (Experiment 4).
- Perturbation: damage brush, seed planting, reset, seeded RNG.
- Stats: per-channel mean/min/max/variance over time, loss-to-target curve.
- Controls: speed, fire_rate, step_size, grid size.

## Repo layout

```
nca/export.py             export CLI + parity verification
exports/                  committed models + manifest.json
frontend/                 Vite app (publicDir -> ../exports)
```

## Milestones

1. **M1** — `export.py` + Python parity check at two shapes. ✅ done
   - Uses the modern `torch.onnx.export(dynamo=True)` path (opset 18) with
     `dynamic_shapes`. Gotcha: `dynamic_shapes` entries are **positional / forward
     arg names** (`x`, `w`, `fire`, `step_size`), *not* ONNX input names.
   - The dynamo graph is also smaller (14.9 KB vs 36.3 KB for the legacy exporter).
2. **M2** — Vite app: load one model, grow, RGB view, play/pause/speed. ✅ done
3. **M3** — channel/gradient/fire-mask views + timeline scrubber.
4. **M4** — kernel lab + perturbation tools.
5. **M5** — stats, polish, pin ORT build, profile graph-capture trade-off.

## Open questions

- Checkpoint handoff from the GPU cluster: same repo dir vs. separate artifacts path?
  (Exporter already takes an arbitrary `--checkpoint` path, so this is low-risk.)
