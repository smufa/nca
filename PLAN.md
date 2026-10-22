# Plan: NCA → ONNX → playground

Reimplementation of *Growing Neural Cellular Automata* (Mordvintsev et al., Distill
2020). This document covers the **inference/export pipeline and the web frontend**.
Training lives in `README.md` / `TRAINING.md`; measurements in `docs/findings/`.

Its job is now narrower than when it was written: it records the **decisions that are
expensive to re-derive** (graph interface, dynamic shapes, export flags) and the
roadmap. Anything that turned out to be a measurement lives in `docs/findings/`.

## Goals

- Export a trained checkpoint to ONNX as a **single CA step**.
- Run it in the browser two ways: `onnxruntime-web` (WASM, compatibility) and a native
  **WGSL** engine (performance, the default fast path).
- A **general playground** for deep analysis of a single model — not a guided
  narrative, not multi-model comparison.

## Decisions

| # | Decision | Why / consequence |
|---|---|---|
| 1 | **Perception kernels are a graph *input***, not initializers | the kernel lab can rotate / scramble them without re-exporting. Costs a runtime transpose of ~1.7 KB per step and forfeits graph capture; measured negligible. |
| 2 | Perception + update in **one** ONNX model | one `session.run`/dispatch per step; the exported graph *is* the step. |
| 3 | Two frontend engines, `wasm` + shader | the ORT WebGPU EP was removed — see `docs/findings/frontend-engines.md`. |
| 4 | General **playground**, single-model deep analysis | per-model UI, no cross-model tables. |
| 5 | Training is separate; the exporter is **standalone** and takes a checkpoint path | cluster → laptop handoff stays a file copy. |
| 6 | **Commit** exported models (~10–50 KB each) | the playground works from a clone, no download step, and the metas are the regression corpus for `nca/spec.py`. |
| 7 | Dynamic `H`/`W` | see below; `--static` opts out. |
| 8 | The model's **shape is data** (`nca/spec.py`, `spec.json`, `meta.json`) | never infer `C`/hidden from tensor shapes — that was the `nca/sample.py` bug. |

## ONNX graph interface

```
Inputs
  x                   float32 [B, C, H, W]     state grid
  perception_weight   float32 [3C, 1, 3, 3]    JS expands the 3 base kernels per channel
  fire_mask           float32 [B, 1, H, W]     per-cell update mask (0/1)
  step_size           float32 []               optional
  cond                float32 [B, cond_n, 1, 1]  only for --targets models
Output
  x_next              float32 [B, C, H, W]
```

Graph: depthwise conv (`groups=C`, the kernels above) → `3C→hidden` → ReLU →
(optional hidden→hidden) → `hidden→C` → `x + dx*fire_mask*step_size` → living-cell
masking (`MaxPool` → `Greater` → `And` → `Mul`), all inside one
`Pad(mode="wrap")`. The exact layer stack is `ModelSpec.hidden_layers`; the weight order is
`ModelSpec.weight_offsets()` (which `weights.bin` and the shader both follow).

The UI edits three 3×3 matrices; JS expands them with `w[3*ci+cm] = kernel[cm]` so the
conv output stays `[state, grad_x, grad_y]` per channel, preserving the trained update
rule's semantics.

### Dynamic shapes

ORT's session (`wasm`) does not validate input dims against the graph: `run` checks
feed presence/names and builds tensors from the supplied dims. The classic "expected
shape `[,,,]`" error comes from the legacy onnxjs/WebGL backend, which is not in play.
So `H`/`W` stay dynamic and `--verify` re-runs at a second shape to prove it. The
shader engine has no shape at all (`H`/`W` are storage-buffer extents set per launch).

## Export pipeline

```bash
uv run python -m nca.export --checkpoint runs/x/ckpt_final.pt --id NAME \
    --experiment regenerating --target "emoji:1f980"   # parity checked by default
uv run python -m nca.export --all        # every model whose checkpoint is still reachable
uv run python -m nca.export --reindex    # rebuild manifest.json from exports/*/meta.json
```

`exports/<id>/` = `model.onnx`, `meta.json`, `weights.bin`, `target.png`, `spec.json`.

- **Parity is on by default** (`--no-verify` to skip): N steps in PyTorch vs
  onnxruntime with identical masks at two shapes; ~1e-6 is normal.
- **Dynamo exporter** (`torch.onnx.export(dynamo=True)`, opset 19,
  `external_data=False` so the `.onnx` is self-contained — ORT-Web will not fetch
  `model.onnx.data`). `dynamic_shapes` takes positional / forward arg names, not ONNX
  input names.
- **Wrap padding is one `Pad(mode="wrap")` node** (20 Slice/Concat nodes → 1).
- `meta.json` is validated against `nca/spec.py` by `scripts/check_meta.py` (CI); it
  records version, parity error, git rev, training provenance and the sweep it belongs
  to. Re-exporting 45 models by hand is how drift happens — `--all` + `check_meta` is
  the supported path.

## Frontend architecture

Vite + TypeScript, minimal deps, two engines behind one interface.

```
frontend/src/
  engine.ts     CaEngine: the only surface main.ts talks to (step/render/brush/sync/cond)
  runner.ts     CaRunner   : ort-wasm session, ping-pong Float32Array, CPU mirror
  shader.ts     ShaderRunner: WGSL pipelines, GPU state buffer, render + brush on GPU
  wgsl/         pass1 (perception+MLP+masks), pass2, draw (premultiplied composite)
  nca.ts        pure helpers: kernel expansion, seed, channel->RGB mapping
  main.ts       UI: views, brush, kernel lab, engine/grid switches
  models.ts     manifest types, target labels/emoji, picker grouping
```

- **Loop**: `requestAnimationFrame`, steps/frame from a speed control.
- **Views**: N panels, each mapping any three state channels to RGB (plus alpha and
  background) — the first defaults to 0/1/2 over white, new ones to the next hidden
  channel as grayscale.
- **Kernel lab**: identity / sobel_x / sobel_y editors + rotation. This is the probe
  that found the robustness result in `docs/findings/robustness.md`.
- **Target picker**: only for conditional models; a *vector* of sliders (-1..1), and
  switching it **does not reset the state** (in-place conversion is the measurement —
  `docs/findings/conditioning.md`).

## Roadmap

Shipped: M1 export + parity · M2 Vite app load/grow/play · M3 views/brush/grid ·
M4 kernel lab · M5 WGSL engine + headless bench · M6 `ModelSpec` + run-directory
provenance + tests/CI.

Next, in rough order:

1. **Larger-grid / texture targets** and the shader headroom that implies
   (tiled GEMM; `docs/findings/shader-perf.md` lists what was already rejected).
2. **Timeline scrubber** (ring buffer of snapshots) + per-channel stats over time.
3. **Perception-view export** — a graph variant that also emits `[state, gx, gy]`.
4. **Scoring harness** (`nca/score.py`) so the conversion/quality numbers in
   `docs/findings/*` are generated, not typed.
5. Docs: delete `README.md`'s duplicated design sections once `PLAN.md` covers them
   (Phase 3 in `docs/HOUSEKEEPING.md`).

## Open questions

- Checkpoint handoff: upload artifacts vs. same repo dir? (The exporter takes an
  arbitrary `--checkpoint`, so this stays low-risk until runs get big.)
- Whether the 45 committed exports are still worth their weight as a *regression
  corpus* once `ModelSpec` round-trips are covered by fixtures — they also cost
  frontend picker clutter.
