# Frontend engines

| engine | runtime | notes |
|---|---|---|
| `wasm` | onnxruntime-web WASM | **compatibility fallback**; runs anything the shader cannot |
| `shader` | native WGSL | **the performance path**; 2 passes, or 4 when work-skipping (deep models keep 2), zero-copy render + GPU brush |

Both implement `CaEngine` (`frontend/src/engine.ts`); the UI picks one from the
Engine dropdown (`?ep=wasm|shader`, persisted in `localStorage["nca.ep"]`).

## The ORT WebGPU EP was removed

Each ONNX node becomes its own dispatch (~1–12 ms per node on an iGPU), so the EP
lost to both WASM and the shader. Dropping it also moved the import to
`onnxruntime-web/wasm`, which halved the bundled wasm (26.8 → 14.2 MB) and cut the
JS by ~30%.

Do not reintroduce its provider options (`validationMode`, buffer cache modes
either): they abort with `InternalError: out of memory` in the current ORT-Web
build.

## Why `shader` has two passes

`CaRunner`/`ShaderRunner` keep the CA state in one GPU buffer. `pass1` runs the
perception conv + MLP + fire/living masks, `pass2` (or the render path) writes what
each view shows. `shader` renders each view straight from the GPU state buffer
(`renderTo`) and applies brush strokes with a compute shader (`brush`), so nothing
is downloaded per frame. `wasm` renders through Canvas2D and calls `sync()` to
refresh the CPU mirror `x`.

## Work-skipping: 4 passes instead of 2

At `fire_rate 0.5` half of every grid is multiplied by zero *after* the update rule
has run, and the update rule is ~73% of a step. So for single-hidden-layer models
(the common case) `ShaderRunner` swaps `pass1` for three passes:

1. `frontend/src/wgsl/compact.wgsl` — copy the state, append firing cells to a work
   list with a global atomic,
2. `frontend/src/wgsl/finalize.wgsl` — list length -> indirect dispatch args,
3. `frontend/src/wgsl/step_list.wgsl` — perception + MLP for the listed cells only.

It is **bit-identical** to the plain path (same maths, same order; measured
`max|d| = 0` over 20 steps) and 1.3x (72²) to 1.6x (256²) faster. Deep models keep
`pass1`: only the shallow rule has a list-driven entry point. `?ws=0` forces the
plain path so the two can be compared side by side, and
`Capabilities.workSkip` reports which one is running.

The three shaders are the shipped sources, so `scripts/work_skip_bench.py`
(`bash scripts/bench.sh work-skip <model>`) measures production code; it checks
parity first and refuses to time a mismatch. `tests/test_engine_bindings.py`
asserts the engine's bind groups still match the shaders' `@binding` order — with
`layout: 'auto'` a wrong order is a silent buffer swap.

Weight layout is the contract with `nca/export.py` (`weights.bin`) and comes from
`ModelSpec.weight_offsets()`; see `shader-perf.md` for the load-bound measurements
and the compile-time overrides (`C`, `NP4`, `W1S4`, `HG`, `CONDN`).

## Brush edits

Brush writes go through `CaEngine.brush` (shader) or `CaEngine.edit` (ORT), which
queue the edit while a step is in flight. A direct write during `session.run` is
lost.
