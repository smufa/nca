# Frontend engines

| engine | runtime | notes |
|---|---|---|
| `wasm` | onnxruntime-web WASM | **compatibility fallback**; runs anything the shader cannot |
| `shader` | native WGSL | **the performance path**; 2 dispatches, zero-copy render + GPU brush |

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

Weight layout is the contract with `nca/export.py` (`weights.bin`) and comes from
`ModelSpec.weight_offsets()`; see `shader-perf.md` for the load-bound measurements
and the compile-time overrides (`C`, `NP4`, `W1S4`, `HG`, `CONDN`).

## Brush edits

Brush writes go through `CaEngine.brush` (shader) or `CaEngine.edit` (ORT), which
queue the edit while a step is in flight. A direct write during `session.run` is
lost.
