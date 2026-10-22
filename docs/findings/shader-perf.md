# Shader performance

`scripts/shader_bench.py` runs the **real** WGSL shaders headlessly on the GPU via
Vulkan (`wgpu-py`) and diffs the result against onnxruntime — use it instead of
guessing. See the `headless-webgpu` skill for the NixOS/Vulkan setup.

```bash
bash scripts/bench.sh crab.regenerating --steps 150 --reference   # correctness + perf
bash scripts/bench.sh crab.regenerating --grid 256
bash scripts/bench.sh crab.regenerating --shader scripts/variants/foo.wgsl
```

`--reference` reports a **1-step** diff as the correctness gate. The N-step diff is
larger because the CA is chaotic — a 1-step agreement plus a visual check is the
real signal, not a 100-step max-abs.

## What the step costs (Intel UHD 620)

- The **MLP is ~98%** of the step; perception is ~0.06 ms.
- It is bound by **weight loads**, not FLOPs.
- `vec4` weight loads + 4-hidden-at-a-time: **3.2–3.6×** over the naive shader
  (`frontend/src/wgsl/pass1.wgsl`). Current: **72² 1.54 ms · 128² 4.40 ms ·
  256² 16.8 ms**.

## Tried and rejected (slower)

| experiment | result |
|---|---|
| `perc` in workgroup shared memory | added LDS traffic + cost occupancy; the private array was already in registers |
| explicit 4-way ILP in the MLP loop | the compiler already schedules it |
| 16×16 workgroup | worse occupancy than the current size |
| 2-cell register tiling | register spill → 5.2 ms **and** wrong output |

They are kept in `scripts/variants/` so the negative results are still runnable.

## Remaining ideas (not tried)

1. **Tiled GEMM for the MLP** — tile cells into shared memory and stream the
   weights once per tile. The proper fix at 512²+, but a real WGSL matmul project.
2. **fp16** weights/activations — the iGPU exposes `shader-f16`, maybe ~2×, but a
   real precision risk. The bench settles it immediately.
3. **Split hidden units across workgroups** and reduce `dx` with atomics, to cut
   per-thread weight reads.

## The kernel is size-generic, at compile time

`C`, `NP4`, `W1S4` and `HG` are WGSL `override` constants set per pipeline. They
must stay **compile-time** — dynamic loop bounds cost ~1.9×. `nca/export.py` pads
each `W1` row from `3C` to `ceil(3C/4)*4` floats so `vec4` loads stay aligned; at
`C=16` that is a no-op, so old `weights.bin` files are unchanged (at `C=6`, 2 floats
per row). Supported: `4 ≤ C ≤ 16`, `hidden % 4 == 0` (`ShaderRunner.supports`),
otherwise the frontend falls back to the WASM engine.

Multi-layer MLPs are supported too (`hidden_layers > 1`, `cross8.mismatch.l=2`), but the
`hidden -> hidden` blocks cost ~2.7× the `W1` work plus register pressure from
holding the hidden vector: ~98 steps/s vs ~620 for a 128-wide single-layer model.
