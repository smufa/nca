---
name: headless-webgpu
description: Run, validate and benchmark WGSL compute shaders headlessly on a real GPU without a browser, using Python wgpu. Use when testing WebGPU/WGSL code, measuring GPU kernel performance, iterating on shaders when no browser is available, or debugging "No suitable graphics adapter found" / Vulkan setup errors (includes the NixOS specifics).
---

# Headless WebGPU / WGSL harness

Run a WGSL compute shader on the GPU from Python, compare its output against a
reference, and time it — no browser, no JS. This is the fastest loop for shader
development. `scripts/wgpu_run.py` is a minimal, runnable example; copy it and
edit the shader + buffers.

## 0. Environment (do this first)

```bash
uv run --with wgpu --with numpy python -c \
  "import wgpu; print(wgpu.gpu.request_adapter_sync().info)"
```

If that raises `No suitable graphics adapter found`, the Vulkan **loader** is not
on the library path. On NixOS the ICD drivers live in `/run/opengl-driver/lib`
but the loader ships separately in the nix store:

```bash
LOADER_DIR=$(dirname "$(ls /nix/store/*vulkan-loader*/lib/libvulkan.so.1 | head -1)")
export LD_LIBRARY_PATH="$LOADER_DIR:/run/opengl-driver/lib"
export VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/intel_icd.x86_64.json
```

- Pick a hardware ICD from `/run/opengl-driver/share/vulkan/icd.d/`
  (`intel_icd`, `radeon_icd`, `nouveau_icd`, …).
- Use `lvp_icd.x86_64.json` (lavapipe) to force the **software** adapter — handy
  for CI, but do not benchmark on it.
- Inspect what you actually got: `adapter.info["device"]`,
  `adapter.info["adapter_type"]`.

## 1. wgpu-py API gotchas

These differ from the browser `navigator.gpu` API and cost time if unknown:

- The exposed API is `sync` by default: `request_adapter_sync()`,
  `request_device_sync()`, `buffer.map_sync(...)`.
- **There is no `device.poll()`** — submit, then `map_sync(mode=wgpu.MapMode.READ)`
  blocks until the data is ready.
- **Readback** needs a buffer with `MAP_READ | COPY_DST`, a
  `copy_buffer_to_buffer`, a submit, then `map_sync` + `read_mapped()`
  (returns a `memoryview`; `np.frombuffer(...).copy()` before `unmap()`).
- **Any buffer you copy *from* needs `COPY_SRC`** — a `STORAGE` buffer is not
  copyable by default. Missing this gives a confusing
  "Usage flags ... do not contain required usage flags (COPY_SRC)" error.
- **Do not use `layout="auto"` for pipelines driven from a Python harness.**
  Auto-layout omits bindings the shader doesn't statically use, so your bind
  group (which has all of them) fails to match — "Number of bindings ... does not
  match". Create an explicit `create_bind_group_layout` listing every binding and
  `create_pipeline_layout` from it. This also lets shader variants that drop a
  binding still run.
- Enums accept strings (`usage="storage"`), but
  `wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST` is clearer.
- `device.create_buffer_with_data(data=..., usage=...)` for static data;
  `device.queue.write_buffer(buf, 0, data.tobytes())` to update it.

## 2. Validate against a reference

Never trust "it runs" — diff it. Keep a trusted implementation (numpy, PyTorch,
onnxruntime, or the previous shader) and feed **identical inputs**, then assert
`max|diff| < tol`. For iterative kernels, run the same number of steps on both
sides with the same per-step inputs, then compare the final buffer.

Tolerance: `~1e-5` for f32 reduction-order differences; anything larger is a bug.

## 3. Measure properly

- Time **best of N** runs, and call a **readback at the end** to drain the GPU —
  otherwise you time submission, not execution.
- A GPU submit is asynchronous: if a "run" looks impossibly fast, you are only
  measuring the enqueue. Force completion with `map_sync` (or a fence) before
  stopping the clock.
- Use `--reference` style flags so correctness and speed are one command.

## 4. Finding the bottleneck (the method that actually works)

1. **Isolate stages in the shader**: comment out parts (e.g. replace the inner
   layer with a trivial expression) and re-benchmark. In one case this showed the
   MLP inner loop was 98% of the step and the neighbouring stage was noise.
2. **Scale a dimension** (grid size, batch, iteration count). If time scales with
   work, it is throughput-bound; if it is flat, it is per-dispatch/per-pass
   overhead. That distinction decides the fix: fewer passes vs. better kernels.
3. **Count memory loads, not just FLOPs.** Kernel throughput is often load-bound.
   **Vectorise loads with `vec4`** (`array<vec4<f32>>` view of the same buffer)
   and compute several outputs per load — this is usually the cheapest big win.
4. **Reject experiments that regress.** Keep them in a `variants/` directory with
   the measured numbers so nobody re-tries them (e.g. shared-memory staging can
   cost occupancy, explicit ILP can duplicate what the compiler already does,
   register tiling can spill).

## Reference

- `scripts/wgpu_run.py` — runnable compute-shader skeleton: buffers, explicit
  layouts, dispatch, readback, timing.
- Real-world example: `scripts/shader_bench.py` in this repo compares a WGSL
  cellular-automaton step against onnxruntime and benchmarks it.
