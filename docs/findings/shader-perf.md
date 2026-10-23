# Shader performance

`scripts/shader_bench.py` runs the **real** WGSL shaders headlessly on the GPU via
Vulkan (`wgpu-py`) and diffs the result against onnxruntime — use it instead of
guessing. See the `headless-webgpu` skill for the NixOS/Vulkan setup.

```bash
bash scripts/bench.sh crab.regenerating --steps 150 --reference   # correctness + perf
bash scripts/bench.sh crab.regenerating --grid 256
# several shaders per invocation are benchmarked ROUND-ROBIN -- always do this
bash scripts/bench.sh crab.regenerating --steps 40 --grids 256,512 \
    --shader scripts/variants/pass1_head.wgsl frontend/src/wgsl/pass1.wgsl
```

`--reference` reports a **1-step** diff as the correctness gate. The N-step diff is
larger because the CA is chaotic — a 1-step agreement plus a visual check is the
real signal, not a 100-step max-abs.

## Measure two shaders at a time, or not at all

The integrated GPU (Intel UHD 620) drifts up to **±25%** between sessions: the same
unchanged shader measured 13.0 ms and 16.2 ms/step on different runs, and its clock
depends on what ran before it. Absolute numbers in this file are therefore only
meaningful next to a reference shader measured **in the same invocation**, which is
what `--shader a.wgsl b.wgsl` does (one repetition per shader, repeated, best-of-N
per shader). Ratios reproduce to a few percent; absolute numbers do not.

## The step, as measured

`crab.regenerating` (C=16, hidden=128, one layer), 40 steps, interleaved A/B:

| grid | naive (`pass1_head.wgsl`) | current `pass1.wgsl` | speedup |
|---|---|---|---|
| 72² | 1.93 ms | 1.49 ms | 1.29× |
| 128² | 5.92 | 4.36 | 1.36× |
| 256² | 22.58 | 16.60 | 1.36× |
| 512² | 93.50 | 68.29 | 1.37× |
| 1024² | 376.31 | 280.78 | 1.34× |

The speedup is flat in grid size, so it is not about traffic or occupancy — it is
per-cell instruction latency.

**What that 1.35× is:** splitting the shallow and deep update rules into two entry
points (`step`, `step_deep`) plus 4-way ILP accumulators in the `W1` loop. The split
is worth more than the ILP: a single kernel that contains both paths keeps the deep
path's `h`/`h2` arrays live and runs the whole thing at ~2.2× the shallow cost
(30.3 vs 13.8 ms/step at 256², measured before the split), because the compiler
allocates for the worst branch.

**Where the time goes** (256², same session): total 16.63 ms, of which perception +
copy-out is 4.55 ms (`pass1_nomlp.wgsl`, MLP replaced by a no-op) — so the MLP is
~73% and perception ~27%. That ratio is the reason the MLP (`W1`: 128×12 `vec4`
FXPs per cell) is the only place worth optimising.

## Tried and rejected (all slower, all still runnable)

One interleaved run at 256², 30 steps, current shader as the reference:

| variant | ms/step | vs current | why it lost |
|---|---|---|---|
| `frontend/src/wgsl/pass1.wgsl` | 16.63 | 1.00× | — |
| `pass1_fireearly.wgsl` | 18.46 | 1.11× slower | skip the MLP when `fire < 0.5`; half a workgroup takes the copy path while the rest run the full MLP, so nothing is short-circuited and the branch costs more than it saves |
| `pass1_wg16x16.wgsl` | 22.80 | 1.37× slower | fewer blocks to hide latency |
| `pass1_fastperception.wgsl` | 23.64 | 1.42× slower | perception was never the bottleneck (27%), and the hoisted version raises register pressure |
| `pass1_tiled.wgsl` | 43.73 | 2.63× slower | stages `W1` (32 hidden rows/chunk) and all of `W2` in workgroup memory; the barriers cost far more than the global/L2 traffic they save, because the weight stream was already served from L1 |
| `pass1_nomlp.wgsl` (probe) | 4.55 | — | not a candidate: measures perception + copy-out alone |
| `pass1_packed.wgsl` | 15.49 | 1.02× faster | channel-interleaved state (4 channels per `vec4`): correct (2.4e-07), but perception is only 27% of the step and already L1-served, so the layout buys ~2% |
| `pass1_noperc.wgsl` (probe) | 73.21 | — | invalid probe: without perception the state goes NaN and denormals/NaN handling dominates |

Lessons, in order of how much they cost to learn:

- **The weights are L1-resident.** Cutting global weight traffic (tiling, staging)
  buys nothing; that is the single most surprising result here.
- **The compiler does not schedule the MLP well on its own.** Explicit
  accumulate-then-sum ILP is worth ~10%, and it is free.
- **A branch that skips work must also skip the registers.** Both the fire early-out
  and the merged deep/shallow kernel lost because the skipped path still shaped
  register allocation.
- **Probes need a sanity check.** `noperc` ran 4.4× slower than the full kernel
  because dropping perception produces NaN state — a probe that changes what the
  data *is* does not measure what you think.

## Remaining ideas

0. ~~**Packed state layout**~~ — **measured, not worth it.** `scripts/variants/
   pass1_packed.wgsl` (+ `pass2_packed.wgsl`: the living mask reads alpha as
   `(3*H+y)*W+x`, so on a packed buffer every cell silently dies) gives 15.49 vs
   15.77 ms/step at 256² — ~2%, for an AoS↔SoA change through the engine, the
   render pass and the brush. The earlier "highest headroom" claim here was a
   guess; the probe data (MLP ~73%, perception L1-served) already predicted it.
1. **fp16 weights/activations** — the iGPU exposes `shader-f16`; maybe ~2× on the
   MLP, a real precision risk. The bench settles it in an afternoon.
3. **GEMM-over-cells** (tile cells into workgroup memory, `W1` as the shared
   operand) — the principled fix for the deep path, and a genuine WGSL matmul
   project rather than a tweak. The naive version of this idea is the tiling above,
   which lost; doing it properly means giving each thread several output cells.
4. **Quantised weights** (`unorm8` + per-tensor scale, as the Distill engine does) —
   only worth it once `hidden` outgrows L1, i.e. not on these models.

## The kernel is size-generic, at compile time

`C`, `NP4`, `W1S4`, `HG`, `CONDN` and `LH` are WGSL `override` constants set per
pipeline. They must stay **compile-time** — dynamic loop bounds cost ~1.9×. `LH`
also selects the entry point: `hidden_layers == 1` uses `step`, deeper models use
`step_deep`. `nca/export.py` pads each `W1` row from `3C` to `ceil(3C/4)*4` floats
so `vec4` loads stay aligned; at `C=16` that is a no-op, so old `weights.bin` files
are unchanged (at `C=6`, 2 floats per row). Supported: `4 ≤ C ≤ 16`,
`hidden % 4 == 0` (`ShaderRunner.supports`), otherwise the frontend falls back to
the WASM engine.

Multi-layer MLPs work (`hidden_layers > 1`), but the `hidden -> hidden` blocks cost
~2.7× the `W1` work plus the register pressure of holding the whole hidden vector:
14.7 ms/step vs 4.4 for a 128-wide single-layer model at 128². That is what
`step_deep` exists to contain, and what idea 3 above would fix.

## Correctness across the branches

Checked with `--reference` after the entry-point split (1-step max-abs vs ONNX):

| model | covers | 1-step diff |
|---|---|---|
| `crab.regenerating` | C=16, hidden=128, 1 layer, wrap | 1.19e-07 |
| `crab.regenerating.c=8.h=128` | `NP4 % 4 != 0` (scalar MLP path) | 5.96e-08 |
| `smile.demo` | zero padding (not toroidal) | 8.94e-08 |
| `cross8.mismatch.l=2` | `hidden_layers = 2` (`step_deep`) | 2.68e-07 |

## fp16: measured, and it is 1.4× *slower* here

The tempting "sacrifice accuracy for speed" switch. Built as
`scripts/variants/pass1_f16.wgsl` — identical structure to `pass1.wgsl`, but the
perception vector's `vec4` groups, `W1`, the ReLU'd hidden vector, `W2` and the
per-channel accumulator are `f16`; the state, the perception convolution and the
final `state + dx * fire * step` stay `f32`.

Measured on `crab.regenerating` (Intel UHD 620, KBL GT2), 40 steps, best of 3:

| grid | f32 `pass1` | `pass1_f16` | |
|---|---|---|---|
| 72² | 1.29 ms | 1.81 ms | **1.40× slower** |
| 256² | 12.92 ms | 18.58 ms | **1.44× slower** |

The adapter *advertises* `shader-f16` (as do `shader-i16`,
`texture-format16bit-norm`), and the shader compiles and runs correctly — but on
Gen9 the Vulkan driver emulates `float16` arithmetic with `float32` conversions,
so every `dot(vec4<f16>)` becomes a convert + the same f32 dot. Halving the ALU
width only pays on hardware with native fp16 ALU (Arc, RDNA, Apple); on this iGPU
the conversions are pure overhead. `scripts/shader_bench.py` now requests
`shader-f16` when the adapter has it, so the experiment is repeatable.

Accuracy, for the record (`scripts/fp16_drift.py`, 500 steps, fixed fire masks,
real `weights.bin`, vs the f32 step):

| variant | state max\|Δ\| | render PSNR | alpha IoU |
|---|---|---|---|
| f32 + 1e-7 seed noise | 4.1e-06 | 132 dB | 1.0000 |
| weights f16, f32 accumulate | 3.6e-03 | 75.5 dB | 1.0000 |
| activations f16, f32 accumulate | 6.7e-04 | 94.1 dB | 1.0000 |
| all f16, including accumulate | 1.4e-03 at 1 step | — | — |
| all f16, 500 steps (no living mask, as the bench compares) | **1.07** | — | — |

Two things worth keeping from that table. The CA is **contractive over short
horizons**: a 1e-7 perturbation reaches only 4.1e-06 after 500 steps, so fp16's
1e-3 error does not run away on its own. But f16 *accumulation* (12 terms for a
`W1` row, 128 for `W2`) raises the per-step error to 1.4e-03, and at ~2%/step of
amplification that saturates at O(1) after 500 steps — the pattern lands on a
visibly similar but different state. fp16 storage with f32 accumulation is the
variant that stays within 1e-2; it is also the slowest thing measured here.

Conclusion: do not expose an fp16 toggle on this hardware, and the earlier
assumption in this file that "the CA is chaotic, so error compounds" was wrong for
short horizons — it is contractive; what amplifies is the *living-mask threshold*
once the error reaches ~0.1, which is why the failure mode is a pattern that
heals into a slightly different pattern rather than a blur.

## Work skipping: compaction instead of a branch (1.5-2.7x, shipped as an experiment)

At `fire_rate 0.5` half of every grid is copied unchanged, yet `pass1.wgsl` runs
the full MLP for those cells and multiplies the result by zero. The branch version
(`pass1_fireearly.wgsl`, -11%) fails because a workgroup of mixed cells pays both
paths. Compaction sidesteps that: list the cells that fire, then dispatch the
update rule with `dispatch_workgroups_indirect` over the list only.

Four dispatches per step instead of two (`scripts/work_skip_bench.py`, which drives
the shipped shaders):

1. `frontend/src/wgsl/compact.wgsl`: `out = state` for every cell, plus `atomicAdd` of firing cells
2. `frontend/src/wgsl/finalize.wgsl` (1 thread): list length -> indirect args, `ceil(n/64)`
3. `frontend/src/wgsl/step_list.wgsl`: the shipped `step` body, with the cell index from the list, dispatched
   with `dispatch_workgroups_indirect` (one 64-wide workgroup per 64 cells)
4. `pass2` unchanged (living mask)

**Shipped**: `ShaderRunner` runs these for single-hidden-layer models (deep models
keep `pass1`, whose `step_deep` has no list entry point). The three shaders moved
from `scripts/variants/` to `frontend/src/wgsl/` so the benchmark and the engine
measure the same code; `?ws=0` selects the plain path in the browser.
`tests/test_engine_bindings.py` guards the one thing TypeScript cannot: with
`layout: 'auto'`, an entry in the wrong order silently swaps buffers.

Bit-exact against the baseline pipeline over 20 steps (`max|d| = 0.000e+00`), because
the MLP code is `pass1.wgsl`'s, unchanged.

| grid | fire_rate | baseline | work-skip | speedup |
|---|---|---|---|---|
| 72² | 0.5 | 1.25 ms | 0.84 ms | **1.50x** |
| 256² | 0.5 | 12.6 | 8.0 | **1.59x** |
| 512² | 0.5 | 49.0 | 29.8 | **1.65x** |
| 1024² | 0.5 | 218 | 138 | **1.59x** |
| 256² | 0.2 | 12.6 | 4.6 | **2.74x** |
| 256² | 1.0 | 12.6 | 13.6 | **0.93x** |

The `fire_rate 1.0` row is the control: with every cell firing, compaction costs
7% (the copy pass, the atomics, and the now-scattered perception gather instead of
a coalesced one). So the saving is `~1/(1 + 0.07) * (1 + f*(1-1/r))`-shaped: it
scales with the fraction of cells that *do not* fire, and the losses are small
enough that the scheme is worth having unconditionally — but only above a few
hundred live-neighbourhood cells, and only where indirect dispatch exists (WebGPU
core; not in WebGL2).

Two yields nobody expected: the fire-rate slider currently does not affect step
cost at all (every cell computes regardless), so this is what makes "lower fire
rate" a real performance knob; and the scattered gather after compaction is cheap
because the perception reads land in the same few L2 lines the baseline already
touches.

### int8: not viable on either axis

Asked as the obvious next step down ("or something like that"). Measured with the
same harness (`scripts/fp16_drift.py --variants int8`, symmetric per-output-row
int8 weights, dequantised, 500 steps, fixed fire masks):

| variant | state max\|Δ\| | render PSNR | alpha IoU |
|---|---|---|---|
| fp16 weights | 3.6e-03 | 75.5 dB | 1.0000 |
| **int8 weights** | **8.8e-01** | **27.1 dB** | **0.9523** |

An 8-bit code gives ~4e-3 *relative* error per weight, which is 1e-2..1e-1 on the
row's largest term, and the CA's alpha channel crosses the 0.1 living-mask
threshold a few steps later — the pattern does not blur, it dies or deforms. Unlike
fp16 there is no safe "storage-only, f32 accumulate" configuration that rescues it.

The speed axis is worse. The adapter exposes 56 features, including `shader-f16`
and `shader-i16` — and **not** `packed-4x8-integer-dot-product`, so WGSL has no
`dot4I8Packed` here. An int8 kernel would unpack 4 bytes to 4 floats and run the
same f32 dot, i.e. the conversion trap that made fp16 a 1.4x *regression*, plus
dequantisation scales. Conclusion: no int8 path, and on this class of hardware no
precision-lowering path at all — the win has to come from doing less arithmetic
(work skipping), not narrower arithmetic.
