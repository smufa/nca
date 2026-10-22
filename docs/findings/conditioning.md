# Conditional (multi-target) models

`--targets SPEC...` makes one CA reproduce several targets: a `cond_n`-wide one-hot
is appended to every cell's perception, so `update[0]` takes `3C + cond_n` inputs.

It is a **separate input, not an extra state channel**: `x = x * life_mask` would
zero it in every dead cell, and since the exported graph *is* the step, the frontend
could not re-clamp it. `prepare_targets()` centres each spec in a `max_size` canvas
so non-square emoji share a grid, and the pool carries a per-slot `cond` (a state
only means something together with the target it was grown towards). Both ONNX (a
`[B, cond_n, 1, 1]` input) and the shader (`CONDN` + the one-hot appended to `perc`)
support it.

## Matched training: capacity, not conditioning, is the limit

`--mismatch-p 0` (hidden stays 128):

| run | targets | params | min loss |
|---|---:|---:|---:|
| `multi2` | 2 | 8592 | 2.4e-3 |
| `multi4.matched` | 4 | 8848 | 1.9e-3 |
| `multi6` | 6 | 9104 | 3.4e-3 |
| `multi8` | 8 | 9360 | 3.9e-3 |

All four grow the *correct* emoji for each one-hot index (verified by rolling each
index out from the seed) — **no condition collapse even at 8 emoji**. Quality does
fall with N (visibly blurrier at 8), so one shared MLP is capacity-limited rather
than conditioning-limited. These losses are ~10× a single-target 2.0e-4 and are not
comparable across regimes — check the playground, not the number.

## `--mismatch-p` is what makes the conditioning *steer* (`cross*` vs `multi*`)

The same sweep with `--mismatch-p 0.3` (`cross2..8`) costs ~1.5–2× in loss
(min 3.2e-3 / 3.4e-3 / 4.1e-3 / 5.8e-3 for N=2/4/6/8 vs 2.4e-3 / 1.9e-3 / 3.4e-3 /
3.9e-3 matched), because the task is strictly harder. But it buys the property the
matched models do not have. Grow emoji 0, then switch the one-hot to emoji 1
mid-run (err vs each target's image, 0-255):

| model | err vs source | err vs dest | result |
|---|---:|---:|---|
| `multi4.matched` | 20.5 | 10.1 | **a purple crab** — recolours a memorised attractor, body plan frozen |
| `cross4` | 24.3 | 5.3 | crab → octopus, morphs |
| `cross8.mismatch` | 24.4 | 5.8 | morphs |

So a matched model treats the one-hot as an *index* into N attractors: it follows
the colour code but cannot restructure. Only cross-target training makes the one-hot
modulate the update rule itself. Loss cannot show this — it is not comparable across
regimes and both models score well — which is why the check is to switch the target
mid-run in the playground.

The frontend's `setCond()` therefore does **not** reset the state: switching the
Target picker converts in place, which is the only way to see this. The picker is a
vector of sliders (-1..1), not a one-hot selector, so out-of-distribution mixtures
(0.5/0.5) can be tried too.

## Wider vs deeper for 8 emoji

Both 8-target, `--mismatch-p 0.3`, same schedule; baseline `cross8.mismatch` is hidden=128,
1 layer, 9360 params.

| run | shape | params | min loss | 8/8 correct | converts | shader |
|---|---|---:|---:|---|---|---|
| `cross8.mismatch` | h128 ×1 | 9360 | 5.8e-3 | yes | yes | 1.8 ms |
| `cross8.mismatch.h=256` | h256 ×1 | 18.7k | 3.8e-3 | yes | yes | 3.3 ms |
| `cross8.mismatch.l=2` | h128 ×2 | 25.9k | **2.7e-3** | yes | yes | 10.2 ms |

All three grow the right emoji for every index and convert 0→1, so neither change
broke conditioning. Depth is the better use of capacity (2.7e-3 vs 3.8e-3 at 38%
more params), and the visual difference is subtle — the loss ranking is the clearer
signal here, but they do have different budgets, so this does not isolate width from
depth.

Shader cost is worth knowing: the `hidden → hidden` layer is ~2.7× the `W1` work and
the rest is register pressure from holding the hidden vector, so `cross8.mismatch.l=2` runs
at ~98 steps/s vs ~620 for a 128-wide single-layer model. Still interactive.

## Known gap

`multi*`/`cross*` exports have no `labels[]`, so the playground's Target picker shows
raw indices. The labels are derivable from `targets` in `frontend/src/models.ts`
(`emoji:1f980` → 🦀) — see `docs/HOUSEKEEPING.md`.
