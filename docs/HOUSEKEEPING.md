# Housekeeping plan

Working plan from the repo review (2026-10). Goal: keep the existing model
quality and the parity/testing discipline, and remove the friction that will
otherwise dominate the *next* experiments (bigger grids / tiled GEMM, textures,
non-uniform conditioning).

Nothing here changes tensor math. **Phases 0–4 must leave the graph untouched**,
so the 45 committed exports and their recorded `parity_max_abs_diff` stay valid
and no re-export is required. Only Phase 6 (features) changes the graph.

## Decisions taken

| # | Question | Decision |
|---|---|---|
| 1 | Keep the WASM/ORT path? | **Yes.** It is the real fallback (WebGPU is not universal), so both engines must advertise capabilities; the shader engine is not the only citizen. |
| 2 | `exports/` archive or gallery? | **Keep all models in `exports/`**, but migrate to **one naming convention** (Phase 2). |
| 3 | Next experiments | **(a) bigger grids → tiled GEMM**, then **(b) textures**, then **(c) conditioning that is not uniform across the canvas**. |
| 4 | Where do findings live? | **`AGENTS.md` becomes orientation only** (invariants, commands, gotchas, workflows). All measurements move to `docs/findings/`. |

## Ground rules

- `bash scripts/verify.sh` stays green after every phase.
- No `git mv` of `exports/` until Phase 1 (ModelSpec) exists — regenerating 45
  metas by hand is how drift happens.
- Anything derived from a checkpoint must be **read** (from `spec.json` / meta),
  never **inferred** from tensor shapes. `nca/sample.py:26` is the cautionary tale.
- One fact, one home. If a number appears in a doc and in code, the code (or its
  generator) is the source.
- Do not reintroduce: the ORT WebGPU EP, dynamic `C/NP4/HG` loop bounds (~1.9×),
  or `cond` as state channels (see Phase 6c).

---

## Phase 0 — Truth, docs, and the safety net

*No behaviour change; this is what makes the rest safe.*

- [x] `exports/meta.schema.json` + `scripts/check_meta.py`: validate every
      `exports/*/meta.json`. Catches the `wmid`/`p4`/`offsets` class of breakage
      (see "known drift" below) and is the cheapest possible CI gate.
      *Done: 45/45 clean, `--fix` repaired the 19 broken ones; also gained
      `--target/--experiment` cross-checks against the checkpoint's `train.json`.*
- [x] Split `AGENTS.md` (336 lines) → orientation only. *Done: 159 lines, with
      `docs/findings/{shader-perf,training-perf,capacity,robustness,conditioning,
      frontend-engines}.md` + index; every number moved out.*
      Move the results tables out:
      - `docs/findings/shader-perf.md` — MLP is 98% of the step, weight-load
        bound, vec4 3.2–3.6×, the rejected experiments (`perc` shared mem, ILP,
        wg16, tile2 spill), current 1.54 / 4.40 / 16.8 ms at 72²/128²/256².
      - `docs/findings/training-perf.md` — throughput-bound table, bf16 1.3–1.55×,
        `torch.compile` no help, memory ~ linear in `batch × train_steps`.
      - `docs/findings/capacity.md` — the 8-row hidden/channel table.
      - `docs/findings/robustness.md` — kernel-lab OOD failure + state noise.
      - `docs/findings/conditioning.md` — `multi*` vs `cross*`, conversion table,
        wide-vs-deep.
      - `docs/findings/README.md` — index, one line each, with the *policy*:
        findings are append-only, numbers must be regenerable by a script.
- [x] Rewrite or delete `PLAN.md` (179 lines). *Done: roadmap + the decisions worth
      re-deriving; the removed engines and NHWC/JSEP notes are gone.*
      ORT WebGPU EP, NHWC layout transforms, JSEP pinning and a "WebGPU-unavailable
      gate". Keep only what is still true: the ONNX interface, the dynamic-shape
      finding, the export pipeline decisions.
- [x] Fix stale comments: `slurm/submit_capacity_sweep.sh:11` claims the shader
      engine needs `C=16, hidden=128` (false since `e7f5a6c`). *Done.*
- [x] `frontend/tsconfig.json`: add `"strict": true` and fix the fallout.
      *Done: zero fallout — `tsc --noEmit` was already clean.*
- [x] Delete the dead `--compile` branch (`nca/train.py:176-197`, duplicated
      `forward()`). *Deleted (`cfg.compile`, the CLI flag, one `forward`); the comment
      now cites the measurement and `scripts/profile_train.py --compile` as the way to
      revisit it. Nothing in `slurm/` used it.*
- [x] Fix `nca/sample.py` (broken for 3 of 4 model families — see Phase 1) or
      mark it explicitly deprecated in favour of `nca.export --verify` + the bench.
      *Done in Phase 1: `--run <dir>` reads `spec.json`, `--cond IDX` selects a
      conditional target, `_infer_dims` is gone.*
- [x] CI: `.github/workflows/ci.yml` — `pytest` + `check_meta` on Python 3.13/3.14,
      `tsc --noEmit` + `vite build` on Node 22. (No `node --test` yet: there is no JS
      test file; that lands with the Phase 6 coverage maths.)
- [x] Resolve the Python version pin. *Done differently than proposed: 3.14 is the
      laptop (`.python-version`) and 3.13 the cluster, so `requires-python =
      ">=3.13"` with a CI matrix over both is the honest pin — a single 3.13 would
      misdescribe the laptop. `uv.lock` updated to match (run `uv sync` once).*

**Acceptance:** `python scripts/check_meta.py` passes on all 45 models, CI green,
`AGENTS.md` ≤ 150 lines, every number in the old AGENTS.md findable in
`docs/findings/`.

### Known drift to fix while here

- ~~`crab_c16_h128/meta.json` says `grid: 64`; every other capacity model says 72.~~
  **Not drift**: the capacity sweep really was trained and exported at 64² (its
  `target.png` is 64×64); all 45 `grid` values match their `target.png`. The real
  wart is the *name* — `grid` means "dummy export shape", not "the model's grid",
  and with `dynamic_shape: true` the UI should not read it at all. Rename to
  `export_grid` in Phase 3.
- `crab_noisy/meta.json` records `experiment: "noisy"` but **not** `noise_n` /
  `noise_std`, so the regime cannot be reproduced from the export (the run dir is
  gone). `check_meta.py` cannot catch it: it compares against `spec.json`/
  `train.json`, and there is none.
- `crab_c16_h128` and `crab_regenerating` are the same configuration (8336 params,
  same experiment); one is a leftover of the capacity sweep.
- `crab_regen_8000` vs `crab_regen_8000_static` are the **same checkpoint**,
  differing only by `dynamic_shape` — a naming problem, not two runs.

---

## Phase 1 — `nca/spec.py`: one `ModelSpec`

The single source of truth for a model's shape, replacing four hand-synchronised
copies (`model.py` ctor, `export.py` inference, `models.ts` types, `shader.ts`
overrides).

```python
@dataclass(frozen=True)
class ModelSpec:
    channel_n: int; hidden_n: int; hidden_layers: int = 1
    cond_n: int = 0; learn_perception: bool = False
    wrap: bool = True; fire_rate: float = 0.5
    # derived
    perception_n: int          # 3*C + cond_n
    perception_p4: int         # ceil(perception_n / 4) * 4
    hidden_groups: int         # hidden_n // 4
    def weight_offsets(self) -> dict[str, int]   # w1,b1,[wmid,bmid,mid_stride],w2,b2
    @classmethod
    def from_state_dict(cls, sd: Mapping[str, Tensor]) -> "ModelSpec"
    @classmethod
    def from_run_dir(cls, path: Path) -> "ModelSpec"   # reads spec.json, no inference
```

- [x] `nca/spec.py` + `NeuralCA.from_spec(spec)`; keep the existing ctor as sugar.
      *Done: `spec.py` is torch-free; the ctor and `from_spec` are the same code
      path (`ModelSpec.build()`/`.build_class()` share one `__init__` call). Also
      added `weights_meta()` (the `meta.json "weights"` block) and
      `save_run_manifest()` (the shared `spec.json`/`train.json` writer).*
- [x] `export.py`: `_load_ca`, `infer_dims`, `_last_layer_index`, `_hidden_layers`,
      `_save_weights` all take/publish a `ModelSpec`. Offsets are computed **once**.
      *Done: `_hidden_layers` deleted, `infer_dims` → `ModelSpec.from_state_dict`;
      the checkpoint's `ca.spec` wins over flags, flags only fill the gaps (this is
      how the 45 metas stay byte-identical).*
- [x] `nca/sample.py` uses `from_run_dir`/`from_state_dict` and passes `cond`
      (it currently constructs `NeuralCA` from `in_channels // 3`, which is wrong
      for `3C + cond_n`, and never passes `cond` at all). Delete `_infer_dims`.
      *Done: `--run` reads `spec.json`/`meta.json` (no `--checkpoint` needed), and
      `--channel-n/--hidden-n/--learn-perception` now default to `None` so the spec
      is used unless overridden. `--cond IDX` / `--cond-scale` for multi-target
      models.*
- [x] `runs/<name>/spec.json` written by `train.py`; `export.py` copies it into
      `meta.json` untouched. Then `nca.sample`/`nca.export` need only a run dir.
      *Done: `train.py` writes `spec.json` + `train.json` (final resolved config +
      provenance), `loss.json`, viz files. `meta.json` records `spec_id` +
      `source_run`.*
- [x] `python -m nca.export --all`: walk `exports/*/meta.json`, re-export,
      re-verify, print a parity table. This is AGENTS.md's "re-export all models
      if the graph changed" rule, automated — needed before Phase 6.
      *Done: `--all` (skips models whose checkpoint is gone, prints a skip table)
      and `--reindex` (rebuild `manifest.json` from `meta.json`, which is now the
      source of truth).*
- [ ] Tests (see below) `test_spec_roundtrip`, `test_weight_offsets`,
      `test_export_parity_small`.

**Acceptance:** no `_infer_dims` anywhere; `nca.sample` renders all 45 exports
(and fails cleanly on a cond model without `--cond`); `export --all` reproduces
the committed parity values on a no-op run.

### Phase 1 as-built (what actually happened)

Landed: `nca/spec.py`, `spec.json`/`train.json` from `train.py`, `export.py`
rewired (`--all`, `--reindex`, spec-derived `meta.json`), `sample.py` rewritten
(`--run`, `--cond`), `tests/` (329 tests, ~5 s, no GPU).

Verification that it is a true no-op: all 45 committed `meta.json` files are
byte-identical to what the new `export.py` produces, `_save_weights` reproduces
every committed `weights.bin` **byte for byte** (this is now
`tests/test_export.py`, but it needs the source checkpoint -- which is gitignored --
so it skips in a clean clone), and `nca.export --reindex` reproduces the committed
`manifest.json`.

Flag parity is *not* one-to-one: `nca/cli.py` dropped `--compile` (it measured
neutral-to-slower, see `docs/findings/training-perf.md`) and `nca/export.py` gained
`--all/--only/--reindex/--target-size/--padding`. The sets are meant to differ:
training flags configure a run, export flags configure a re-export.

Three bugs/pitfalls found on the way — all fixed or noted:

1. **`nca/train.py` crashed for every single-target run with `batch_size > 1`**
   (`IndexError: index 1 is out of bounds for dimension 0 with size 1`): with no
   `cond`, `tgt` stayed `[1, 4, H, W]` and the ranking line `tgt[rank]` indexed it
   per sample. Now expanded to `[B, 4, H, W]`. `tests/test_train.py` covers it.
   Stayed invisible because every recent run was either conditional or batch 1.
2. **`export_batch`/variants metas keep only `experiment`**, not the full
   provenance, and 24 of the 45 checkpoints live in `/tmp` (`sweep_runs/*`,
   `crab_regen/*`) — `--all` skips those by design, but they should be moved into
   the repo (or `runs/`) before `/tmp` is wiped. `meta.json` records
   `source_checkpoint`, and the run directory beside it is self-describing
   (`spec.json` + `train.json`), so a re-export is reproducible from the run dir
   alone -- as long as the run dir survives.
3. **Resuming does not restore the sample pool unless `--save-pool` was used**
   (`nca/train.py:130-136`: the pool is ~340 MB at the paper size, so keeping it
   is opt-in and resume prints "no pool in resume state; restarting the pool").
   Documented behaviour, not a bug — but `--save-pool` is not printed at the end
   of a run, so it is easy to lose.

Also: `--verify` never existed (the flag is `--verify-steps N`; parity is on by
default). `README.md` and `AGENTS.md` both documented `--verify` — fixed.

### Tests (all CPU, no GPU)

1. `tests/test_spec.py` — `from_state_dict` round-trips the 4 families
   (plain / `learn_perception` / conditional / `hidden_layers=2`);
   `weight_offsets()["count"] == len(weights.bin)` for `cross8_deep` (25872).
   *Landed, parametrised over all 45 exports (skip when the checkpoint is gone).*
2. `tests/test_export.py` — `_save_weights` reproduces every committed
   `weights.bin` byte for byte; `reindex` rebuilds the manifest from metas
   (idempotent); `export_all` skips missing checkpoints and still reindexes.
   *Landed.*
3. `tests/test_train.py` — 2-iteration smoke runs on a 16×16 grid: single
   target (the batch>1 regression above), conditional + `mismatch_p`, resume; plus
   `spec.json`/`train.json` round-trip through `ModelSpec.from_run_dir`. *Landed.*
4. `tests/test_meta_schema.py` — every `exports/*/meta.json` validates (Phase 0
   script, promoted to a test). *Still Phase 0.*
5. `tests/test_kernels.py` — the python canonical kernels equal the literals in
   `frontend/src/nca.ts` (`CANONICAL_KERNELS`, regex-parsed) and the WGSL defaults.
   *Still Phase 2.*
6. `tests/test_export_parity.py` — smallest model (C=6, h=8): export → ORT CPU →
   1-step diff < 1e-5. *Still Phase 0 (needs `onnxruntime` in the CPU dev env).*
7. `frontend/tests/nca.test.ts` (`node --test`, no new deps) — `seedState`,
   `mapChannels`, `rotateKernels`, `modelKernels`. *Still Phase 2.*

---

## Phase 2 — Naming convention, and a picker that doesn't parse ids

### The grammar

```
<subject>.<task>[.<key>=<value>]*[.<flag>]*
```

lowercase ASCII, dot-separated, no spaces. `subject` = one emoji name, or a
conditional set (`multiN` / `crossN`). `task` = `growing | persistent |
regenerating | matched | mismatch=… | demo`. Knob keys: `c` channels, `h` hidden,
`l` hidden layers, `ckpt` training iteration, `noise`, `perc`, `static`.

Rules: the id is for humans (sortable, scannable in `ls` and in `squeue`);
**every number in the id must also exist in `meta.json`, and meta is
authoritative.** The full training config lives in meta, not in the id.

### Migration table (45 models)

| current | new | | current | new |
|---|---|---|---|---|
| `cactus_growing` | `cactus.growing` | | `crab_c6_h128` | `crab.regenerating.c=6.h=128` |
| `cactus_persistent` | `cactus.persistent` | | `crab_c8_h128` | `crab.regenerating.c=8.h=128` |
| `cactus_regenerating` | `cactus.regenerating` | | `crab_c10_h128` | `crab.regenerating.c=10.h=128` |
| `crab_growing` | `crab.growing` | | `crab_c12_h128` | `crab.regenerating.c=12.h=128` |
| `crab_persistent` | `crab.persistent` | | `crab_c16_h8` | `crab.regenerating.c=16.h=8` |
| `crab_regenerating` | `crab.regenerating` | | `crab_c16_h16` | `crab.regenerating.c=16.h=16` |
| `mushroom_growing` | `mushroom.growing` | | `crab_c16_h32` | `crab.regenerating.c=16.h=32` |
| `mushroom_persistent` | `mushroom.persistent` | | `crab_c16_h64` | `crab.regenerating.c=16.h=64` |
| `mushroom_regenerating` | `mushroom.regenerating` | | `crab_c16_h128` | `crab.regenerating.c=16.h=128` |
| `octopus_growing` | `octopus.growing` | | `crab_regen_2000` | `crab.regenerating.ckpt=2000` |
| `octopus_persistent` | `octopus.persistent` | | `crab_regen_4000` | `crab.regenerating.ckpt=4000` |
| `octopus_regenerating` | `octopus.regenerating` | | `crab_regen_6000` | `crab.regenerating.ckpt=6000` |
| `trex_growing` | `trex.growing` | | `crab_regen_8000` | `crab.regenerating.ckpt=8000` |
| `trex_persistent` | `trex.persistent` | | `crab_regen_8000_static` | `crab.regenerating.ckpt=8000.static` |
| `trex_regenerating` | `trex.regenerating` | | `crab_learnperc` | `crab.regenerating.perc` |
| `turtle_growing` | `turtle.growing` | | `crab_noisy` | `crab.regenerating.noise` |
| `turtle_persistent` | `turtle.persistent` | | `multi2`…`multi8` | `multiN.matched` |
| `turtle_regenerating` | `turtle.regenerating` | | `cross2`…`cross8` | `crossN.mismatch=0.3` |
| `smile-demo` | `smile.demo` | | `cross8_wide` | `cross8.mismatch=0.3.h=256` |
| | | | `cross8_deep` | `cross8.mismatch=0.3.l=2` |

### Mechanics

- [ ] Write `exports/<id>/meta.json` first, then `git mv` the dir, then regenerate
      `manifest.json` from the metas (`export --reindex`, not incremental append —
      `_update_manifest` currently appends and sorts, which cannot express renames).
- [ ] `meta.json` gains: `name` (= the id), `sweep` (which study produced it),
      `train` (the full `Experiment`), and an optional `picker: {group, label,
      order}` override.
- [ ] Grouping in `frontend/src/models.ts`: **delete the 7-pattern regex ladder**
      in `modelGroup`. Group comes from `meta.sweep` (written by the sweep
      manifest in Phase 3); the label is the id segments after the task
      (`c=16.h=8` → "c=16 h=8"). ~10 lines instead of ~60.
- [ ] Add `?model=<id>` deep link (no model id is persisted today — only
      `nca.ep` in localStorage — so the rename is zero-risk for saved state).
- [ ] Decide the fate of the two exact duplicates (`crab_c16_h128` vs
      `crab.regenerating`): drop the sweep copy, or keep it with `sweep: capacity`
      and a `duplicate_of` field.

**Acceptance:** `ls exports` sorts into readable families; the picker groups
without regexes; old URLs break once, loudly (no silent wrong model).

---

## Phase 3 — Experiments as data

- [ ] `nca/experiments.py`: `@dataclass Experiment(name, targets, regime, noise,
      mismatch_p, unroll, lr_milestones, ...)` plus named presets. Kills the
      `preset[...] if args.x is None else args.x` triple in `cli.py:182-187` —
      every default has exactly one home.
- [ ] `slurm/sweeps/*.toml`: one file per study (`emoji-sweep`, `capacity`,
      `multi`, `cross`, `grid-scaling`, …), each listing runs with an explicit
      `name:` and `group:`.
- [ ] **One** `slurm/submit_sweep.sh <toml>` replacing the four near-identical
      scripts (`submit_sweep.sh`, `submit_capacity_sweep.sh`,
      `submit_multitarget_sweep.sh`, `submit_crosstarget_sweep.sh`), each of which
      re-declares `PARTITION/CONSTRAINT/GRES/TIME/MEM/CPUS` and the same `sbatch`
      line.
- [ ] `meta.json` records the `Experiment` (`train:` block) so a model's regime
      is reproducible from the export alone.

**Acceptance:** adding a sweep = adding a TOML file, no shell edits; the four
current studies are reproducible from their TOMLs.

---

## Phase 4 — `nca/eval.py`: the missing measurement loop

The repo's best claims (conversion `err vs dest 5.3`, capacity table, robustness
under noise) were computed ad hoc and cannot be regenerated. That is the biggest
functional gap for a project whose value is quantitative.

- [ ] `score(checkpoint_or_spec, targets, steps, seeds) -> {per_target_err,
      converged_at, min_loss, final_alpha_coverage}` — CPU-only.
- [ ] `score --sweep <dir>` → a markdown table, so `docs/findings/*` tables are
      generated, not transcribed.
- [ ] `eval --cond-switch a b` for the `multi*` vs `cross*` conversion metric.

**Acceptance:** every table in `docs/findings/conditioning.md` and
`docs/findings/capacity.md` is reproducible by one command.

---

## Phase 5 — Frontend seams

- [ ] `frontend/src/main.ts` (983 lines) split behind the seams that already
      exist: `ui/views.ts` (`View`, `ChannelMapping`, canvas draw), `ui/kernelLab.ts`,
      `ui/condPanel.ts`, `ui/brush.ts`, `ui/perf.ts`, and `session.ts` (owns
      `runner`, `selectToken`, `whenIdle`, `busy`). `main.ts` becomes wiring.
- [ ] Capability negotiation replaces `?.` probing. Today
      `ShaderRunner.supports()` returns a bare bool and `main.ts` silently falls
      back to WASM; `setCond?`, `renderTo?`, `brush?` are optional methods the
      caller feature-detects ad hoc. Replace with:

```ts
interface Capabilities {
  kind: 'ort' | 'shader';
  cond: boolean; gpuBrush: boolean; gpuRender: boolean;
  maxC: number; maxHidden: number; maxLayers: number; maxCond: number; maxGridPx: number;
}
supports(spec: ModelSpec): { ok: true } | { ok: false; reason: string }
```

  so the Engine/model picker can say *why* a model runs on WASM ("hidden=256 >
  shader limit"), and a third backend can be added without touching `main.ts`.
  **Both engines stay first-class** (decision 1): `ort` advertises
  `maxC/maxHidden = ∞` and `gpuRender: false`.
- [ ] `nca/export.py` (506 lines) → `nca/export/{graph,weights,verify,manifest}.py`.
- [ ] `nca/step.py`: torch-free canonical primitives (`ident/dx/dy`,
      `perception_weight`, `make_seed`, `render`) importable by
      `scripts/shader_bench.py` (which currently reimplements them in numpy at
      `shader_bench.py:32-52`) and mirrored by `nca.ts`. Plus `viz.py`'s
      `to_rgb` generalised onto `ChannelMapping` (it is hardcoded to channels 0-3
      over white, while `nca.ts:mapChannels` and `draw.wgsl` are general).

---

## Phase 6 — The three next experiments (features, not housekeeping)

Each entry lists only what the abstractions above must provide.

### 6a. Bigger grids → tiled GEMM in WGSL *(do first)*

At 512² the current kernel is ~67 ms/step (16.8 ms at 256², quadratic) ≈ 15
steps/s. The fix is AGENTS.md's remaining idea #1: tile cells into workgroup
shared memory and stream the weights once per tile.

- The tile shape must be derived from the spec (`C`, `NP4`, `HG`, `CONDN` stay
  **compile-time overrides** — dynamic bounds cost ~1.9×).
- `supports()` needs a `maxGridPx` based on real limits: the shader holds 3 state
  buffers of `C·H·W·4` bytes (at 1024², C=16: 67 MB each → ~200 MB, above the
  default 128 MiB `maxStorageBufferBindingSize`).
- `meta.grid` must stop being read as "the model's grid" (see Phase 0 drift);
  the grid becomes a UI/capability concern.
- `scripts/shader_bench.py` gains a `--grids 72,128,256,512,1024` sweep with JSON
  output so `docs/findings/shader-perf.md` is generated.
- Variants stay in `scripts/variants/`; promote to `pass1.wgsl` only with
  1-step `--reference` parity < 1e-5 plus a full size table.
- No re-export needed (the ONNX graph is shape-agnostic).

### 6b. Textures

- `nca/targets.py` gains `texture:<path>` and richer `procedural:` specs;
  `prepare_target`'s `padding` becomes opt-in (a texture wants the whole canvas).
- New regime rather than new flags: full-canvas target, no padding, longer
  unroll, possibly multiple seeds. This is why Phase 3 (experiments as data)
  comes first.
- `viz.py` needs a colour-correct render path (and `eval.py` a texture metric).
- Depends on 6a: a 512² texture is only interactive on the shader path.

### 6c. Non-uniform conditioning (a condition *field*, not a vector)

Today `cond` is `[B, cond_n]`, broadcast in-graph to `[B, cond_n, H, W]`, and a
small uniform buffer in WGSL. Making it spatial lets one CA compose a scene from
a layout map, or blend targets per region.

Invariants that must survive:

1. **`cond` must not become state channels.** `x = x * life_mask` would zero it
   in dead cells, and the exported graph *is* the step, so the frontend could not
   re-clamp it. Keep it a separate input.
2. Add `cond_kind: "vector" | "field"` to `ModelSpec`. Vector models keep the
   `[B, cond_n, 1, 1]` ONNX input (existing exports stay valid, **no re-export**);
   field models take `[B, cond_n, H, W]` in the *same* input slot. No optional
   inputs, no ambiguity — engine and UI dispatch on `cond_kind`.
3. Life mask must stay cond-independent (it is today).
4. Shader: the `cond` uniform becomes a `storage` buffer (one more binding);
   `CONDN` stays an override.
5. Training: a cond-field generator (region one-hots, Voronoi/blurred maps) and a
   field analogue of `--mismatch-p` (permute regions between targets). The pool
   then carries `(state, cond_field)`; for `cond_n=8, C=16` that is ×1.5 pool
   memory (340 MB → ~500 MB, relevant for `--save-pool` on a MIG slice).
6. Frontend: a **condition painter** — the brush writes the cond buffer instead
   of state channels (`brushTargets` generalises to "which buffer + which
   channels"). This is the natural exploration UI, and it is the same shape of
   change as the kernel lab: a probe that will find the failure modes.
7. `eval.py`: per-region error, boundary sharpness, bleed between regions.
8. This one **is** a graph change → run `export --all` and re-verify (Phase 1).

---

## Definition of done (housekeeping)

- [ ] `AGENTS.md` ≤ 150 lines, orientation only; every measurement lives in
      `docs/findings/` and is regenerable by a command.
- [ ] `PLAN.md` no longer describes removed engines.
- [ ] No `_infer_dims`; `nca.sample` works on all 45 exports.
- [ ] One naming convention, enforced by a test; the picker groups from
      `meta.sweep`, not from regexes over ids.
- [ ] One `slurm/submit_sweep.sh` + TOML sweeps.
- [ ] `pytest` (5 files) + `node --test` + `tsc --noEmit` in CI; `verify.sh` stays
      as the GPU-inclusive check.
- [ ] `meta.schema.json` validates all exports; `export --all` reproduces parity.
- [ ] `export --reindex` can rebuild `manifest.json` from the metas.
- [ ] `nca/step.py` and `nca/spec.py` exist; `nca.ts` mirrors them with a test
      that fails on drift.

## Order and risk

| Phase | Risk | Why here |
|---|---|---|
| 0 | none (docs + validator) | catches every later drift |
| 1 | low — touches no math | unblocks 2, 3, 4, 6 |
| 2 | low (metas only, no ONNX/torch change) | needs 1 to regenerate metas |
| 3 | low (shell + config) | needs 2 for `sweep`/`name` fields |
| 4 | low (new code only) | needs 1 to score any checkpoint |
| 5 | medium (frontend refactor) | needs 1 for the capability spec |
| 6a/6b/6c | feature work | need 1 + 5, 3, and 1 + 4 respectively |

## Open questions

- Are `crab_c16_h128` / `crab_regenerating` (identical) and
  `crab_regen_8000` / `crab_regen_8000_static` (same checkpoint, static-shape
  export) both worth keeping in `exports/`?
- Does the 512²+ future keep WASM at all? It technically can (ORT CPU), but a
  large-grid model on WASM will be seconds/step — is the fallback "same model,
  visibly slow" or "small-grid models only"? This decides whether `maxGridPx` is
  a shader-only limit or a global one.
- Textures: one emoji-shaped target padded on a large canvas, or full-canvas
  patterns with a different living-mask story? The latter may want the alpha
  channel to stop meaning "life".
