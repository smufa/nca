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
      - `docs/findings/shader-perf.md` — interleaved A/B tables (naive vs current),
        where the time goes (MLP ~73%, perception ~27%), the rejected variants and
        *why* they lost, and the rule that two shaders must be measured in one
        invocation because the iGPU drifts ±25%.
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
  differing only by `dynamic_shape` — a naming problem, not two runs. (Phase 2
  resolved the naming: `crab.regenerating.ckpt=8000` vs `…8000.static`.)
- **Found while renaming**: re-exporting `crab.regenerating.c=6.h=128` from its
  surviving checkpoint (`runs/cap_c6_h128`) produced a `model.onnx` **41 bytes
  larger** than the committed one (parity unchanged, 3.8e-06). Every other export
  re-exported byte-identically. Root cause, found in Phase 3: the dynamo exporter
  stamps `pkg.torch.onnx.stack_trace` on every node, and it contains
  `nca/export.py` **line numbers** — the file had shifted by three lines. A second
  contributor, `pkg.torch.export.ExportedProgram.range_constraints`, names the
  symbolic dims (`{s0, s10}` vs `{s0, s10, s43, s51}`) and varies by torch
  version. Both are non-semantic, and both would make `export --all` + `git diff`
  (the only possible exporter-drift guard, since CI has no checkpoints) permanently
  noisy. `nca/export.py` now strips them (`--strip-metadata` for existing graphs);
  a fresh re-export of that model is **byte-identical** to the committed one, and
  the files shrank 39% (29 kB -> 18 kB, no more local paths in a public repo).
  `tests/test_export.py` fails if a committed graph carries metadata again.

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
   `crab_regen/*`) — `--all` used to *skip* those. **Fixed:** an export is a
   complete description of its model, and `nca/export/rebuild.py` proves it —
   `ModelSpec.state_from_weights` (the inverse of the blob layout) plus the
   learned kernel from `meta.perception_kernel` rebuild the module with no
   checkpoint at all. `nca.export --all` now falls back to it automatically
   (`--rebuild` forces it), and all 45 exports re-export byte-identically with
   every `source_checkpoint` pointed at a nonexistent path. The run dirs are
   still worth keeping (`--save-pool` resume, `loss.json`), just no longer
   load-bearing.
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
| `smile-demo` | `smile.demo` | | `cross8` | `cross8.mismatch` | | `cross8_wide` | `cross8.mismatch.h=256` |
| | | | `cross8_deep` | `cross8.mismatch.l=2` |

### Mechanics

**As built** (`scripts/rename_exports.py` is the migration and the record of the
old names; it is the only place the legacy grammar survives):

- [x] `scripts/rename_exports.py` writes `exports/<id>/meta.json` first, then
      `git mv`s the dir, then rebuilds `manifest.json` from the metas
      (`export --reindex` — `_update_manifest` appends and sorts, which cannot
      express a rename). It is idempotent and refuses a table that does not cover
      `exports/` exactly.
- [x] `meta.json` gained `name` (= the id) and `sweep` (the study that produced it
      — `emoji-sweep`, `capacity`, `checkpoints`, `experiments`, `multitarget`,
      `crosstarget`, `demo`), plus `train` for the numbers the id encodes but the
      exporter never recorded (`iteration`, `mismatch_p`, each with
      `recovered_from`). `picker: {group, label, order}` is supported and unused so
      far.
- [x] `frontend/src/models.ts`: the 7-pattern regex ladder is gone. Group comes
      from `meta.sweep`, the label from the id segments after the task
      (`c=16.h=8` → "c=16 h=8", `ckpt=2000` → "@2000"), the sort key from a
      numeric parse. ~40 lines instead of ~60, and no model name knowledge.
- [x] `?model=<id>` deep link; a stale id now fails loudly
      (`setStatus("unknown model in ?model=…", 'err')`) instead of showing the
      first model. Only `nca.ep` was ever persisted, so nothing else broke.
- [x] Duplicates: `crab.regenerating` (sweep copy, grid 72) and
      `crab.regenerating.c=16.h=128` (capacity sweep, grid 64) are **not** the same
      model — different canvas, so different exported graph. Both stay, told apart
      by `sweep`; no `duplicate_of` field needed.

Design deviation: the id grammar allows no `.` inside a value, so the cross-target
family is `cross8.mismatch` (task word) and the probability lives in
`train.mismatch_p`. A `mismatch=0.3` segment would split mid-value in every tool
that treats the id as dot-separated (the picker label did exactly that).

**Acceptance:** `ls exports` sorts into readable families; the picker groups
without regexes; old URLs break once, loudly (no silent wrong model).

---

## Phase 3 — Experiments as data

- [x] `nca/experiments.py`: `@dataclass Experiment(name, targets, regime, noise,
      mismatch_p, unroll, lr_milestones, ...)` plus named presets. Kills the
      `preset[...] if args.x is None else args.x` triple in `cli.py:182-187` —
      every default has exactly one home.
- [x] `slurm/sweeps/*.toml`: one file per study (`emoji-sweep`, `capacity`,
      `multi`, `cross`, `grid-scaling`, …), each listing runs with an explicit
      `name:` and `group:`.
- [x] **One** `slurm/submit_sweep.sh <toml>` replacing the four near-identical
      scripts (`submit_sweep.sh`, `submit_capacity_sweep.sh`,
      `submit_multitarget_sweep.sh`, `submit_crosstarget_sweep.sh`), each of which
      re-declares `PARTITION/CONSTRAINT/GRES/TIME/MEM/CPUS` and the same `sbatch`
      line.
- [x] `meta.json` records the `Experiment` (`train:` block) so a model's regime
      is reproducible from the export alone.

**Acceptance:** adding a sweep = adding a TOML file, no shell edits; the four
current studies are reproducible from their TOMLs.

### Phase 3 notes (done)

- **Done.** `nca/experiments.py` (`Experiment`, `REGIMES`, `Sweep`,
  `load_sweep`, `experiment_from_run_dir`), `nca/sweep.py` (`python -m nca.sweep`,
  renders the sbatch calls), six `slurm/sweeps/*.toml`, and one
  `slurm/submit_sweep.sh`; the three per-study shell scripts are deleted.
- The `preset[...] if args.x is None else args.x` ladder is gone: CLI flags default
  to `None` and are applied as overrides, so `Experiment()` *is* the CLI default
  (`tests/test_experiments.py::test_cli_defaults_come_from_the_experiment`).
- Precedence is `--config/--run` (file) < explicit flags < `--set k=v`.
- A run records `experiment_config` in `train.json`; the exporter copies it into
  `meta.json`'s `train` block, so `--experiment`/`--train` became *corrections*,
  not inputs. Old runs without the block are reconstructed from
  `spec.json`+`train.json` (`experiment_from_run_dir`).
- Run names are the export ids (`cap.c=16.h=8` was `cap_c16_h8`), so a run dir,
  its checkpoint and its export share one label.
- `tests/test_experiments.py::test_every_exported_model_has_a_toml_recipe` fails
  if an export with a `sweep` label has no `[[run]]` — the TOMLs cannot silently
  rot behind `exports/`.

### Phase 3 by-products

- **`export --strip-metadata`** (new): drops the exporter's non-semantic metadata
  (`stack_trace` with `nca/export.py` line numbers + the author's home path, and
  `range_constraints` naming symbolic dims that vary by torch version). A
  re-export of an unchanged model is now **byte-identical**, which is what makes
  "`export --all` then `git diff`" usable as the exporter-drift guard. Committed
  graphs shrank 39% (29 kB -> 18 kB) and no longer leak local paths.
- **`--reindex --refresh-targets`** re-renders `target.png` + `target.npy` from
  `target_size`/`padding`, and is a byte-identical no-op on a correct export. It
  also wrote the 45 missing `exports/*/target.npy`, so each export is now
  self-contained. Writing `target.npy` uses the same single-frame framing as
  `export_one`, which is why the 35 already-correct `target.png` did not change.
- **The demo recipe was recovered, not assumed**: `smile.demo`'s meta said
  `target_size=32, padding=0`, but re-rendering that way produced a different
  picture. `prepare_targets(['emoji:😀'], max_size=24, padding=4)` reproduces the
  committed `target.png` byte-for-byte, so the meta was corrected (with a
  `recovered_from` note) and `demo.toml` uses 24/4. A wrong number in a meta is
  invisible until something re-derives an artefact from it.
- Not done, still open: a `grid-scaling` study (the only sweep in the plan with no
  committed runs) and `meta.json` for the two `experiments.toml` runs predating
  `iteration` bookkeeping (`--train iteration=N` fills it in by hand).

---

## Phase 4 — measurement loop — DEFERRED (decided 2026-04-18)

**Not doing it now.** `loss.json`/`meta.train.min_loss` already answers the
quantitative questions the project is currently asking, because the comparisons in
`docs/findings/` are same-regime (same curriculum, same loss definition). Building a
scoring framework ahead of needing it would be a solution looking for a problem.

Two things stay unmeasured, and that is a known, accepted gap:

- `docs/findings/conditioning.md:42`'s `err vs source 24.3 / err vs dest 5.3` has no
  code path — it was computed by hand and 24 of its checkpoints lived in `/tmp`.
- "self-healing" (README) is shown as a GIF (`nca.sample --damage-at`), never scored.

**When it comes back**, the cheap shape is a `--score` flag on `nca/sample.py`
(≈60 lines, which already has the rollout loop, `_damage()` and `_select_cond()`),
printing `{loss, converged_at, alpha_cov, damage:{residual, recovered_at}}` instead
of authoring `nca/eval.py`. Revisit when the first *cross-regime* comparison is
needed (growing vs regenerating is not comparable through `min_loss` today) or when
6b (textures) needs a texture metric.

---

## Phase 5 — Frontend seams

- [x] `frontend/src/main.ts` (983 lines) split behind the seams that already
      exist. **As built:** `main.ts` is 23 lines of wiring; the modules are
      `ui/dom.ts` (one `el()` lookup per element, throws if `index.html` drifts),
      `ui/app.ts` (the shared mutable state + `busy`/`whenIdle` + status/perf),
      `ui/views.ts` (canvas panels, `ChannelMapping`), `ui/channels.ts` (the
      per-view channel selects), `ui/brush.ts` (stroke + tool wiring, GPU brush
      or `edit()`), `ui/kernels.ts` (kernel lab), `ui/cond.ts` (conditioning
      sliders), `ui/picker.ts` (manifest → grouped options → session swap),
      `ui/info.ts` (the model panel), `ui/loop.ts` (frame loop + transport),
      `ui/console.ts` (`window.nca`). Two cycles are load-bearing and safe
      (function-level, never at module init): `views ↔ brush` and
      `picker → views/cond/kernels`.
- [x] Capability negotiation replaces `?.` probing. Today
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

  **As built:** `Capabilities` + `checkSpec()` live in `frontend/src/engine.ts`,
  each engine exports a `CAPABILITIES` object with `supports(meta)` delegating to
  `checkSpec`; `main.ts` branches on `runner.capabilities.{gpuRender,gpuBrush,cond}`
  and shows the *reason* in the status bar ("shader cannot run this model:
  `3*C+cond_n=52 > 48`"), and the picker disables unrunnable models with that
  reason as the tooltip. `setCond`/`setCondVector` became required methods (both
  engines implement them); only `renderTo`/`brush` stay optional, gated by
  capabilities. Shader bounds are now traced to `wgsl/pass1.wgsl`:
  `3C+cond <= 64` (`perc`/`perc4` arrays), `hidden/4 <= 64` (`HG <= HMAX`),
  `layers <= 2` (one `wmid` block in `weights.bin`), and a grid budget of
  128 MiB / (3*C*4) pixels ≈ 836² at C=16.
  Still open: no JS unit test runner (the outcome is covered by `tsc` + manual).
- [x] `nca/export.py` (819 lines) → a package: `graph.py` (162: `ExportStep`,
      `export_step`, the metadata strip), `verify.py` (98: torch-vs-ONNX rollout
      parity), `weights.py` (54: `weights.bin`), `manifest.py` (187: targets,
      `--reindex`), `cli.py` (379: `export_one`/`export_all`/argparse),
      `__init__.py` re-exporting the old surface. Proof it is a move and not a
      rewrite: re-exporting `crab.regenerating` from its checkpoint produces a
      **byte-identical** `model.onnx` and `weights.bin`. Tests that monkeypatched
      `nca.export.export_one` now patch `nca.export.cli.export_one` (the CLI
      resolves it in its own module).
- [x] `nca/step.py`: torch-free canonical primitives (`ident/dx/dy`,
      `perception_weight`, `make_seed`, `map_channels`/`ChannelMapping`) imported
      by `nca/model.py`, `nca/targets.py`, `nca/viz.py` and
      `scripts/shader_bench.py` (which used to re-typye the kernels in numpy);
      `viz.to_rgb`/`render` now take a `ChannelMapping`, so the Python gif path can
      draw hidden-channel views like the frontend does. `tests/test_step.py`
      checks the torch path, the seed, the TS mirror *textually*, and that the
      module never imports torch. Rounding: `map_channels` rounds (the TS side
      writes a `Uint8ClampedArray`), which is 1 LSB different from the old
      truncating `np.uint8(arr * 255)`.

---

## Phase 6 — The three next experiments (features, not housekeeping)

Each entry lists only what the abstractions above must provide.

### 6a. Bigger grids → shader headroom

*Done 2026-02, with a negative result on the part that looked most promising.*

Measured state (interleaved, `docs/findings/shader-perf.md`): 68 ms/step at 512²,
281 ms at 1024² for `crab.regenerating` — 4 steps/s and 3.5 steps/s, i.e. a 512²
texture is not interactive and 1024² is not usable. **The prescribed fix in the
first draft of this plan (tile cells + stage the weights in shared memory) was
wrong**: implemented and measured, it is 2.6× *slower*, because the weights are
L1-resident and the barriers cost more than the traffic they save. What did work
was the entry-point split + explicit ILP (1.35×).

- **Done:** `--grids a,b,c` + `--json` + `--dump` in `scripts/shader_bench.py`, and
  `--shader` accepting several files benchmarked round-robin (the ±25% drift fix).
- **Done:** `step`/`step_deep` split; deep models no longer run at 2.2× the
  shallow cost.
- **Done:** the size constants stay compile-time overrides; `--reference` parity is
  < 3e-07 on all four branch-covering models.
- **Done and rejected: the packed state layout** (`[H,W,C/4]` vec4 texels).
  `scripts/variants/pass1_packed.wgsl` + `pass2_packed.wgsl` are correct
  (2.4e-07) and ~2% faster at 256² (15.49 vs 15.77 ms/step) — perception is only
  27% of the step and already L1-served, so the engine/render/brush churn an
  AoS↔SoA change needs buys nothing. The variants stay as the record.
- **Not started, and now the interesting lever:** a GEMM-over-cells formulation
  (give each thread several cells, `W1` as the shared operand) — the MLP is
  latency-bound, so only changing *how* it is evaluated moves the number.
- **Done and shipped: work-skipping.** At `fire_rate 0.5` the update rule ran for
  cells whose output was then multiplied by zero. `compact` + `finalize` +
  `step_list` (indirect dispatch over a work list) is 1.3× at 72², 1.6× at 256²,
  1.65× at 512² and **bit-identical** to the plain path, so `ShaderRunner` now runs
  it for single-hidden-layer models (deep models keep `pass1`). Measurement and
  the per-grid table: `docs/findings/shader-perf.md`.
- **Done and rejected: fp16** (1.42× *slower* — WebGPU's fp16 is storage-only on
  Gen9, so every op gains conversions) and **int8** (no packed 4×8-bit dot in
  WebGPU, and 0.88 drift / 27 dB quality). Both recorded with numbers, so a future
  GPU can re-measure instead of re-deriving.
- `supports()` still needs a real `maxGridPx` (3 state buffers × `C·H·W·4` bytes:
  200 MB at 1024²/C=16, above the 128 MiB default binding size).
- No re-export needed (the ONNX graph is shape-agnostic).

**Method warning, learned the hard way:** the iGPU's clock drifts ±25% between
sessions, so a number measured in one invocation cannot be compared with one
measured in another. Always `--shader old.wgsl new.wgsl` in a single run. Every
absolute number in the first draft of `shader-perf.md` was collected that way and
had to be re-measured.

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

- [x] `AGENTS.md` ≤ 150 lines, orientation only; every measurement lives in
      `docs/findings/` and is regenerable by a command (`scripts/bench.sh`).
- [x] `PLAN.md` no longer describes removed engines.
- [x] No `_infer_dims`; `nca.sample` works on all 45 exports.
- [x] One naming convention, enforced by a test; the picker groups from
      `meta.sweep`, not from regexes over ids.
- [x] One `slurm/submit_sweep.sh` + TOML sweeps.
- [x] `pytest` + `tsc --noEmit` + `vite build` in CI; `verify.sh` stays as the
      GPU-inclusive check. (The `nca.ts` drift check is a *pytest* reading the `.ts`
      source, not `node --test`: it needs no TS build step, so CI cannot skip it.)
- [x] `scripts/check_meta.py` validates all exports; `export --all` reproduces
      parity (byte-identical since `--strip-metadata`).
- [x] `export --reindex` can rebuild `manifest.json` from the metas.
- [x] `nca/step.py` and `nca/spec.py` exist; `nca.ts` mirrors them with a test
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
