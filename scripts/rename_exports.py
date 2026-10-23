"""One-time migration to the dotted model-id grammar (Phase 2 of docs/HOUSEKEEPING.md).

The id grammar is ``<subject>.<task>[.<key>=<value>]*[.<flag>]*`` -- lowercase,
dot-separated, sortable in ``ls`` and readable in ``squeue``. Numbers in an id are
*duplicated* there for humans; ``meta.json`` stays authoritative and gains
``name`` + ``sweep``, which is what lets the frontend group models without parsing
ids (the regex ladder in ``frontend/src/models.ts`` is deleted in the same commit).

    uv run python -m scripts.rename_exports --dry-run
    uv run python -m scripts.rename_exports --apply

The table below is the *only* place the legacy names survive. ``--apply`` rewrites
each ``meta.json`` first, then ``git mv``s the directory, then rebuilds
``manifest.json`` from the metas (``export --reindex``; the append-only
``_update_manifest`` cannot express a rename). Re-runnable: an already migrated
directory is skipped, a half-done one is reported.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"

# old id -> (new id, sweep)
#
# `sweep` names the study that produced the model. It is written into meta.json and
# is the grouping key for the picker; the display names live in the frontend.
TABLE: dict[str, tuple[str, str]] = {
    # Emoji regime sweep: 6 subjects x {growing, persistent, regenerating}.
    **{
        f"{subject}_{task}": (f"{subject}.{task}", "emoji-sweep")
        for subject in ("cactus", "crab", "mushroom", "octopus", "trex", "turtle")
        for task in ("growing", "persistent", "regenerating")
    },
    # Capacity sweep (channels / hidden width), grid 64.
    **{
        f"crab_c{c}_h{h}": (f"crab.regenerating.c={c}.h={h}", "capacity")
        for c, h in ((6, 128), (8, 128), (10, 128), (12, 128), (16, 8), (16, 16), (16, 32), (16, 64), (16, 128))
    },
    # Training-time checkpoints of one run (`ckpt_00N000.pt`), plus a static-shape copy.
    "crab_regen_2000": ("crab.regenerating.ckpt=2000", "checkpoints"),
    "crab_regen_4000": ("crab.regenerating.ckpt=4000", "checkpoints"),
    "crab_regen_6000": ("crab.regenerating.ckpt=6000", "checkpoints"),
    "crab_regen_8000": ("crab.regenerating.ckpt=8000", "checkpoints"),
    "crab_regen_8000_static": ("crab.regenerating.ckpt=8000.static", "checkpoints"),
    # Single-target experiments: a learned kernel, and the noisy regime.
    "crab_learnperc": ("crab.regenerating.perc", "experiments"),
    "crab_noisy": ("crab.regenerating.noise", "experiments"),
    # Conditional sweeps: matched (state, cond) pairs vs cross-target mismatches.
    "multi2": ("multi2.matched", "multitarget"),
    "multi4": ("multi4.matched", "multitarget"),
    "multi6": ("multi6.matched", "multitarget"),
    "multi8": ("multi8.matched", "multitarget"),
    # The mismatch *probability* is a hyperparameter, so it lives in meta.train and not
    # in the id: ids are dot-separated, and ``mismatch=0.3`` would split mid-value.
    "cross2": ("cross2.mismatch", "crosstarget"),
    "cross4": ("cross4.mismatch", "crosstarget"),
    "cross6": ("cross6.mismatch", "crosstarget"),
    "cross8": ("cross8.mismatch", "crosstarget"),
    "cross8_wide": ("cross8.mismatch.h=256", "crosstarget"),
    "cross8_deep": ("cross8.mismatch.l=2", "crosstarget"),
    "smile-demo": ("smile.demo", "demo"),
}

# Numbers that the new ids encode and that meta.json must therefore also carry.
# Both are *recovered* facts, not things the exporter recorded at the time, hence
# `recovered_from`: the id is a human-readable copy of meta, never the other way.
RECOVERED: dict[str, dict] = {
    **{
        old: {
            "mismatch_p": 0.3,
                            "recovered_from": "slurm/submit_crosstarget_sweep.sh (MISMATCH_P default; "
                            "the study now lives in slurm/sweeps/cross.toml)",
        }
        for old in ("cross2", "cross4", "cross6", "cross8", "cross8_wide", "cross8_deep")
    },
    **{
        old: {"iteration": n, "recovered_from": "source_checkpoint filename ckpt_00N000.pt"}
        for old, n in (
            ("crab_regen_2000", 2000),
            ("crab_regen_4000", 4000),
            ("crab_regen_6000", 6000),
            ("crab_regen_8000", 8000),
            ("crab_regen_8000_static", 8000),
        )
    },
}


def _with_head(meta: dict, old: str, new: str, sweep: str) -> dict:
    """``meta`` with `name`/`sweep`(/`train`) inserted right after `id`."""
    out: dict = {}
    for key, value in meta.items():
        if key != "id":
            out[key] = value
            continue
        out["id"] = new
        out["name"] = new
        out["sweep"] = sweep
        if old in RECOVERED:
            out["train"] = RECOVERED[old]
    return out


def plan() -> list[tuple[str, str, str]]:
    """Validate the table against the directory and return (old, new, sweep)."""
    present = {p.parent.name for p in EXPORTS.glob("*/meta.json")}
    known = set(TABLE)
    missing = sorted(present - known - {new for new, _ in TABLE.values()})
    if missing:
        raise SystemExit(f"exports/ has directories the table does not cover: {missing}")
    stale = sorted(known - present)
    if stale:
        raise SystemExit(f"the table lists directories that do not exist: {stale}")
    new_ids = [new for new, _ in TABLE.values()]
    dupes = sorted({i for i in new_ids if new_ids.count(i) > 1})
    if dupes:
        raise SystemExit(f"table maps two models to the same id: {dupes}")
    return [(old, *TABLE[old]) for old in TABLE if (EXPORTS / old).is_dir()]


def migrate(apply: bool) -> int:
    steps = plan()
    print(f"{len(steps)} model(s) to rename" + ("" if apply else " (dry run)"))
    for old, new, sweep in steps:
        meta_path = EXPORTS / old / "meta.json"
        meta = json.loads(meta_path.read_text())
        if meta.get("id") != old:
            raise SystemExit(f"{meta_path}: id {meta['id']!r} does not match its directory")
        meta = _with_head(meta, old, new, sweep)
        train = f" train={meta['train']}" if "train" in meta else ""
        print(f"  {old:24s} -> {new:36s} [{sweep}]{train}")
        if not apply:
            continue
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        subprocess.run(
            ["git", "mv", str(EXPORTS / old), str(EXPORTS / new)],
            check=True,
            cwd=ROOT,
        )

    if not apply:
        print("\nre-run with --apply")
        return 0

    from nca.export import reindex

    models = reindex(EXPORTS)
    print(f"rebuilt manifest.json ({len(models)} models)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="actually rename (default: dry run)")
    parser.add_argument("--dry-run", action="store_true", help="explicitly do nothing")
    args = parser.parse_args(argv)
    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive")
    try:
        return migrate(apply=args.apply and not args.dry_run)
    except SystemExit as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
