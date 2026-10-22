"""Validate ``exports/*/meta.json`` against ``nca.spec.ModelSpec`` and the graphs.

``meta.json`` is the contract between the exporter, the frontend and the WGSL
engine, but nothing used to check it: the ``weights.offsets`` block, ``cond_n``
and the per-layer arrays were hand-written copies of the model's shape. This
script re-derives all of it from ``ModelSpec`` and diffs, so shape drift is a
failure instead of a silently wrong render.

    python -m scripts.check_meta                 # every model, exit 1 on error
    python -m scripts.check_meta --strict        # warnings are errors too
    python -m scripts.check_meta --fix           # rewrite drifted fields from the spec
    python -m scripts.check_meta --only cross8   # one model (prefix match)

Checks, cheapest first:

1. required keys, types, ``id`` == directory name
2. spec fields (``channel_n``/``hidden_n``/``hidden_layers``/``cond_n``/...) agree
   with ``ModelSpec.from_meta`` and ``spec.meta_fields()``
3. the ``weights`` block (offsets, count, per-layer dims) agrees with the spec,
   and ``count * 4`` matches the real ``weights.bin`` size
4. if ONNX Runtime is installed, the exported graph's inputs/outputs match the
   spec (``x`` has ``C`` channels, ``perception_weight`` is ``[3C,1,3,3]``, the
   ``cond`` input exists iff ``cond_n > 0``, opset 19)
5. ``target.png`` / ``model.onnx`` exist; ``source_checkpoint`` still exists
   (a warning: exports are valid even when the run directory is gone)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nca.spec import ModelSpec

EXPORTS = Path(__file__).resolve().parents[1] / "exports"

REQUIRED_KEYS = (
    "id",
    "channel_n",
    "hidden_n",
    "fire_rate",
    "num_parameters",
    "grid",
    "dynamic_shape",
    "wrap",
    "opset",
    "experiment",
    "target",
    "source_checkpoint",
    "weights",
)

# Keys the exporter writes that the spec owns. Missing optional ones are warnings:
# the frontend defaults ``hidden_layers ?? 1`` and falls back to the canonical
# Sobel kernel, so an old meta still renders -- it is just less explicit.
SPEC_KEYS = ("channel_n", "hidden_n", "hidden_layers", "cond_n", "fire_rate", "learn_perception", "wrap")


@dataclass(frozen=True)
class Problem:
    level: str  # "error" | "warn" | "info"
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.message}"


def _error(msg: str) -> Problem:
    return Problem("error", msg)


def _warn(msg: str) -> Problem:
    return Problem("warn", msg)


def _check_files(model_dir: Path, meta: dict[str, Any]) -> list[Problem]:
    problems: list[Problem] = []
    for name in ("model.onnx", "weights.bin", "target.png"):
        if not (model_dir / name).is_file():
            problems.append(_error(f"missing {name}"))
    checkpoint = Path(str(meta.get("source_checkpoint", "")))
    if not checkpoint.is_file():
        problems.append(_warn(f"source_checkpoint {checkpoint} is gone (re-export needs it)"))
    if meta.get("weights", {}).get("file", "weights.bin") != "weights.bin":
        problems.append(_error("weights.file must be weights.bin (the frontend fetches it by name)"))
    return problems


def _check_spec_fields(
    model_id: str, model_dir: Path, meta: dict[str, Any], spec: ModelSpec
) -> list[Problem]:
    problems: list[Problem] = []
    if meta.get("id") != model_id:
        problems.append(_error(f"id {meta.get('id')!r} != directory {model_id!r}"))
    # The picker groups by `sweep` and expects `name` to be the id; without them a
    # model falls into "Other" (and a re-export used to drop them silently, which
    # this catches -- `export_one(carry=...)` now preserves them).
    if not meta.get("sweep"):
        problems.append(_warn("no sweep[] -- the picker cannot group this model"))
    if "name" in meta and meta["name"] != meta.get("id"):
        problems.append(_error(f"name {meta['name']!r} != id {meta.get('id')!r}"))

    expected = spec.meta_fields()
    for key in SPEC_KEYS:
        if key not in meta:
            problems.append(_warn(f"{key} missing (spec says {expected[key]!r}); re-export or --fix"))
        elif meta[key] != expected[key]:
            problems.append(_error(f"{key} is {meta[key]!r}, spec says {expected[key]!r}"))
    if meta.get("num_parameters") != expected["num_parameters"]:
        problems.append(
            _error(f"num_parameters is {meta.get('num_parameters')!r}, spec says {expected['num_parameters']}")
        )
    if meta.get("cond_n", 0) and not meta.get("labels"):
        problems.append(Problem("info", "conditional model without labels[] (the picker shows indices)"))
    labels = meta.get("labels")
    if labels is not None and len(labels) != meta.get("cond_n", 0):
        problems.append(_error(f"labels has {len(labels)} entries, cond_n is {meta.get('cond_n', 0)}"))
    targets = meta.get("targets")
    if targets is not None and len(targets) != max(meta.get("cond_n", 0), 1):
        problems.append(_error(f"targets has {len(targets)} entries, cond_n is {meta.get('cond_n', 0)}"))
    if not isinstance(meta.get("grid"), int) or meta["grid"] <= 0:
        problems.append(_error("grid must be a positive int (dummy export shape)"))
    # grid is the canvas the run grew on: target_size + 2*padding. Pre-spec runs
    # have no target.npy, so the split cannot be recovered from the export alone --
    # say so once instead of inventing numbers.
    target_size, padding = meta.get("target_size"), meta.get("padding")
    if target_size is None or padding is None:
        level = "info" if (model_dir / "target.png").exists() else "warn"
        problems.append(
            Problem(
                level,
                "target_size/padding not recorded (run predates target.npy); "
                "target.png is committed, but a re-export needs both flags",
            )
        )
    elif isinstance(meta.get("grid"), int) and target_size + 2 * padding != meta["grid"]:
        problems.append(
            _error(f"target_size {target_size} + 2*padding {padding} != grid {meta['grid']}")
        )
    return problems


def _check_weights(model_dir: Path, meta: dict[str, Any], spec: ModelSpec) -> list[Problem]:
    problems: list[Problem] = []
    block = meta.get("weights")
    if not isinstance(block, dict):
        return [_error("weights block missing or not an object")]

    expected = spec.weights_meta()
    if block.get("count") != expected["count"]:
        problems.append(_error(f"weights.count is {block.get('count')!r}, spec says {expected['count']}"))
    if block.get("offsets") != expected["offsets"]:
        problems.append(_error(f"weights.offsets is {block.get('offsets')!r}, spec says {expected['offsets']}"))
    for key in ("hidden_layers", "perception_n", "perception_padded_n"):
        if key not in block:
            problems.append(_warn(f"weights.{key} missing (spec says {expected[key]!r}); re-export or --fix"))
        elif block[key] != expected[key]:
            problems.append(_error(f"weights.{key} is {block[key]!r}, spec says {expected[key]!r}"))
    for key in ("channel_n", "hidden_n"):
        if block.get(key) != expected[key]:
            problems.append(_error(f"weights.{key} is {block.get(key)!r}, spec says {expected[key]!r}"))

    blob = model_dir / str(block.get("file", "weights.bin"))
    if blob.is_file():
        size = blob.stat().st_size
        if size != expected["count"] * 4:
            problems.append(_error(f"weights.bin is {size} bytes, spec wants {expected['count'] * 4}"))
    return problems


def _dims(value_info) -> list[Any]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param or "?")
    return dims


def _check_graph(model_dir: Path, meta: dict[str, Any], spec: ModelSpec) -> list[Problem]:
    try:
        import onnx
    except ImportError:  # pragma: no cover - onnx is a dev dependency
        return []

    path = model_dir / "model.onnx"
    if not path.is_file():
        return []  # already reported
    model = onnx.load(str(path))
    graph = model.graph
    inputs = {i.name: _dims(i) for i in graph.input}
    outputs = {o.name: _dims(o) for o in graph.output}

    problems: list[Problem] = []
    want_in = {
        "x": [None, spec.channel_n, None, None],
        # 3C, not perception_n: the one-hot is concatenated after the conv.
        "perception_weight": [spec.conv_n, 1, 3, 3],
        "fire_mask": [None, 1, None, None],
        "step_size": [],
    }
    if spec.cond_n:
        want_in["cond"] = [None, spec.cond_n, 1, 1]
    for name, want in want_in.items():
        if name not in inputs:
            problems.append(_error(f"graph input {name!r} missing"))
            continue
        got = inputs[name]
        if len(got) != len(want):
            problems.append(_error(f"graph input {name!r} has rank {len(got)}, expected {len(want)}"))
            continue
        for axis, (g, w) in enumerate(zip(got, want, strict=True)):
            if w is not None and g != w:
                problems.append(_error(f"graph input {name!r} axis {axis} is {g!r}, expected {w}"))
    extra = set(inputs) - set(want_in)
    if extra:
        problems.append(_error(f"unexpected graph input(s) {sorted(extra)}"))

    if "x_next" not in outputs:
        problems.append(_error("graph output 'x_next' missing"))
    else:
        got = outputs["x_next"]
        if len(got) != 4 or got[1] != spec.channel_n:
            problems.append(_error(f"graph output x_next is {got}, expected [*, {spec.channel_n}, *, *]"))

    opset = [o.version for o in model.opset_import if o.domain in ("", "ai.onnx")]
    if opset != [meta.get("opset", 19)]:
        problems.append(_error(f"graph opset is {opset}, meta says {meta.get('opset')}"))
    if meta.get("torch_version") and not meta.get("parity_max_abs_diff"):
        problems.append(_warn("no parity_max_abs_diff recorded (exported with --no-verify?)"))
    for grid, diff in (meta.get("parity_max_abs_diff") or {}).items():
        if diff > 1e-4:
            problems.append(_warn(f"parity at {grid}px is {diff:.2e} (> 1e-4)"))
    if meta.get("size_bytes") and path.stat().st_size != meta["size_bytes"]:
        problems.append(_warn(f"model.onnx is {path.stat().st_size} bytes, meta says {meta['size_bytes']}"))
    return problems


def check_model(model_dir: Path, *, graph: bool = True) -> tuple[dict[str, Any] | None, list[Problem]]:
    """Check one exported model directory. Returns ``(meta, problems)``."""
    meta_path = model_dir / "meta.json"
    if not meta_path.is_file():
        return None, [_error("no meta.json")]
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        return None, [_error(f"meta.json is not valid JSON: {exc}")]

    model_id = model_dir.name
    problems: list[Problem] = [_error(f"missing key {key!r}") for key in REQUIRED_KEYS if key not in meta]
    try:
        spec = ModelSpec.from_meta(meta)
    except (KeyError, TypeError, ValueError) as exc:
        return meta, problems + [_error(f"ModelSpec.from_meta failed: {exc}")]

    problems += _check_files(model_dir, meta)
    problems += _check_spec_fields(model_id, model_dir, meta, spec)
    problems += _check_weights(model_dir, meta, spec)
    if graph:
        problems += _check_graph(model_dir, meta, spec)
    return meta, problems


def derive_spec(model_dir: Path, meta: dict[str, Any]) -> tuple[ModelSpec | None, str]:
    """Recover the spec from the artifacts (graph + blob), not from ``meta.json``.

    ``--fix`` used to rebuild the spec out of the very file it was repairing, so a
    wrong ``hidden_n`` was "repaired" into a self-consistent but wrong meta (and
    the wrong ``weights.count`` / ``num_parameters`` overwrote the last correct
    fields). The ONNX graph and ``weights.bin`` are the ground truth; ``meta.json``
    is what we are checking. ``learn_perception``/``fire_rate`` are not in either
    artifact, so they are taken from ``meta`` (``learn_perception`` only adds the
    18-float Sobel pair, which is *not* in the blob).
    """
    try:
        import onnx
    except ImportError:
        return None, "onnx is not installed"
    path = model_dir / "model.onnx"
    if not path.is_file():
        return None, "no model.onnx"
    graph = onnx.load(str(path)).graph
    inputs = {i.name: _dims(i) for i in graph.input}
    pw = inputs.get("perception_weight")
    if not pw or len(pw) != 4 or not isinstance(pw[0], int) or pw[0] % 3:
        return None, f"perception_weight has unusable shape {pw}"
    channel_n = pw[0] // 3
    cond = inputs.get("cond")
    cond_n = int(cond[1]) if cond and isinstance(cond[1], int) else 0

    layers: list[tuple[int, list[int]]] = []
    for init in graph.initializer:
        if init.name.startswith("ca.update.") and init.name.endswith(".weight"):
            idx = int(init.name.split(".")[2])
            layers.append((idx, [int(d) for d in init.dims]))
    if not layers:
        return None, "no ca.update.*.weight initializers"
    layers.sort()
    hidden_n = layers[0][1][0]
    hidden_layers = len(layers)
    perception_n = layers[0][1][1]
    if perception_n != 3 * channel_n + cond_n:
        return None, f"update[0] fan-in {perception_n} != 3C+cond_n ({3 * channel_n + cond_n})"
    if [idx for idx, _ in layers] != [2 * i for i in range(hidden_layers)]:
        return None, f"unexpected update layer indices {[i for i, _ in layers]}"

    wrap = True
    for node in graph.node:
        if node.op_type == "Pad":
            for attr in node.attribute:
                if attr.name == "mode" and attr.s == b"constant":
                    wrap = False

    try:
        spec = ModelSpec(
            channel_n=channel_n,
            hidden_n=hidden_n,
            hidden_layers=hidden_layers,
            cond_n=cond_n,
            learn_perception=bool(meta.get("learn_perception", False)),
            wrap=wrap,
            fire_rate=float(meta.get("fire_rate", 0.5)),
        )
    except ValueError as exc:
        return None, f"derived spec is invalid: {exc}"

    blob = model_dir / "weights.bin"
    if blob.is_file() and blob.stat().st_size != spec.weights_count() * 4:
        return None, (
            f"weights.bin is {blob.stat().st_size} bytes but the graph implies "
            f"{spec.weights_count() * 4} ({spec})"
        )
    return spec, ""


def _run_layout(meta: dict[str, Any]) -> tuple[int | None, int | None]:
    """``(target_size, padding)`` from the run's own ``train.json``, if reachable."""
    checkpoint = meta.get("source_checkpoint")
    if not checkpoint:
        return None, None
    train_json = Path(checkpoint).parent / "train.json"
    if not train_json.is_file():
        return None, None
    try:
        data = json.loads(train_json.read_text())
    except json.JSONDecodeError:
        return None, None
    size, pad = data.get("target_size"), data.get("padding")
    return (int(size) if size is not None else None, int(pad) if pad is not None else None)


def fix_meta(
    meta: dict[str, Any], spec: ModelSpec, layout: tuple[int | None, int | None] = (None, None)
) -> dict[str, Any]:
    """Return ``meta`` with every spec-owned field rewritten from ``spec``.

    Only fields for which there is evidence are written: the shape comes from
    ``spec`` (derived from the graph + blob), the canvas split comes from the run's
    ``train.json``. Nothing is invented -- a value we cannot reconstruct is left
    absent so the check keeps reporting it.
    """
    fixed = dict(meta)
    fixed.update(spec.meta_fields())
    fixed["weights"] = spec.weights_meta(str(meta.get("weights", {}).get("file", "weights.bin")))
    target_size, padding = layout
    if fixed.get("target_size") is None and target_size is not None:
        fixed["target_size"] = target_size
    if fixed.get("padding") is None and padding is not None:
        fixed["padding"] = padding
    return fixed


def repair_target_png(model_dir: Path, meta: dict[str, Any]) -> Problem | None:
    """Recreate a missing ``target.png`` from ``meta`` alone (used by ``--fix``)."""
    if (model_dir / "target.png").exists():
        return None
    specs = meta.get("targets") or [meta.get("target")]
    specs = [str(s) for s in specs if s and s != "unknown"]
    if not specs:
        return _warn("target.png missing and no target spec recorded")
    import torch

    from nca.targets import prepare_targets
    from nca.viz import render, save_image

    try:
        arr = prepare_targets(
            specs[:1],
            max_size=int(meta.get("target_size") or 40),
            padding=int(meta.get("padding") or 16),
        )
    except (OSError, ValueError) as exc:
        return _warn(f"target.png missing and {specs[0]!r} could not be reloaded ({exc})")
    save_image(render(torch.from_numpy(arr.numpy()))[0], model_dir / "target.png")
    return Problem("info", f"target.png rebuilt from {specs[0]!r}")


def check_exports(
    exports: Path = EXPORTS,
    *,
    only: str | None = None,
    graph: bool = True,
    fix: bool = False,
) -> dict[str, list[Problem]]:
    results: dict[str, list[Problem]] = {}
    rewrote = False
    for model_dir in sorted(p for p in exports.iterdir() if p.is_dir()):
        if only and not model_dir.name.startswith(only):
            continue
        meta, problems = check_model(model_dir, graph=graph)
        if fix and meta is not None:
            notes: list[Problem] = []
            derived, why = derive_spec(model_dir, meta)
            if derived is None:
                notes.append(_warn(f"--fix skipped: {why}"))
                fixed = meta
            else:
                fixed = fix_meta(meta, derived, _run_layout(meta))
            if fixed != meta:
                (model_dir / "meta.json").write_text(json.dumps(fixed, indent=2) + "\n")
                meta = fixed
                rewrote = True
                notes.append(Problem("info", "meta.json rewritten by --fix"))
            repaired = repair_target_png(model_dir, meta)
            if repaired is not None:
                notes.append(repaired)
            # Re-check instead of trusting the repair, and report only what is
            # still wrong plus what was just done.
            problems = check_model(model_dir, graph=graph)[1] + notes
        results[model_dir.name] = problems
    if rewrote:
        # The manifest embeds a copy of each meta, so rewritten metas need a reindex.
        from nca.export import reindex

        reindex(exports)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exports", type=Path, default=EXPORTS)
    parser.add_argument("--only", help="only models whose directory name starts with this")
    parser.add_argument("--strict", action="store_true", help="treat warnings as errors")
    parser.add_argument("--fix", action="store_true", help="rewrite drifted spec-owned fields")
    parser.add_argument("--no-graph", action="store_true", help="skip the ONNX graph checks")
    args = parser.parse_args(argv)

    results = check_exports(args.exports, only=args.only, graph=not args.no_graph, fix=args.fix)
    errors = warns = infos = 0
    for model_id, problems in results.items():
        for problem in problems:
            print(f"{model_id}: {problem}")
            errors += problem.level == "error"
            warns += problem.level == "warn"
            infos += problem.level == "info"
    ok = len(results) - sum(any(p.level == "error" for p in ps) for ps in results.values())
    print(f"\n{ok}/{len(results)} models clean | {errors} error(s), {warns} warning(s), {infos} note(s)")
    if errors or (args.strict and warns):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
