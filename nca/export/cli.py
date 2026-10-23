"""`python -m nca.export`: one model, all models, or a manifest rebuild."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..spec import ModelSpec
from .graph import DEFAULT_OPSET, _load_ca, export_step, strip_export_metadata
from .manifest import (
    _carry_over,
    _find_target,
    _git_rev,
    _run_layout,
    _run_train_json,
    _train_block,
    _update_manifest,
    derive_targets,
    loss_curve,
    reindex,
)
from .rebuild import model_from_export
from .verify import verify
from .weights import save_weights
from ..viz import render, save_image

def export_one(
    checkpoint: str | Path,
    model_id: str,
    out_dir: Path,
    grid: int | None,
    dynamic: bool,
    opset: int,
    experiment: str | None,
    target: str,
    do_verify: bool,
    verify_steps: int,
    wrap: bool = True,
    targets: list[str] | None = None,
    max_size: int | None = None,
    padding: int | None = None,
    carry: dict | None = None,
    run: dict | None = None,
    train: dict | None = None,
    rebuild: bool = False,
) -> dict:
    import onnx
    import onnxruntime as ort

    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)
    model_dir = out_dir / model_id
    prior_meta = None
    prior_onnx = None
    if (model_dir / "meta.json").is_file():
        prior_meta = json.loads((model_dir / "meta.json").read_text())
    if (model_dir / "model.onnx").is_file():
        prior_onnx = (model_dir / "model.onnx").read_bytes()
    model_dir.mkdir(parents=True, exist_ok=True)

    if rebuild:
        # No training run needed: the export is self-describing (meta + weights.bin
        # + the learned kernel, if any). This is the path that keeps the models
        # re-exportable after their run directories are gone.
        prior = json.loads((model_dir / "meta.json").read_text())
        ca = model_from_export(model_dir)
        max_size = max_size if max_size is not None else prior.get("target_size")
        padding = padding if padding is not None else prior.get("padding")
        run = run if run is not None else prior.get("train")
        loss = None
        experiment = experiment or (run or {}).get("regime") or prior.get("experiment") or "unknown"
    else:
        # Explicit flag > the run's own train.json > unknown. Deliberately no default:
        # inventing one is what put target.png on a different canvas than the run.
        run_size, run_padding = _run_layout(checkpoint)
        max_size = max_size if max_size is not None else run_size
        padding = padding if padding is not None else run_padding

        # The experiment the run recorded (nca/cli.py writes it into train.json), so
        # a re-export needs no --experiment/--train flags to stay reproducible.
        run = _run_train_json(checkpoint) if run is None else run
        ca = _load_ca(checkpoint, wrap=wrap)
        # The convergence shape, carried into the meta so the picker can show it
        # (the run's loss.json is not part of an export, and for many models the
        # run does not survive; a rebuild keeps whatever was recorded).
        loss = loss_curve(checkpoint.parent)

    train_block = _train_block(run or {}, train)
    # A multi-target run records the whole list; an export of it must not be
    # labelled "unknown" (see derive_targets).
    target, targets = derive_targets(target, targets, train_block)
    experiment = experiment or train_block.get("regime") or "unknown"
    spec = ca.spec
    if spec.num_parameters() != ca.num_parameters:
        raise RuntimeError(
            f"{model_id}: ModelSpec accounts for {spec.num_parameters()} parameters "
            f"but the module has {ca.num_parameters}. nca/spec.py is out of date with "
            f"nca/model.py (a new layer or parameter was added)."
        )
    target_arr = _find_target(model_dir if rebuild else checkpoint, target, max_size, padding)
    if grid is None:
        grid = int(target_arr.shape[-1]) if target_arr is not None else 64

    onnx_path = model_dir / "model.onnx"
    print(f"Exporting {model_id} (grid {grid}, dynamic={dynamic}, opset {opset})")
    export_step(
        ca,
        onnx_path,
        grid,
        dynamic=dynamic,
        opset=opset,
        cond_n=spec.cond_n,
    )

    parity = None
    if do_verify:
        parity = verify(ca, onnx_path, grid, verify_steps, dynamic)
    elif (
        prior_meta
        and prior_meta.get("parity_max_abs_diff")
        and prior_onnx is not None
        and prior_onnx == onnx_path.read_bytes()
    ):
        # An unverified re-export must not erase the record: if the bytes are
        # identical to the ones that were verified, the old measurement still
        # describes this graph. (Before this, `nca.export --all` without --verify
        # silently dropped the parity block from all 45 metas.)
        parity = prior_meta["parity_max_abs_diff"]

    if target_arr is not None:
        save_image(render(torch.from_numpy(target_arr))[0], model_dir / "target.png")

    weight_layout = save_weights(ca, model_dir, spec)

    model_proto = onnx.load(str(onnx_path))
    meta = {
        "id": model_id,
        **(carry or {}),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "channel_n": spec.channel_n,
        "cond_n": spec.cond_n,
        # 1 = the paper's single hidden block. The WGSL step implements 1..n.
        "hidden_layers": spec.hidden_layers,
        "hidden_n": spec.hidden_n,
        "fire_rate": spec.fire_rate,
        "learn_perception": spec.learn_perception,
        # The [3,3,3] (identity, d/dx, d/dy) kernels in use, flattened. For a
        # --learn-perception model these are the trained ones, so the frontend
        # must use them (and not the canonical Sobel). Kept at full float32
        # precision: rounding them here would make the meta a lossy copy, and
        # `--rebuild` has to reproduce the model exactly from the export.
        "perception_kernel": [float(np.float32(v)) for v in ca.kernels().detach().reshape(-1)],
        "num_parameters": spec.num_parameters(),
        "grid": grid,
        "dynamic_shape": dynamic,
        # How the target was placed on the grid: grid == target_size + 2*padding.
        # Recorded so target.png (and a future re-export) can be reproduced from
        # the meta alone -- 9 of the capacity-sweep runs predate target.npy.
        "target_size": max_size,
        "padding": padding,
        "wrap": spec.wrap,
        "opset": opset,
        "experiment": experiment,
        "target": target,
        # Full spec list for a conditional (--targets) run, so the playground can
        # label its target selector. Falls back to the single target.
        "targets": list(targets) if targets else [target],
        "source_checkpoint": str(checkpoint),
        "torch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
        "onnx_ir_version": model_proto.ir_version,
        "parity_max_abs_diff": parity,
        "weights": weight_layout,
        "size_bytes": onnx_path.stat().st_size,
        "git_rev": _git_rev(),
    }
    if train_block:
        meta["train"] = train_block
    # The study this model came from (see nca/experiments.py): the picker groups by
    # it, and it is the only field that says why 13 near-identical models exist.
    sweep = train_block.get("sweep") or (carry or {}).get("sweep")
    if sweep:
        meta["sweep"] = sweep
    # `loss` is the one training artefact the picker shows; a rebuild carries the
    # recorded curve (PRESENTATION_KEYS) since the run's loss.json is gone.
    curve = loss if loss is not None else (carry or {}).get("loss")
    if curve:
        meta["loss"] = curve
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    entry = {
        "id": model_id,
        "path": f"{model_id}/model.onnx",
        "meta": meta,
    }
    _update_manifest(out_dir, entry)
    print(f"  wrote {onnx_path} ({meta['size_bytes']} bytes) and meta.json")
    return meta


def export_all(
    out_dir: Path,
    *,
    opset: int = DEFAULT_OPSET,
    do_verify: bool = True,
    verify_steps: int = 8,
    only: list[str] | None = None,
    rebuild: bool = False,
) -> list[dict]:
    """Re-export every model in ``out_dir``, from its checkpoint or its own export.

    Grid / dynamic shape / experiment / target are taken from the existing
    ``meta.json`` so a re-export reproduces the same artefacts. This is the
    command to run after any change to ``nca/model.py`` or ``nca/export``; it
    then rewrites ``manifest.json`` via ``reindex`` so the manifest cannot drift.

    ``rebuild=True`` ignores the recorded ``source_checkpoint`` and rebuilds each
    model from its committed ``meta.json`` + ``weights.bin`` (see
    :mod:`nca.export.rebuild`), which is what makes the exports re-exportable
    once the run directories are gone.
    """
    out_dir = Path(out_dir)
    wanted = set(only) if only else None
    exported: list[dict] = []
    from_export: list[str] = []

    for meta_path in sorted(out_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        model_id = meta.get("id") or meta_path.parent.name
        if wanted is not None and model_id not in wanted:
            continue
        checkpoint = Path(meta.get("source_checkpoint") or "")
        # Not an error: the export describes its model completely, so a run that is
        # gone (cluster scratch, /tmp, a reboot) is not a reason to skip a model.
        rebuild_this = rebuild or not checkpoint.is_file()
        if rebuild_this and not rebuild:
            from_export.append(model_id)
        spec = ModelSpec.from_meta(meta)
        exported.append(
            export_one(
                checkpoint=checkpoint,
                model_id=model_id,
                out_dir=out_dir,
                grid=int(meta["grid"]) if meta.get("grid") else None,
                dynamic=bool(meta.get("dynamic_shape", True)),
                opset=opset,
                experiment=meta.get("experiment"),
                target=meta.get("target", "unknown"),
                do_verify=do_verify,
                verify_steps=verify_steps,
                wrap=spec.wrap,
                targets=meta.get("targets"),
                # Preserve the *recorded* canvas (None stays None: a legacy meta
                # without target_size/padding must not silently gain a default).
                max_size=meta.get("target_size"),
                padding=meta.get("padding"),
                carry=_carry_over(meta),
                rebuild=rebuild_this,
            )
        )

    if from_export:
        print(
            f"\nRebuilt {len(from_export)} model(s) from their own export "
            f"(source checkpoint gone): {', '.join(from_export)}"
        )

    reindex(out_dir)
    print(f"\nExported {len(exported)} model(s) into {out_dir}")
    return exported


def strip_all_metadata(out_dir: Path) -> list[str]:
    """Strip every ``exports/*/model.onnx`` in place. Returns the models touched."""
    touched = []
    for onnx_path in sorted(out_dir.glob("*/model.onnx")):
        if strip_export_metadata(onnx_path):
            touched.append(onnx_path.parent.name)
    return touched


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a trained NCA to ONNX.")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="path to a .pt checkpoint (required unless --all/--reindex)",
    )
    p.add_argument("--id", default=None, help="model id (default: run dir name)")
    p.add_argument("--out", default="exports", help="output directory")
    p.add_argument(
        "--all",
        action="store_true",
        help="re-export every model in --out (from its checkpoint, or from its own "
        "export when the run is gone)",
    )
    p.add_argument(
        "--only",
        nargs="+",
        metavar="ID",
        default=None,
        help="with --all: restrict to these model ids",
    )
    p.add_argument(
        "--reindex",
        action="store_true",
        help="rebuild manifest.json from */meta.json and exit",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="re-export from the committed export itself (meta.json + weights.bin) "
        "instead of the training checkpoint -- works after the run dir is gone. "
        "With --all it rebuilds every model; with --checkpoint exports/<id> (or "
        "--id) it rebuilds one",
    )
    p.add_argument("--grid-size", type=int, default=None, help="dummy grid size")
    p.add_argument(
        "--static",
        action="store_true",
        help="fix H/W instead of exporting dynamic spatial axes",
    )
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    p.add_argument("--experiment", default=None, help="training regime (default: the run's train.json)")
    p.add_argument("--target", default="unknown")
    p.add_argument(
        "--train",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="override a recorded train number (e.g. --train mismatch_p=0.3)",
    )
    p.add_argument(
        "--strip-metadata",
        action="store_true",
        help="rewrite every exports/*/model.onnx without the exporter's stack-trace "
        "metadata (no checkpoint needed; makes re-exports byte-stable)",
    )
    p.add_argument(
        "--refresh-targets",
        action="store_true",
        help="with --reindex: re-render target.png/target.npy for every meta that records "
        "one, and backfill the `loss` block from the run dirs that still exist "
        "(target_size+padding repairs exports whose target.npy is gone)",
    )
    p.add_argument(
        "--targets",
        nargs="+",
        default=None,
        metavar="SPEC",
        help="target spec list for a conditional model (metadata only)",
    )
    p.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="target canvas this run was trained with (default: the run's train.json; "
        "needed only when the run has no target.npy and --target must be reloaded)",
    )
    p.add_argument(
        "--padding",
        type=int,
        default=None,
        help="border this run was trained with; grid == target-size + 2*padding "
        "(default: the run's train.json)",
    )
    p.add_argument(
        "--no-wrap",
        action="store_true",
        help="use zero (non-periodic) boundaries instead of circular wrapping",
    )
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--verify-steps", type=int, default=8)
    return p


def _parse_train(items: list[str] | None) -> dict | None:
    """``--train mismatch_p=0.3`` -> ``{"mismatch_p": 0.3}`` (numbers/bools coerced)."""
    if not items:
        return None
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--train expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        for cast in (int, float):
            try:
                out[key] = cast(raw)
                break
            except ValueError:
                continue
        else:
            out[key] = {"true": True, "false": False}.get(raw.lower(), raw)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.strip_metadata:
        models = strip_all_metadata(Path(args.out))
        print(f"Stripped exporter metadata from {len(models)} model(s) in {Path(args.out)}")
        return

    if args.reindex:
        models = reindex(Path(args.out), fix_targets=args.refresh_targets, fix_loss=args.refresh_targets)
        print(f"Reindexed {len(models)} model(s) into {Path(args.out) / 'manifest.json'}")
        return

    if args.all:
        export_all(
            Path(args.out),
            opset=args.opset,
            do_verify=not args.no_verify,
            verify_steps=args.verify_steps,
            only=args.only,
            rebuild=args.rebuild,
        )
        return

    if args.rebuild:
        # `--checkpoint exports/<id> --rebuild`, or `--id <id> --rebuild`.
        model_dir = Path(args.checkpoint) if args.checkpoint else Path(args.out) / (args.id or "")
        if not (model_dir / "meta.json").is_file():
            parser.error(f"--rebuild needs an export directory with meta.json (got {model_dir})")
        export_one(
            checkpoint=model_dir,
            model_id=args.id or model_dir.name,
            out_dir=Path(args.out),
            grid=args.grid_size,
            dynamic=not args.static,
            opset=args.opset,
            experiment=args.experiment,
            target=args.target,
            do_verify=not args.no_verify,
            verify_steps=args.verify_steps,
            wrap=not args.no_wrap,
            targets=args.targets,
            max_size=args.target_size,
            padding=args.padding,
            train=_parse_train(args.train),
            rebuild=True,
        )
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --all or --reindex is given")
    checkpoint = Path(args.checkpoint)
    model_id = args.id or checkpoint.parent.name or checkpoint.stem

    export_one(
        checkpoint=checkpoint,
        model_id=model_id,
        out_dir=Path(args.out),
        grid=args.grid_size,
        dynamic=not args.static,
        opset=args.opset,
        experiment=args.experiment,
        target=args.target,
        do_verify=not args.no_verify,
        verify_steps=args.verify_steps,
        wrap=not args.no_wrap,
        targets=args.targets,
        max_size=args.target_size,
        padding=args.padding,
        train=_parse_train(args.train),
    )


if __name__ == "__main__":
    main()
