"""Headless WebGPU (wgpu-py) correctness + benchmark harness for the WGSL CA step.

Runs the exact shaders in `frontend/src/wgsl/` on the real GPU (Vulkan), compares
N steps against the ONNX model (onnxruntime CPU), and reports timing. This lets us
iterate on the shaders without a browser.

On NixOS the Vulkan loader/driver need to be on the library path, e.g.::

    LOADER=$(dirname $(ls /nix/store/*vulkan-loader*/lib/libvulkan.so.1 | head -1))
    LD_LIBRARY_PATH=$LOADER:/run/opengl-driver/lib \
    VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/intel_icd.x86_64.json \
    uv run --with wgpu,numpy,onnxruntime python scripts/shader_bench.py crab.regenerating --steps 200

Use ``--shader path.wgsl ...`` to benchmark variants of pass1. Several shaders at
once are measured round-robin per grid: that is the only honest comparison on a
GPU whose clocks drift (the same shader measured 13 ms and 16 ms on the UHD 620 in
different sessions).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import wgpu

ROOT = Path(__file__).resolve().parent.parent
WGSL_DIR = ROOT / "frontend" / "src" / "wgsl"
# This script is run as `python scripts/shader_bench.py`, so sys.path[0] is
# scripts/ -- make the repo importable before reaching for the shared kernels.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nca import step as steplib  # noqa: E402  (after the sys.path fix)


def canonical_kernels() -> np.ndarray:
    """The 27-float kernel buffer the shader takes, `[identity, dx, dy]` flat."""
    return steplib.canonical_kernels().reshape(-1)


def perception_weight(C: int) -> np.ndarray:
    """[3C, 1, 3, 3] depthwise weight for the ONNX model."""
    return steplib.perception_weight(C)


def make_seed(C: int, grid: int) -> np.ndarray:
    return steplib.make_seed(C, grid, grid)[None]


def read_back(device, src, nbytes: int) -> np.ndarray:
    staging = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )
    enc = device.create_command_encoder()
    enc.copy_buffer_to_buffer(src, 0, staging, 0, nbytes)
    device.queue.submit([enc.finish()])
    staging.map_sync(mode=wgpu.MapMode.READ)
    data = np.frombuffer(staging.read_mapped(), dtype=np.float32).copy()
    staging.unmap()
    staging.destroy()
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="model id under exports/")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument(
        "--shader",
        type=Path,
        nargs="+",
        default=None,
        help="override pass1.wgsl; several paths are benchmarked interleaved so the "
        "comparison is not polluted by clock/thermal drift (the iGPU swings ~25%% "
        "between sessions, see docs/findings/shader-perf.md)",
    )
    ap.add_argument("--reference", action="store_true", help="run the ONNX comparison")
    ap.add_argument("--grids", default=None,
                    help="comma-separated grid sizes to sweep, e.g. 72,128,256,512")
    ap.add_argument("--json", default=None, help="write the results to this file")
    ap.add_argument("--runs", type=int, default=3, help="benchmark repeats")
    ap.add_argument("--covx", type=int, default=8, help="cells covered per workgroup in x")
    ap.add_argument("--covy", type=int, default=8, help="cells covered per workgroup in y")
    ap.add_argument(
        "--entry",
        default="auto",
        help="pass1 entry point: auto (step/step_deep by hidden_layers when the size "
        "constants are passed, else main), step, step_deep or main",
    )
    ap.add_argument(
        "--consts",
        action="store_true",
        help="pass the size override constants even with --shader (needed for a "
        "modified pass1 that keeps the override declarations)",
    )
    args = ap.parse_args()

    out_dir = ROOT / "exports" / args.model
    meta = json.loads((out_dir / "meta.json").read_text())
    C = int(meta["channel_n"])
    cond_n = int(meta.get("cond_n", 0) or 0)
    grid = args.grid or meta["grid"]
    hidden = meta["weights"]["hidden_n"]
    layers = int(meta["weights"].get("hidden_layers", 1))
    off = meta["weights"]["offsets"]
    weights = np.fromfile(out_dir / "weights.bin", dtype="<f4")
    pass2_wgsl = (WGSL_DIR / "pass2.wgsl").read_text()

    kernel = canonical_kernels()

    # ---- device / pipelines -------------------------------------------------
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    print(f"adapter: {adapter.info['device']} ({adapter.info['adapter_type']})")

    m2 = device.create_shader_module(code=pass2_wgsl)
    def make_layout(spec):
        entries = []
        for binding, kind in enumerate(spec):
            entries.append(
                {
                    "binding": binding,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": kind},
                }
            )
        return device.create_bind_group_layout(entries=entries)

    layout1 = make_layout(
        [
            "read-only-storage",  # 0 state
            "read-only-storage",  # 1 fire
            "read-only-storage",  # 2 weights (f32)
            "read-only-storage",  # 3 kernel
            "storage",            # 4 out
            "uniform",            # 5 params
            "read-only-storage",  # 6 w4 (vec4 weights)
            "read-only-storage",  # 7 cond (one-hot selector)
        ]
    )
    layout2 = make_layout(["read-only-storage", "read-only-storage", "storage", "uniform"])
    pl1 = device.create_pipeline_layout(bind_group_layouts=[layout1])
    pl2 = device.create_pipeline_layout(bind_group_layouts=[layout2])
    w1s4 = -(-(3 * C + cond_n) // 4)  # ceil((3C + cond_n) / 4)
    const1 = {
        "C": C, "NP4": w1s4, "W1S4": w1s4, "HG": hidden // 4,
        "CONDN": cond_n, "LH": layers,
    }
    const2 = {"C": C}

    def patch_pass1(path: Path) -> tuple[str, bool]:
        """The variant's source, plus whether it takes the size override constants."""
        src = path.read_text()
        # HMAX must be a const-expression equal to the real hidden-group count, so
        # it is substituted per model (same as the frontend does).
        src = src.replace("HMAX: u32 = 64u", f"HMAX: u32 = {max(hidden // 4, 1)}u")
        return src, path.resolve() == (WGSL_DIR / "pass1.wgsl").resolve()

    # One pipeline per shader: the shipped pass1 ships the shallow and deep update
    # rules as two entry points so neither pays for the other's registers; the
    # scripted variants have a single `main`.
    def make_pass1(path: Path):
        src, is_shipped = patch_pass1(path)
        module = device.create_shader_module(code=src, label=str(path))
        entry = args.entry
        if entry == "auto":
            entry = ("step_deep" if layers > 1 else "step") if is_shipped else "main"
        constants = const1 if (args.consts or is_shipped) else {}
        pipe = device.create_compute_pipeline(
            layout=pl1, compute={"module": module, "entry_point": entry, "constants": constants}
        )
        return pipe, entry

    shader_paths = args.shader or [WGSL_DIR / "pass1.wgsl"]
    variants = []
    for path in shader_paths:
        label = path.name if path.parent != WGSL_DIR else "pass1"
        pipe, entry = make_pass1(path)
        variants.append({"label": label, "path": path, "pipe": pipe, "entry": entry})
    print("shaders: " + ", ".join(f"{v['label']}({v['entry']})" for v in variants))
    p2 = device.create_compute_pipeline(
        layout=pl2, compute={"module": m2, "entry_point": "main", "constants": const2}
    )

    def bench_grid(grid: int, variants: list) -> dict:
        plane = grid * grid
        state_nbytes = C * plane * 4
        seed = make_seed(C, grid)
        d1 = None
        S = wgpu.BufferUsage.STORAGE
        D = wgpu.BufferUsage.COPY_DST
        def sbuf(data, usage=S | D):
            b = device.create_buffer(size=max(len(data.tobytes()), 16), usage=usage)
            device.queue.write_buffer(b, 0, data.tobytes())
            return b

        SRC = wgpu.BufferUsage.COPY_SRC
        stateA = sbuf(seed.reshape(-1), S | D | SRC)
        stateB = sbuf(np.zeros(C * plane, np.float32), S | D | SRC)
        tmp = sbuf(np.zeros(C * plane, np.float32))
        fire = sbuf(np.ones(plane, np.float32), usage=S | D)
        weightBuf = sbuf(weights)
        w4Buf = sbuf(weights)
        kernelBuf = sbuf(kernel)
        # One-hot target selector; only index 0 is exercised (any consistent choice
        # works for a parity check).
        cond_vals = np.zeros(max(cond_n, 1), np.float32)
        cond_vals[0] = 1.0
        condBuf = sbuf(cond_vals)
        params = bytearray(48)
        struct.pack_into("<8I", params, 0, C, grid, grid, hidden, off["w1"], off["b1"], off["w2"], off["b2"])
        struct.pack_into("<f", params, 32, 1.0)
        struct.pack_into("<I", params, 36, off.get("wmid", 0))
        paramsBuf = device.create_buffer_with_data(data=bytes(params), usage=wgpu.BufferUsage.UNIFORM | D)

        def bg1(state):
            return device.create_bind_group(
                layout=layout1,
                entries=[
                    {"binding": 0, "resource": {"buffer": state}},
                    {"binding": 1, "resource": {"buffer": fire}},
                    {"binding": 2, "resource": {"buffer": weightBuf}},
                    {"binding": 3, "resource": {"buffer": kernelBuf}},
                    {"binding": 4, "resource": {"buffer": tmp}},
                    {"binding": 5, "resource": {"buffer": paramsBuf}},
                    {"binding": 6, "resource": {"buffer": w4Buf}},
                    {"binding": 7, "resource": {"buffer": condBuf}},
                ],
            )

        def bg2(old, nxt):
            return device.create_bind_group(
                layout=layout2,
                entries=[
                    {"binding": 0, "resource": {"buffer": old}},
                    {"binding": 1, "resource": {"buffer": tmp}},
                    {"binding": 2, "resource": {"buffer": nxt}},
                    {"binding": 3, "resource": {"buffer": paramsBuf}},
                ],
            )

        bgs1 = [bg1(stateA), bg1(stateB)]
        bgs2 = [bg2(stateA, stateB), bg2(stateB, stateA)]
        dispatch = ((grid + args.covx - 1) // args.covx, (grid + args.covy - 1) // args.covy)

        def run_steps(masks, cur, pipe):
            for m in masks:
                device.queue.write_buffer(fire, 0, m.tobytes())
                enc = device.create_command_encoder()
                pass1 = enc.begin_compute_pass()
                pass1.set_pipeline(pipe)
                pass1.set_bind_group(0, bgs1[cur])
                pass1.dispatch_workgroups(dispatch[0], dispatch[1])
                pass1.end()
                pass2 = enc.begin_compute_pass()
                pass2.set_pipeline(p2)
                pass2.set_bind_group(0, bgs2[cur])
                pass2.dispatch_workgroups(dispatch[0], dispatch[1])
                pass2.end()
                device.queue.submit([enc.finish()])
                cur ^= 1
            return cur

        rng = np.random.default_rng(0)
        masks = [(rng.random(plane) < 0.5).astype(np.float32) for _ in range(args.steps)]

        # ---- correctness vs ONNX ------------------------------------------------
        if args.reference:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(out_dir / "model.onnx"), providers=["CPUExecutionProvider"])
            w_onx = perception_weight(C)
            dense_masks = [m.reshape(1, 1, grid, grid) for m in masks]

            def onnx_step(state, m):
                feeds = {
                    "x": state,
                    "perception_weight": w_onx,
                    "fire_mask": m,
                    "step_size": np.array(1.0, np.float32),
                }
                if cond_n:
                    feeds["cond"] = cond_vals.reshape(1, cond_n, 1, 1)
                return sess.run(["x_next"], feeds)[0]

            def shader_run(nsteps, pipe):
                device.queue.write_buffer(stateA, 0, seed.reshape(-1).tobytes())
                cur = run_steps(masks[:nsteps], 0, pipe)
                got = read_back(device, stateB if cur else stateA, state_nbytes)
                return got.reshape(1, C, grid, grid)

            # 1 step is the definitive check: the CA is chaotic, so a bare fp32
            # rounding difference amplifies ~1.1-1.2x per step over long runs.
            ref1 = onnx_step(seed, dense_masks[0])
            ref = ref1
            if args.steps > 1:
                for m in dense_masks[1:]:
                    ref = onnx_step(ref, m)
            for v in variants:
                d1 = float(np.max(np.abs(shader_run(1, v["pipe"]) - ref1)))
                v["parity_1step"] = d1
                print(f"correctness vs ONNX [{v['label']}]: 1 step = {d1:.3e}  "
                      f"[{'ok' if d1 < 1e-4 else 'FAIL'}]")
                if args.steps > 1:
                    dn = float(np.max(np.abs(shader_run(args.steps, v["pipe"]) - ref)))
                    print(f"                    {args.steps} steps = {dn:.3e}  (amplified by chaos)")

            device.queue.write_buffer(stateA, 0, seed.reshape(-1).tobytes())

        # ---- benchmark ----------------------------------------------------------
        # Round-robin over the shaders, one repetition each, so slow clock drift hits
        # every variant equally instead of the last one measured.
        best = {v["label"]: float("inf") for v in variants}
        for _ in range(args.runs):
            for v in variants:
                t0 = time.perf_counter()
                run_steps(masks, 0, v["pipe"])
                read_back(device, stateA, state_nbytes)  # drain
                ms = (time.perf_counter() - t0) / args.steps * 1000
                best[v["label"]] = min(best[v["label"]], ms)
        for v in variants:
            ms = best[v["label"]]
            v.setdefault("grids", {})[grid] = ms
            print(f"grid {grid}x{grid} · {v['label']:<22} · {args.steps} steps · "
                  f"{ms:.2f} ms/step best of {args.runs} ({1000 / ms:.0f} steps/s)")

        return {"grid": grid, "best": best}

    grids = [int(g) for g in args.grids.split(",")] if args.grids else [args.grid or meta["grid"]]
    per_grid = [bench_grid(g, variants) for g in grids]
    if len(grids) > 1 or len(variants) > 1:
        header = "| grid | " + " | ".join(v["label"] for v in variants) + " |"
        print()
        print(header)
        print("|" + "---|" * (len(variants) + 1))
        for row in per_grid:
            cells = " | ".join(f"{row['best'][v['label']]:.2f}" for v in variants)
            print(f"| {row['grid']}² | {cells} |")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "shaders": [str(v["path"]) for v in variants],
                    "entries": {v["label"]: v["entry"] for v in variants},
                    "steps": args.steps,
                    "runs": args.runs,
                    "results": [
                        {
                            "grid": row["grid"],
                            "ms_step": row["best"],
                            "parity_1step": {
                                v["label"]: v.get("parity_1step") for v in variants
                            },
                        }
                        for row in per_grid
                    ],
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
