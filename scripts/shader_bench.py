"""Headless WebGPU (wgpu-py) correctness + benchmark harness for the WGSL CA step.

Runs the exact shaders in `frontend/src/wgsl/` on the real GPU (Vulkan), compares
N steps against the ONNX model (onnxruntime CPU), and reports timing. This lets us
iterate on the shaders without a browser.

On NixOS the Vulkan loader/driver need to be on the library path, e.g.::

    LOADER=$(dirname $(ls /nix/store/*vulkan-loader*/lib/libvulkan.so.1 | head -1))
    LD_LIBRARY_PATH=$LOADER:/run/opengl-driver/lib \
    VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/intel_icd.x86_64.json \
    uv run --with wgpu,numpy,onnxruntime python scripts/shader_bench.py crab.regenerating --steps 200

Use ``--shader path.wgsl`` to benchmark a variant of pass1.
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
    ap.add_argument("--shader", type=Path, default=None, help="override pass1.wgsl")
    ap.add_argument("--reference", action="store_true", help="run the ONNX comparison")
    ap.add_argument("--runs", type=int, default=3, help="benchmark repeats")
    ap.add_argument("--covx", type=int, default=8, help="cells covered per workgroup in x")
    ap.add_argument("--covy", type=int, default=8, help="cells covered per workgroup in y")
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
    pass1_wgsl = (args.shader or (WGSL_DIR / "pass1.wgsl")).read_text()
    # HMAX must be a const-expression equal to the real hidden-group count, so it
    # is substituted per model (same as the frontend does).
    pass1_wgsl = pass1_wgsl.replace(
        "HMAX: u32 = 64u", f"HMAX: u32 = {max(hidden // 4, 1)}u"
    )
    pass2_wgsl = (WGSL_DIR / "pass2.wgsl").read_text()

    plane = grid * grid
    state_nbytes = C * plane * 4

    seed = make_seed(C, grid)
    kernel = canonical_kernels()

    # ---- device / pipelines -------------------------------------------------
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    print(f"adapter: {adapter.info['device']} ({adapter.info['adapter_type']})")

    m1 = device.create_shader_module(code=pass1_wgsl)
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
    # Size constants are pipeline overrides; the scripted variants hardcode their
    # own sizes, so only the default pass1/pass2 get them.
    use_consts = args.consts or not args.shader
    const1 = {
        "C": C, "NP4": w1s4, "W1S4": w1s4, "HG": hidden // 4,
        "CONDN": cond_n, "LH": layers,
    } if use_consts else {}
    const2 = {"C": C} if use_consts else {}
    p1 = device.create_compute_pipeline(
        layout=pl1, compute={"module": m1, "entry_point": "main", "constants": const1}
    )
    p2 = device.create_compute_pipeline(
        layout=pl2, compute={"module": m2, "entry_point": "main", "constants": const2}
    )

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

    def run_steps(masks, cur):
        for m in masks:
            device.queue.write_buffer(fire, 0, m.tobytes())
            enc = device.create_command_encoder()
            pass1 = enc.begin_compute_pass()
            pass1.set_pipeline(p1)
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

        def shader_run(nsteps):
            device.queue.write_buffer(stateA, 0, seed.reshape(-1).tobytes())
            cur = run_steps(masks[:nsteps], 0)
            got = read_back(device, stateB if cur else stateA, state_nbytes)
            return got.reshape(1, C, grid, grid)

        # 1 step is the definitive check: the CA is chaotic, so a bare fp32
        # rounding difference amplifies ~1.1-1.2x per step over long runs.
        ref1 = onnx_step(seed, dense_masks[0])
        d1 = float(np.max(np.abs(shader_run(1) - ref1)))
        print(f"correctness vs ONNX: 1 step = {d1:.3e}  [{'ok' if d1 < 1e-4 else 'FAIL'}]")
        if args.steps > 1:
            ref = ref1
            for m in dense_masks[1:]:
                ref = onnx_step(ref, m)
            dn = float(np.max(np.abs(shader_run(args.steps) - ref)))
            print(f"                    {args.steps} steps = {dn:.3e}  (amplified by chaos)")

        device.queue.write_buffer(stateA, 0, seed.reshape(-1).tobytes())

    # ---- benchmark ----------------------------------------------------------
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        run_steps(masks, 0)
        read_back(device, stateA, state_nbytes)  # drain
        times.append((time.perf_counter() - t0) / args.steps * 1000)
    best = min(times)
    print(f"grid {grid}x{grid} · {args.steps} steps · {best:.2f} ms/step best of {args.runs} "
          f"({1000 / best:.0f} steps/s)")


if __name__ == "__main__":
    main()
