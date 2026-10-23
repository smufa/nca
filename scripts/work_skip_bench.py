"""Does skipping the update rule for non-firing cells pay off?

At `fire_rate 0.5` half of every grid is not updated, yet `pass1.wgsl` computes the
full MLP for every cell and multiplies the result by zero. This measures a
work-list implementation against the shipped two-pass pipeline:

  baseline     pass1 (perception + MLP + fire mix, every cell) -> pass2 (living mask)
  work-skip    compact (copy + list the firing cells) -> finalize (indirect args)
               -> MLP over the list (indirect dispatch) -> pass2 (living mask)

Both run the same maths, so the parity check must be exact (the MLP code is
copied from pass1 verbatim; only the cell index, the fire multiply and the
workgroup size differ). An 11% slower branch-based version already exists --
see docs/findings/shader-perf.md -- the interesting question is whether
compaction avoids the divergence cost.

Usage: bash scripts/bench.sh work-skip crab.regenerating   (.venv/bin/python works
if it has wgpu; the NixOS loader env is only needed for the adapter)
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np
import wgpu

ROOT = Path(__file__).resolve().parents[1]
# The shaders under test are the ones the engine ships (`ShaderRunner` runs the same
# three passes), so this measures production code, not a copy of it.
WGSL_DIR = ROOT / "frontend" / "src" / "wgsl"


def canonical_kernels() -> np.ndarray:
    """Identity, Sobel-x, Sobel-y -- the perception the ONNX graph is exported with."""
    ident = np.zeros(9, np.float32)
    ident[4] = 1.0
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32).reshape(-1)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32).reshape(-1)
    return np.concatenate([ident, sobel_x, sobel_y])


def perception_weight(C: int) -> np.ndarray:
    """[3C, 1, 3, 3]: per channel identity / dx / dy, as the exporter writes it."""
    ident = np.zeros(9, np.float32)
    ident[4] = 1.0
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32).reshape(-1)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32).reshape(-1)
    w = np.zeros((3 * C, 1, 3, 3), np.float32)
    for c in range(C):
        w[c] = ident.reshape(3, 3)
        w[C + c] = sobel_x.reshape(3, 3)
        w[2 * C + c] = sobel_y.reshape(3, 3)
    return w


def make_seed(C: int, grid: int) -> np.ndarray:
    x = np.zeros((1, C, grid, grid), np.float32)
    cx, cy = grid // 2, grid // 2
    x[0, 3, cy, cx] = 1.0
    x[0, :3, cy, cx] = 1.0
    return x


def read_back(device, src, nbytes: int) -> np.ndarray:
    staging = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )
    enc = device.create_command_encoder()
    enc.copy_buffer_to_buffer(src, 0, staging, 0, nbytes)
    device.queue.submit([enc.finish()])
    staging.map_sync(wgpu.MapMode.READ, 0, nbytes)
    out = np.frombuffer(staging.read_mapped(0, nbytes), dtype=np.float32).copy()
    staging.unmap()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=str, help="export id, e.g. crab.regenerating")
    ap.add_argument("--grids", type=str, default="72,256")
    ap.add_argument("--steps", type=int, default=40, help="steps per timed repetition")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--parity-steps", type=int, default=20)
    ap.add_argument("--fire-rate", type=float, default=None,
                    help="probability that a cell updates (default: meta fire_rate; "
                         "1.0 = every cell, the no-savings control)")
    args = ap.parse_args()

    out_dir = ROOT / "exports" / args.model
    meta = json.loads((out_dir / "meta.json").read_text())
    C = int(meta["channel_n"])
    hidden = int(meta["weights"]["hidden_n"])
    layers = int(meta["weights"].get("hidden_layers", 1))
    cond_n = int(meta.get("cond_n", 0) or 0)
    fire_rate = args.fire_rate if args.fire_rate is not None else float(meta.get("fire_rate", 0.5))
    off = meta["weights"]["offsets"]
    weights = np.fromfile(out_dir / "weights.bin", dtype="<f4")
    kernel = canonical_kernels()

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:  # pragma: no cover
        sys.exit("no adapter; run through scripts/bench.sh for the Vulkan loader env")
    device = adapter.request_device_sync(required_features=[])
    print(f"adapter: {adapter.info['device']} ({adapter.info['adapter_type']})")
    print(f"model {args.model}: C={C} hidden={hidden} layers={layers} "
          f"fire_rate={fire_rate} ({fire_rate * 100:.0f}% of cells update)")

    def make_layout(spec):
        return device.create_bind_group_layout(
            entries=[
                {"binding": b, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": kind}}
                for b, kind in enumerate(spec)
            ]
        )

    w1s4 = -(-(3 * C + cond_n) // 4)
    consts = {"C": C, "NP4": w1s4, "W1S4": w1s4, "HG": hidden // 4, "CONDN": cond_n}
    consts_deep = dict(consts, LH=layers)

    def patch(src: str) -> str:
        return src.replace("HMAX: u32 = 64u", f"HMAX: u32 = {max(hidden // 4, 1)}u")

    def pipe(src: str, entry: str, layout, constants):
        return device.create_compute_pipeline(
            layout=layout,
            compute={
                "module": device.create_shader_module(code=src),
                "entry_point": entry,
                "constants": constants,
            },
        )

    # ---- baseline: the shipped pipelines ------------------------------------
    layout1 = make_layout([
        "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage",
        "storage", "uniform", "read-only-storage", "read-only-storage",
    ])
    layout2 = make_layout(["read-only-storage", "read-only-storage", "storage", "uniform"])
    pl1 = device.create_pipeline_layout(bind_group_layouts=[layout1])
    pl2 = device.create_pipeline_layout(bind_group_layouts=[layout2])
    base1 = pipe(
        patch((WGSL_DIR / "pass1.wgsl").read_text()),
        "step_deep" if layers > 1 else "step",
        pl1,
        consts_deep,
    )
    base2 = pipe((WGSL_DIR / "pass2.wgsl").read_text(), "main", pl2, {"C": C})

    # ---- work-skip ----------------------------------------------------------
    layout_c = make_layout([
        "read-only-storage", "read-only-storage", "storage", "storage", "storage", "uniform",
    ])
    layout_f = make_layout(["read-only-storage", "storage"])
    layout_m = make_layout([
        "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage",
        "storage", "uniform", "read-only-storage", "read-only-storage", "read-only-storage",
    ])
    pipe_c = pipe((WGSL_DIR / "compact.wgsl").read_text(), "main",
                  device.create_pipeline_layout(bind_group_layouts=[layout_c]), {"C": C})
    pipe_f = pipe((WGSL_DIR / "finalize.wgsl").read_text(), "main",
                  device.create_pipeline_layout(bind_group_layouts=[layout_f]), {})
    pipe_m = pipe(
        patch((WGSL_DIR / "step_list.wgsl").read_text()),
        "main",
        device.create_pipeline_layout(bind_group_layouts=[layout_m]),
        consts,
    )
    if layers > 1:
        sys.exit("work-skip variant implements the shallow (LH=1) update rule only")

    def bench_grid(grid: int) -> dict:
        plane = grid * grid
        state_nbytes = C * plane * 4
        seed = make_seed(C, grid)
        S = wgpu.BufferUsage.STORAGE
        D = wgpu.BufferUsage.COPY_DST

        def sbuf(data, usage=S | D):
            b = device.create_buffer(size=max(len(data.tobytes()), 16), usage=usage)
            device.queue.write_buffer(b, 0, data.tobytes())
            return b

        sa = sbuf(seed.reshape(-1), S | D | wgpu.BufferUsage.COPY_SRC)
        sb = sbuf(np.zeros(C * plane, np.float32), S | D | wgpu.BufferUsage.COPY_SRC)
        tmp = sbuf(np.zeros(C * plane, np.float32))
        fire = sbuf(np.ones(plane, np.float32))
        wbuf = sbuf(weights)
        kbuf = sbuf(kernel)
        cvals = np.zeros(max(cond_n, 1), np.float32)
        cvals[0] = 1.0
        cbuf = sbuf(cvals)
        lst = sbuf(np.zeros(plane, np.uint32))
        cnt = sbuf(np.zeros(4, np.uint32))
        ind = device.create_buffer(size=16, usage=S | D | wgpu.BufferUsage.INDIRECT)

        params = bytearray(48)
        struct.pack_into("<8I", params, 0, C, grid, grid, hidden,
                         off["w1"], off["b1"], off["w2"], off["b2"])
        struct.pack_into("<f", params, 32, 1.0)
        struct.pack_into("<I", params, 36, off.get("wmid", 0))
        pbuf = device.create_buffer_with_data(
            data=bytes(params), usage=wgpu.BufferUsage.UNIFORM | D
        )

        def bg1(state):
            return device.create_bind_group(layout=layout1, entries=[
                {"binding": 0, "resource": {"buffer": state}},
                {"binding": 1, "resource": {"buffer": fire}},
                {"binding": 2, "resource": {"buffer": wbuf}},
                {"binding": 3, "resource": {"buffer": kbuf}},
                {"binding": 4, "resource": {"buffer": tmp}},
                {"binding": 5, "resource": {"buffer": pbuf}},
                {"binding": 6, "resource": {"buffer": wbuf}},
                {"binding": 7, "resource": {"buffer": cbuf}},
            ])

        def bg2(old, nxt):
            return device.create_bind_group(layout=layout2, entries=[
                {"binding": 0, "resource": {"buffer": old}},
                {"binding": 1, "resource": {"buffer": tmp}},
                {"binding": 2, "resource": {"buffer": nxt}},
                {"binding": 3, "resource": {"buffer": pbuf}},
            ])

        def bg_c(state):
            return device.create_bind_group(layout=layout_c, entries=[
                {"binding": 0, "resource": {"buffer": state}},
                {"binding": 1, "resource": {"buffer": fire}},
                {"binding": 2, "resource": {"buffer": tmp}},
                {"binding": 3, "resource": {"buffer": lst}},
                {"binding": 4, "resource": {"buffer": cnt}},
                {"binding": 5, "resource": {"buffer": pbuf}},
            ])

        bg_f = device.create_bind_group(layout=layout_f, entries=[
            {"binding": 0, "resource": {"buffer": cnt}},
            {"binding": 1, "resource": {"buffer": ind}},
        ])

        def bg_m(state):
            return device.create_bind_group(layout=layout_m, entries=[
                {"binding": 0, "resource": {"buffer": state}},
                {"binding": 1, "resource": {"buffer": lst}},
                {"binding": 2, "resource": {"buffer": wbuf}},
                {"binding": 3, "resource": {"buffer": kbuf}},
                {"binding": 4, "resource": {"buffer": tmp}},
                {"binding": 5, "resource": {"buffer": pbuf}},
                {"binding": 6, "resource": {"buffer": wbuf}},
                {"binding": 7, "resource": {"buffer": cbuf}},
                {"binding": 8, "resource": {"buffer": cnt}},
            ])

        b1 = [bg1(sa), bg1(sb)]
        b2 = [bg2(sa, sb), bg2(sb, sa)]
        bc = [bg_c(sa), bg_c(sb)]
        bm = [bg_m(sa), bg_m(sb)]
        dispatch = ((grid + 7) // 8, (grid + 7) // 8)

        def run(masks, cur: int, mode: str) -> int:
            for m in masks:
                device.queue.write_buffer(fire, 0, m.tobytes())
                enc = device.create_command_encoder()
                if mode == "baseline":
                    p = enc.begin_compute_pass()
                    p.set_pipeline(base1)
                    p.set_bind_group(0, b1[cur])
                    p.dispatch_workgroups(*dispatch)
                    p.end()
                    p = enc.begin_compute_pass()
                    p.set_pipeline(base2)
                    p.set_bind_group(0, b2[cur])
                    p.dispatch_workgroups(*dispatch)
                    p.end()
                else:
                    # the counter is the only per-step reset (a 4-byte clear)
                    enc.clear_buffer(cnt, 0, 4)
                    p = enc.begin_compute_pass()
                    p.set_pipeline(pipe_c)
                    p.set_bind_group(0, bc[cur])
                    p.dispatch_workgroups(*dispatch)
                    p.end()
                    p = enc.begin_compute_pass()
                    p.set_pipeline(pipe_f)
                    p.set_bind_group(0, bg_f)
                    p.dispatch_workgroups(1, 1, 1)
                    p.end()
                    p = enc.begin_compute_pass()
                    p.set_pipeline(pipe_m)
                    p.set_bind_group(0, bm[cur])
                    p.dispatch_workgroups_indirect(ind, 0)
                    p.end()
                    p = enc.begin_compute_pass()
                    p.set_pipeline(base2)
                    p.set_bind_group(0, b2[cur])
                    p.dispatch_workgroups(*dispatch)
                    p.end()
                device.queue.submit([enc.finish()])
                cur ^= 1
            return cur

        rng = np.random.default_rng(0)
        masks = [(rng.random(plane) < fire_rate).astype(np.float32)
                 for _ in range(args.steps)]
        parity_masks = masks[: args.parity_steps]

        # ---- correctness: identical maths, so anything non-zero is a bug -------
        def final_state(nsteps: int, mode: str) -> np.ndarray:
            device.queue.write_buffer(sa, 0, seed.reshape(-1).tobytes())
            cur = run(parity_masks[:nsteps], 0, mode)
            got = read_back(device, sb if cur == 0 else sa, state_nbytes)
            return got.reshape(C, grid, grid)

        ref = final_state(args.parity_steps, "baseline")
        got = final_state(args.parity_steps, "work-skip")
        d1 = float(np.max(np.abs(got - ref)))
        live = int((ref[3] > 0.1).sum())
        print(f"parity {args.parity_steps} steps vs baseline: max|d| = {d1:.3e} "
              f"[{'ok' if d1 < 1e-6 else 'MISMATCH'}]  ({live} live cells)")

        # ---- timing ------------------------------------------------------------
        def timed(mode: str) -> float:
            device.queue.write_buffer(sa, 0, seed.reshape(-1).tobytes())
            t0 = time.perf_counter()
            run(masks, 0, mode)
            read_back(device, sa, state_nbytes)  # drain
            return (time.perf_counter() - t0) / len(masks) * 1000

        samples = {"baseline": [], "work-skip": []}
        for _ in range(args.rounds):
            for mode in ("baseline", "work-skip"):
                samples[mode].append(timed(mode))

        rows = {}
        for mode, s in samples.items():
            best, med, mean = min(s), statistics.median(s), statistics.mean(s)
            rows[mode] = best
            print(f"grid {grid}x{grid} · {mode:<10} · best {best:6.2f} · median {med:6.2f} "
                  f"· mean {mean:6.2f} ms/step ({1000 / med:5.0f} steps/s)")
        print(f"grid {grid}x{grid} · speedup (best/median): "
              f"{rows['baseline'] / rows['work-skip']:.3f}x / "
              f"{statistics.median(samples['baseline']) / statistics.median(samples['work-skip']):.3f}x")
        return rows

    for g in (int(x) for x in args.grids.split(",")):
        bench_grid(g)


if __name__ == "__main__":
    main()
