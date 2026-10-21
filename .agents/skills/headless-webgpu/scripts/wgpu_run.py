"""Minimal headless WGSL compute-shader skeleton (wgpu-py).

Squares each element of a float32 buffer, checks it against numpy, and times it.
Copy this file and edit SHADER + the buffers for your own kernel.

Run (NixOS Vulkan env is required - see the headless-webgpu skill's SKILL.md):

    LOADER_DIR=$(dirname "$(ls /nix/store/*vulkan-loader*/lib/libvulkan.so.1 | head -1)")
    LD_LIBRARY_PATH="$LOADER_DIR:/run/opengl-driver/lib" \
    VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/intel_icd.x86_64.json \
    uv run --with wgpu --with numpy python wgpu_run.py
"""

from __future__ import annotations

import time

import numpy as np
import wgpu

N = 1 << 20

SHADER = """
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
struct P { n: u32 };
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n) { return; }
  dst[gid.x] = src[gid.x] * src[gid.x];
}
"""


def main() -> None:
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    print("adapter:", adapter.info["device"], f"({adapter.info['adapter_type']})")
    device = adapter.request_device_sync()

    src = np.arange(N, dtype=np.float32)
    usage = wgpu.BufferUsage

    src_buf = device.create_buffer_with_data(data=src.tobytes(), usage=usage.STORAGE)
    # any buffer you copy FROM needs COPY_SRC
    dst_buf = device.create_buffer(size=src.nbytes, usage=usage.STORAGE | usage.COPY_SRC)
    params_buf = device.create_buffer_with_data(
        data=np.array([N], np.uint32).tobytes(), usage=usage.UNIFORM | usage.COPY_DST
    )

    module = device.create_shader_module(code=SHADER)
    # explicit layout: `layout="auto"` omits unused bindings and breaks bind groups
    layout = device.create_bind_group_layout(
        entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
        ]
    )
    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[layout]),
        compute={"module": module, "entry_point": "main"},
    )
    bind = device.create_bind_group(
        layout=layout,
        entries=[
            {"binding": 0, "resource": {"buffer": src_buf}},
            {"binding": 1, "resource": {"buffer": dst_buf}},
            {"binding": 2, "resource": {"buffer": params_buf}},
        ],
    )

    def run() -> None:
        enc = device.create_command_encoder()
        p = enc.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, bind)
        p.dispatch_workgroups((N + 63) // 64, 1, 1)
        p.end()
        device.queue.submit([enc.finish()])

    def readback() -> np.ndarray:
        staging = device.create_buffer(size=src.nbytes, usage=usage.MAP_READ | usage.COPY_DST)
        enc = device.create_command_encoder()
        enc.copy_buffer_to_buffer(dst_buf, 0, staging, 0, src.nbytes)
        device.queue.submit([enc.finish()])
        # no device.poll() in wgpu-py: map_sync blocks until ready
        staging.map_sync(mode=wgpu.MapMode.READ)
        out = np.frombuffer(staging.read_mapped(), dtype=np.float32).copy()
        staging.unmap()
        staging.destroy()
        return out

    run()
    out = readback()
    diff = float(np.max(np.abs(out - src * src)))
    print(f"correctness: max|diff| = {diff:.3e}  [{'ok' if diff < 1e-5 else 'FAIL'}]")

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        run()
        readback()  # drain the queue so we time execution, not submission
        times.append((time.perf_counter() - t0) * 1e3)
    print(f"best of 5: {min(times):.3f} ms")


if __name__ == "__main__":
    main()
