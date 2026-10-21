struct Brush {
  H: u32, W: u32,
  cx: i32, cy: i32, radius: i32, op: u32,
  factor: f32, mask: u32,
  pad0: u32, pad1: u32,
};
@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<uniform> b: Brush;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= b.W || gid.y >= b.H) { return; }
  let dx = i32(gid.x) - b.cx;
  let dy = i32(gid.y) - b.cy;
  if (dx * dx + dy * dy > b.radius * b.radius) { return; }
  let i = gid.y * b.W + gid.x;
  let plane = b.H * b.W;
  for (var c: u32 = 0u; c < 16u; c = c + 1u) {
    if ((b.mask & (1u << c)) == 0u) { continue; }
    let idx = c * plane + i;
    if (b.op == 0u) { state[idx] = 1.0; }
    else if (b.op == 1u) { state[idx] = 0.0; }
    else { state[idx] = state[idx] * b.factor; }
  }
}
