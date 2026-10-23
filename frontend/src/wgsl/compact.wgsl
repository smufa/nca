// Work-skipping, pass A: make the next state a copy of the old one for every cell,
// then append the cells that will actually update to a work list.
//
// The point is that the update rule (`step_list.wgsl`) only runs for the listed cells.
// At fire_rate 0.5 that is half the grid, and the MLP is ~73% of the step.
//
// `out` is written for every cell (a straight copy) because pass2 (the living
// mask) reads the whole grid; that copy is the state write-back the baseline
// pass1 already pays for.

override C: u32 = 16u;

struct Params {
  C: u32, H: u32, W: u32, hidden: u32,
  w1: u32, b1: u32, w2: u32, b2: u32,
  step: f32,
  wmid: u32,
};
fn sidx(c: u32, y: u32, x: u32) -> u32 { return (c * p.H + y) * p.W + x; }

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<storage, read> fire: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read_write> list: array<u32>;
@group(0) @binding(4) var<storage, read_write> count: atomic<u32>;
@group(0) @binding(5) var<uniform> p: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= p.W || y >= p.H) { return; }

  for (var c: u32 = 0u; c < C; c = c + 1u) {
    out[sidx(c, y, x)] = state[sidx(c, y, x)];
  }
  if (fire[y * p.W + x] >= 0.5) {
    let i = atomicAdd(&count, 1u);
    list[i] = y * p.W + x;
  }
}
