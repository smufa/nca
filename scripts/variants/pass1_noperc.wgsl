struct Params {
  C: u32, H: u32, W: u32, hidden: u32,
  w1: u32, b1: u32, w2: u32, b2: u32,
  step: f32,
};
fn sidx(c: u32, y: u32, x: u32) -> u32 { return (c * p.H + y) * p.W + x; }

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<storage, read> fire: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= p.W || y >= p.H) { return; }

  // perception removed: constant perc (MLP still runs)
  var perc: array<f32, 48>;
  for (var i: u32 = 0u; i < 48u; i = i + 1u) { perc[i] = 0.5 + f32(i) * 0.01; }

  var dxv: array<f32, 16>;
  for (var c: u32 = 0u; c < 16u; c = c + 1u) { dxv[c] = weights[p.b2 + c]; }
  for (var j: u32 = 0u; j < p.hidden; j = j + 1u) {
    var h = weights[p.b1 + j];
    for (var i: u32 = 0u; i < 48u; i = i + 1u) {
      h = h + weights[p.w1 + j * 48u + i] * perc[i];
    }
    h = max(h, 0.0);
    for (var c: u32 = 0u; c < 16u; c = c + 1u) {
      dxv[c] = dxv[c] + weights[p.w2 + c * p.hidden + j] * h;
    }
  }

  let f = fire[y * p.W + x];
  for (var c: u32 = 0u; c < 16u; c = c + 1u) {
    out[sidx(c, y, x)] = state[sidx(c, y, x)] + dxv[c] * f * p.step;
  }
}
