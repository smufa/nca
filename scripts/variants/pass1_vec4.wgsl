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
@group(0) @binding(6) var<storage, read> w4: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= p.W || y >= p.H) { return; }

  var perc: array<f32, 48>;
  for (var c: u32 = 0u; c < 16u; c = c + 1u) {
    var ki = 0.0;
    var kx = 0.0;
    var ky = 0.0;
    for (var dy: u32 = 0u; dy < 3u; dy = dy + 1u) {
      for (var dx: u32 = 0u; dx < 3u; dx = dx + 1u) {
        let yy = (y + dy + p.H - 1u) % p.H;
        let xx = (x + dx + p.W - 1u) % p.W;
        let s = state[sidx(c, yy, xx)];
        let k = dy * 3u + dx;
        ki = ki + s * kernel[k];
        kx = kx + s * kernel[9u + k];
        ky = ky + s * kernel[18u + k];
      }
    }
    perc[c * 3u] = ki;
    perc[c * 3u + 1u] = kx;
    perc[c * 3u + 2u] = ky;
  }

  // pack the 48 perception values into 12 vec4s
  var perc4: array<vec4<f32>, 12>;
  for (var k: u32 = 0u; k < 12u; k = k + 1u) {
    let b = k * 4u;
    perc4[k] = vec4<f32>(perc[b], perc[b + 1u], perc[b + 2u], perc[b + 3u]);
  }

  // update rule with vec4 weight loads (4x fewer loads) + 4-way ILP
  var dxv: array<f32, 16>;
  for (var c: u32 = 0u; c < 16u; c = c + 1u) { dxv[c] = weights[p.b2 + c]; }
  for (var j: u32 = 0u; j < p.hidden; j = j + 1u) {
    var acc = vec4<f32>(0.0);
    let wj = j * 12u;
    for (var k: u32 = 0u; k < 12u; k = k + 1u) {
      acc = acc + w4[wj + k] * perc4[k];
    }
    let h = max(acc.x + acc.y + acc.z + acc.w + weights[p.b1 + j], 0.0);
    for (var c: u32 = 0u; c < 16u; c = c + 1u) {
      dxv[c] = dxv[c] + weights[p.w2 + c * p.hidden + j] * h;
    }
  }

  let f = fire[y * p.W + x];
  for (var c: u32 = 0u; c < 16u; c = c + 1u) {
    out[sidx(c, y, x)] = state[sidx(c, y, x)] + dxv[c] * f * p.step;
  }
}
