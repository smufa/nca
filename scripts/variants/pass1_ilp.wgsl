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

  var dxv: array<f32, 16>;
  for (var c: u32 = 0u; c < 16u; c = c + 1u) { dxv[c] = weights[p.b2 + c]; }
  for (var j: u32 = 0u; j < p.hidden; j = j + 1u) {
    // four independent partial sums to break the FMA dependency chain
    let wb = p.w1 + j * 48u;
    var h0 = 0.0;
    var h1 = 0.0;
    var h2 = 0.0;
    var h3 = 0.0;
    for (var i: u32 = 0u; i < 48u; i = i + 4u) {
      h0 = h0 + weights[wb + i] * perc[i];
      h1 = h1 + weights[wb + i + 1u] * perc[i + 1u];
      h2 = h2 + weights[wb + i + 2u] * perc[i + 2u];
      h3 = h3 + weights[wb + i + 3u] * perc[i + 3u];
    }
    let h = max(h0 + h1 + h2 + h3 + weights[p.b1 + j], 0.0);
    for (var c: u32 = 0u; c < 16u; c = c + 1u) {
      dxv[c] = dxv[c] + weights[p.w2 + c * p.hidden + j] * h;
    }
  }

  let f = fire[y * p.W + x];
  for (var c: u32 = 0u; c < 16u; c = c + 1u) {
    out[sidx(c, y, x)] = state[sidx(c, y, x)] + dxv[c] * f * p.step;
  }
}
