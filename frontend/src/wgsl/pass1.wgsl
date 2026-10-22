// Perception + update-rule step.
//
// C, NP4, W1S4, HG and CONDN are pipeline-overridable constants (set from the
// model's channel count / hidden width / one-hot width at pipeline creation).
// They MUST stay compile-time constants: with dynamic bounds naga can no longer
// unroll the MLP loops and the step becomes ~1.9x slower. Defaults are the paper
// size (C=16, hidden=128, no conditioning).
//
// A conditional model appends a CONDN-wide one-hot to every cell's perception,
// so the vector fed to update[0] is [3C perception | CONDN one-hot], padded up to
// a multiple of 4.

override C: u32 = 16u;     // state channels
override NP4: u32 = 12u;   // ceil((3C + CONDN) / 4): vec4 groups in the padded perception
override W1S4: u32 = 12u;  // vec4 stride of one W1 row (== NP4)
override HG: u32 = 32u;    // hidden groups: hidden / 4
override CONDN: u32 = 0u;  // one-hot selector width (0 = unconditional)

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
@group(0) @binding(7) var<storage, read> cond: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= p.W || y >= p.H) { return; }

  // 64 = 3*16 + 16 is the largest possible perception. Everything in
  // [0, 4*NP4) is written below: perception, the one-hot, then zero padding.
  var perc: array<f32, 64>;
  for (var i: u32 = 3u * C + CONDN; i < 4u * NP4; i = i + 1u) { perc[i] = 0.0; }
  for (var c: u32 = 0u; c < C; c = c + 1u) {
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
  // The one-hot is constant across the grid; it tells the rule which target to grow.
  for (var i: u32 = 0u; i < CONDN; i = i + 1u) { perc[3u * C + i] = cond[i]; }

  var perc4: array<vec4<f32>, 16>;
  for (var k: u32 = 0u; k < NP4; k = k + 1u) {
    let b = k * 4u;
    perc4[k] = vec4<f32>(perc[b], perc[b + 1u], perc[b + 2u], perc[b + 3u]);
  }

  // four hidden units at a time: vec4 loads for both weight matrices
  var dxv: array<f32, 16>;
  for (var c: u32 = 0u; c < C; c = c + 1u) { dxv[c] = weights[p.b2 + c]; }
  let w2v = p.w2 / 4u;
  for (var jg: u32 = 0u; jg < HG; jg = jg + 1u) {
    var acc = vec4<f32>(0.0);
    let base = jg * 4u * W1S4;
    for (var k: u32 = 0u; k < NP4; k = k + 1u) {
      let pv = perc4[k];
      acc = acc + vec4<f32>(
        dot(w4[base + k], pv),
        dot(w4[base + W1S4 + k], pv),
        dot(w4[base + 2u * W1S4 + k], pv),
        dot(w4[base + 3u * W1S4 + k], pv),
      );
    }
    let hs = max(
      acc + vec4<f32>(
        weights[p.b1 + jg * 4u], weights[p.b1 + jg * 4u + 1u],
        weights[p.b1 + jg * 4u + 2u], weights[p.b1 + jg * 4u + 3u]),
      vec4<f32>(0.0),
    );
    for (var c: u32 = 0u; c < C; c = c + 1u) {
      dxv[c] = dxv[c] + dot(w4[w2v + c * HG + jg], hs);
    }
  }

  let f = fire[y * p.W + x];
  for (var c: u32 = 0u; c < C; c = c + 1u) {
    out[sidx(c, y, x)] = state[sidx(c, y, x)] + dxv[c] * f * p.step;
  }
}
