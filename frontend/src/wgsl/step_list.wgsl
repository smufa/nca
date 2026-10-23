// Work-skipping, pass C: the update rule for one listed cell per thread.
//
// This is pass1's `step` entry with three changes, and nothing else may differ --
// the parity check compares it against the baseline pipeline:
//   * the cell index comes from the work list instead of global_invocation_id,
//   * fire is implicit (only firing cells are in the list),
//   * the write-back adds `step` without multiplying by a fire value.
//
// 64 threads per workgroup because the list is 1-D; `finalize.wgsl` converts
// the list length into indirect dispatch args of ceil(n/64).

override C: u32 = 16u;
override NP4: u32 = 12u;
override W1S4: u32 = 12u;
override HG: u32 = 32u;
override CONDN: u32 = 0u;

const HMAX: u32 = 64u;

struct Params {
  C: u32, H: u32, W: u32, hidden: u32,
  w1: u32, b1: u32, w2: u32, b2: u32,
  step: f32,
  wmid: u32,
};
fn sidx(c: u32, y: u32, x: u32) -> u32 { return (c * p.H + y) * p.W + x; }

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<storage, read> list: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;
@group(0) @binding(6) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> cond: array<f32>;
@group(0) @binding(8) var<storage, read> count: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= count[0]) { return; }
  let cell = list[j];
  let x = cell % p.W;
  let y = cell / p.W;

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
  for (var i: u32 = 0u; i < CONDN; i = i + 1u) { perc[3u * C + i] = cond[i]; }

  var perc4: array<vec4<f32>, 16>;
  for (var k: u32 = 0u; k < NP4; k = k + 1u) {
    let b = k * 4u;
    perc4[k] = vec4<f32>(perc[b], perc[b + 1u], perc[b + 2u], perc[b + 3u]);
  }

  var dxv: array<f32, 16>;
  for (var c: u32 = 0u; c < C; c = c + 1u) { dxv[c] = weights[p.b2 + c]; }
  let w2v = p.w2 / 4u;

  if (NP4 % 4u == 0u) {
    for (var jg: u32 = 0u; jg < HG; jg = jg + 1u) {
      var a0 = vec4<f32>(0.0);
      var a1 = vec4<f32>(0.0);
      var a2 = vec4<f32>(0.0);
      var a3 = vec4<f32>(0.0);
      let base = jg * 4u * W1S4;
      for (var k: u32 = 0u; k < NP4; k = k + 4u) {
        let p0 = perc4[k];
        let p1 = perc4[k + 1u];
        let p2 = perc4[k + 2u];
        let p3 = perc4[k + 3u];
        a0 = a0 + vec4<f32>(
          dot(w4[base + k], p0),
          dot(w4[base + W1S4 + k], p0),
          dot(w4[base + 2u * W1S4 + k], p0),
          dot(w4[base + 3u * W1S4 + k], p0),
        );
        a1 = a1 + vec4<f32>(
          dot(w4[base + k + 1u], p1),
          dot(w4[base + W1S4 + k + 1u], p1),
          dot(w4[base + 2u * W1S4 + k + 1u], p1),
          dot(w4[base + 3u * W1S4 + k + 1u], p1),
        );
        a2 = a2 + vec4<f32>(
          dot(w4[base + k + 2u], p2),
          dot(w4[base + W1S4 + k + 2u], p2),
          dot(w4[base + 2u * W1S4 + k + 2u], p2),
          dot(w4[base + 3u * W1S4 + k + 2u], p2),
        );
        a3 = a3 + vec4<f32>(
          dot(w4[base + k + 3u], p3),
          dot(w4[base + W1S4 + k + 3u], p3),
          dot(w4[base + 2u * W1S4 + k + 3u], p3),
          dot(w4[base + 3u * W1S4 + k + 3u], p3),
        );
      }
      let hs = max(
        a0 + a1 + a2 + a3 + vec4<f32>(
          weights[p.b1 + jg * 4u], weights[p.b1 + jg * 4u + 1u],
          weights[p.b1 + jg * 4u + 2u], weights[p.b1 + jg * 4u + 3u]),
        vec4<f32>(0.0),
      );
      for (var c: u32 = 0u; c < C; c = c + 1u) {
        dxv[c] = dxv[c] + dot(w4[w2v + c * HG + jg], hs);
      }
    }
  } else {
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
  }

  for (var c: u32 = 0u; c < C; c = c + 1u) {
    out[sidx(c, y, x)] = state[sidx(c, y, x)] + dxv[c] * p.step;
  }
}
