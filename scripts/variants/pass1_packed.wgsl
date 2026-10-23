// EXPERIMENT (Phase 6a): packed state layout, `[H*W][C/4]` vec4 texels.
//
// Same maths as `frontend/src/wgsl/pass1.wgsl`, but the state tensor is stored
// channel-interleaved instead of channel-planar: one cell is `C/4` consecutive
// vec4 texels instead of `C` f32s at stride `H*W`. Perception therefore reads 9
// `vec4` per channel group (36 loads at C=16) instead of 9 * 16 scalar loads, and
// a cell's own state is 4 contiguous loads rather than 16 strided ones.
//
// Constraints: C % 4 == 0, one hidden layer (entry point `step`). The perception
// accumulators are vec4 (one lane per channel in the group), and the kernel is
// shared by all channels, so `s4 * kernel[k]` is the whole tap.
//
// Only useful if the bench says so -- see docs/findings/shader-perf.md.

override C: u32 = 16u;
override NP4: u32 = 12u;
override W1S4: u32 = 12u;
override HG: u32 = 32u;
override CONDN: u32 = 0u;
override LH: u32 = 1u;  // accepted so the bench's constant set applies; unused

const HMAX: u32 = 64u;

struct Params {
  C: u32, H: u32, W: u32, hidden: u32,
  w1: u32, b1: u32, w2: u32, b2: u32,
  step: f32,
  wmid: u32,
};

@group(0) @binding(0) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> fire: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> p: Params;
@group(0) @binding(6) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> cond: array<f32>;


@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= p.W || y >= p.H) { return; }
  let C4 = C / 4u;  // vec4 groups per cell (an override is not a const-expression)

  var perc: array<f32, 64>;
  for (var i: u32 = 3u * C + CONDN; i < 4u * NP4; i = i + 1u) { perc[i] = 0.0; }

  // Perception: one vec4 load per (channel group, tap), shared kernel weights.
  let row4 = C4 * p.W;
  for (var q: u32 = 0u; q < C4; q = q + 1u) {
    var ai = vec4<f32>(0.0);
    var ax = vec4<f32>(0.0);
    var ay = vec4<f32>(0.0);
    for (var dy: u32 = 0u; dy < 3u; dy = dy + 1u) {
      let yy = (y + dy + p.H - 1u) % p.H;
      for (var dx: u32 = 0u; dx < 3u; dx = dx + 1u) {
        let xx = (x + dx + p.W - 1u) % p.W;
        let s = state[(yy * p.W + xx) * C4 + q];
        let k = dy * 3u + dx;
        ai = ai + s * kernel[k];
        ax = ax + s * kernel[9u + k];
        ay = ay + s * kernel[18u + k];
      }
    }
    let c = q * 4u;
    perc[c * 3u] = ai.x;
    perc[c * 3u + 1u] = ax.x;
    perc[c * 3u + 2u] = ay.x;
    perc[(c + 1u) * 3u] = ai.y;
    perc[(c + 1u) * 3u + 1u] = ax.y;
    perc[(c + 1u) * 3u + 2u] = ay.y;
    perc[(c + 2u) * 3u] = ai.z;
    perc[(c + 2u) * 3u + 1u] = ax.z;
    perc[(c + 2u) * 3u + 2u] = ay.z;
    perc[(c + 3u) * 3u] = ai.w;
    perc[(c + 3u) * 3u + 1u] = ax.w;
    perc[(c + 3u) * 3u + 2u] = ay.w;
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
    // 4-way ILP over the k loop (same as pass1.wgsl) and 4-way over the group.
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

  // Packed write-back: one cell is C4 contiguous vec4 texels.
  let cell = (y * p.W + x) * C4;
  let f = fire[y * p.W + x];
  for (var q: u32 = 0u; q < C4; q = q + 1u) {
    let s = state[cell + q];
    let c = q * 4u;
    let d = vec4<f32>(dxv[c], dxv[c + 1u], dxv[c + 2u], dxv[c + 3u]);
    out[cell + q] = s + d * f * p.step;
  }
}
