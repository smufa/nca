// pass1_tiled: the step with the *weights* staged in workgroup memory.
//
// The default pass1 re-reads the whole MLP for every cell: `HG * W1S4` vec4 of
// W1 plus `C * HG` vec4 of W2, i.e. ~32 kB per cell at the paper size. Those reads
// hit L1/L2, but they still cost issue slots and bandwidth per cell, which is why
// the step is quadratic in grid area.
//
// A workgroup covers 8x8 = 64 cells, so every weight vec4 is read 64 times from
// the same address in the same cycle anyway. Staging them once per workgroup in
// workgroup memory turns 64 global/L2 reads into 1 global read + 64 broadcasts:
//
//   W1: `CH` (32) hidden units at a time -> CH * W1S4 vec4 = 6 kB, barrier, reuse
//   b1: `CH` floats                                                   128 B
//   W2: all of it, C * HG vec4 (8 kB at C=16, hidden=128)                8 kB
//                                                    total at paper size 14.5 kB
//
// That must fit `maxComputeWorkgroupStorageSize` (16384 B by default in WebGPU),
// so this variant is limited to hidden <= 128 at C = 16 (hidden <= 192 at C = 10).
//
// Usage (sizes come from the model's meta, so keep the override declarations):
//   bash scripts/bench.sh <model> --reference \
//       --shader scripts/variants/pass1_tiled.wgsl --consts
//
// Not yet handled: LH > 1 (the deep path needs the whole hidden vector between
// layers; it is a second GEMM over the same staged weights and is a follow-up).

override C: u32 = 16u;     // state channels
override NP4: u32 = 12u;   // ceil((3C + CONDN) / 4)
override W1S4: u32 = 12u;  // vec4 stride of one W1 row (== NP4)
override HG: u32 = 32u;    // hidden groups: hidden / 4
override CONDN: u32 = 0u;  // one-hot selector width
override LH: u32 = 1u;     // must be 1 here

const HMAX: u32 = 64u;     // max hidden groups (hidden <= 256)
const CH: u32 = 32u;       // hidden units staged per chunk
const CHG: u32 = 8u;       // CH / 4: vec4 groups per chunk
const WG: u32 = 64u;       // threads per workgroup (8x8)

struct Params {
  C: u32, H: u32, W: u32, hidden: u32,
  w1: u32, b1: u32, w2: u32, b2: u32,
  step: f32,
  wmid: u32,
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

var<workgroup> w1s: array<vec4<f32>, CH * W1S4>;
var<workgroup> b1s: array<f32, CH>;
var<workgroup> w2s: array<vec4<f32>, C * HG>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  // Barriers below are workgroup-wide, so no early return: the last partial tile
  // of an odd grid must still take part, it just does not write.
  let inb = gid.x < p.W && gid.y < p.H;
  let x = min(gid.x, p.W - 1u);
  let y = min(gid.y, p.H - 1u);

  // ---- stage W2 (and later W1/b1) in workgroup memory ----------------------
  let w2v = p.w2 / 4u;
  for (var i: u32 = lid; i < C * HG; i = i + WG) { w2s[i] = w4[w2v + i]; }
  workgroupBarrier();

  // ---- perception (unchanged from pass1: 9 taps per channel) --------------
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

  // ---- MLP, one chunk of hidden units at a time ---------------------------
  // Every thread runs the same chunk sequence, so the staged weights are read by
  // all 64 lanes at the same address (a broadcast, not a bank conflict).
  for (var c0: u32 = 0u; c0 < HG * 4u; c0 = c0 + CH) {
    // `CH` (not min(CH, ...)): a runtime bound stops naga unrolling this loop,
    // which was worth more than the staging itself. Requires hidden % 32 == 0.
    let rows = CH;
    for (var i: u32 = lid; i < rows * W1S4; i = i + WG) {
      w1s[i] = w4[(p.w1 / 4u) + c0 * W1S4 + i];
    }
    for (var i: u32 = lid; i < rows; i = i + WG) { b1s[i] = weights[p.b1 + c0 + i]; }
    workgroupBarrier();

    let groups = CHG;
    for (var g: u32 = 0u; g < groups; g = g + 1u) {
      let base = g * 4u * W1S4;
      var acc = vec4<f32>(0.0);
      for (var k: u32 = 0u; k < NP4; k = k + 1u) {
        let pv = perc4[k];
        acc = acc + vec4<f32>(
          dot(w1s[base + k], pv),
          dot(w1s[base + W1S4 + k], pv),
          dot(w1s[base + 2u * W1S4 + k], pv),
          dot(w1s[base + 3u * W1S4 + k], pv),
        );
      }
      let b = g * 4u;
      let hs = max(
        acc + vec4<f32>(b1s[b], b1s[b + 1u], b1s[b + 2u], b1s[b + 3u]),
        vec4<f32>(0.0),
      );
      let jg = c0 / 4u + g;
      for (var c: u32 = 0u; c < C; c = c + 1u) {
        dxv[c] = dxv[c] + dot(w2s[c * HG + jg], hs);
      }
    }
    workgroupBarrier();  // w1s is about to be overwritten by the next chunk
  }

  if (!inb) { return; }
  let f = fire[y * p.W + x];
  for (var c: u32 = 0u; c < C; c = c + 1u) {
    out[sidx(c, y, x)] = state[sidx(c, y, x)] + dxv[c] * f * p.step;
  }
}
