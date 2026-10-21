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

fn percep(c: u32, x: u32, y: u32) -> vec3<f32> {
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
  return vec3<f32>(ki, kx, ky);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // two adjacent cells per thread; each weight load is reused for both
  let x0 = gid.x * 2u;
  let x1 = x0 + 1u;
  let y = gid.y;
  if (x1 >= p.W || y >= p.H) { return; }

  var pa: array<vec4<f32>, 12>;
  var pb: array<vec4<f32>, 12>;
  var tmpa: array<f32, 48>;
  var tmpb: array<f32, 48>;
  for (var c: u32 = 0u; c < 16u; c = c + 1u) {
    let ra = percep(c, x0, y);
    let rb = percep(c, x1, y);
    tmpa[c * 3u] = ra.x;
    tmpa[c * 3u + 1u] = ra.y;
    tmpa[c * 3u + 2u] = ra.z;
    tmpb[c * 3u] = rb.x;
    tmpb[c * 3u + 1u] = rb.y;
    tmpb[c * 3u + 2u] = rb.z;
  }
  for (var k: u32 = 0u; k < 12u; k = k + 1u) {
    let b = k * 4u;
    pa[k] = vec4<f32>(tmpa[b], tmpa[b + 1u], tmpa[b + 2u], tmpa[b + 3u]);
    pb[k] = vec4<f32>(tmpb[b], tmpb[b + 1u], tmpb[b + 2u], tmpb[b + 3u]);
  }

  var dxa: array<f32, 16>;
  var dxb: array<f32, 16>;
  for (var c: u32 = 0u; c < 16u; c = c + 1u) {
    dxa[c] = weights[p.b2 + c];
    dxb[c] = weights[p.b2 + c];
  }
  let w2v = p.w2 / 4u;
  for (var jg: u32 = 0u; jg < 32u; jg = jg + 1u) {
    var acca = vec4<f32>(0.0);
    var accb = vec4<f32>(0.0);
    let base = jg * 48u;
    for (var k: u32 = 0u; k < 12u; k = k + 1u) {
      let v0 = w4[base + k];
      let v1 = w4[base + 12u + k];
      let v2 = w4[base + 24u + k];
      let v3 = w4[base + 36u + k];
      let pav = pa[k];
      let pbv = pb[k];
      acca = acca + vec4<f32>(dot(v0, pav), dot(v1, pav), dot(v2, pav), dot(v3, pav));
      accb = accb + vec4<f32>(dot(v0, pbv), dot(v1, pbv), dot(v2, pbv), dot(v3, pbv));
    }
    let bias = vec4<f32>(
      weights[p.b1 + jg * 4u], weights[p.b1 + jg * 4u + 1u],
      weights[p.b1 + jg * 4u + 2u], weights[p.b1 + jg * 4u + 3u]);
    let hsa = max(acca + bias, vec4<f32>(0.0));
    let hsb = max(accb + bias, vec4<f32>(0.0));
    for (var c: u32 = 0u; c < 16u; c = c + 1u) {
      let wv = w4[w2v + c * 32u + jg];
      dxa[c] = dxa[c] + dot(wv, hsa);
      dxb[c] = dxb[c] + dot(wv, hsb);
    }
  }

  let fa = fire[y * p.W + x0];
  let fb = fire[y * p.W + x1];
  for (var c: u32 = 0u; c < 16u; c = c + 1u) {
    out[sidx(c, y, x0)] = state[sidx(c, y, x0)] + dxa[c] * fa * p.step;
    out[sidx(c, y, x1)] = state[sidx(c, y, x1)] + dxb[c] * fb * p.step;
  }
}
