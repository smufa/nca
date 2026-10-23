// EXPERIMENT (Phase 6a): living mask for the packed state layout.
//
// `frontend/src/wgsl/pass2.wgsl` assumes the channel-planar layout when it reads
// alpha as `(3*H + y)*W + x`; on a packed buffer that is an arbitrary float and the
// mask kills every cell. Here one cell is `C/4` contiguous vec4 texels, and alpha
// (channel 3) is the w lane of the cell's first texel.

override C: u32 = 16u;

struct Params {
  C: u32, H: u32, W: u32, hidden: u32,
  w1: u32, b1: u32, w2: u32, b2: u32,
  step: f32,
};

@group(0) @binding(0) var<storage, read> old: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tmp: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> next: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= p.W || y >= p.H) { return; }
  let c4 = C / 4u;
  var pre = 0.0;
  var post = 0.0;
  for (var dy: u32 = 0u; dy < 3u; dy = dy + 1u) {
    for (var dx: u32 = 0u; dx < 3u; dx = dx + 1u) {
      let yy = (y + dy + p.H - 1u) % p.H;
      let xx = (x + dx + p.W - 1u) % p.W;
      let cell = (yy * p.W + xx) * c4;
      pre = max(pre, old[cell].w);  // channel 3 = alpha
      post = max(post, tmp[cell].w);
    }
  }
  var alive = 0.0;
  if (pre > 0.1 && post > 0.1) { alive = 1.0; }
  let cell = (y * p.W + x) * c4;
  for (var q: u32 = 0u; q < c4; q = q + 1u) {
    next[cell + q] = tmp[cell + q] * alive;
  }
}
