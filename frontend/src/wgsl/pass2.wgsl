struct Params {
  C: u32, H: u32, W: u32, hidden: u32,
  w1: u32, b1: u32, w2: u32, b2: u32,
  step: f32,
};
fn sidx(c: u32, y: u32, x: u32) -> u32 { return (c * p.H + y) * p.W + x; }

@group(0) @binding(0) var<storage, read> old: array<f32>;
@group(0) @binding(1) var<storage, read> tmp: array<f32>;
@group(0) @binding(2) var<storage, read_write> next: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= p.W || y >= p.H) { return; }
  var pre = 0.0;
  var post = 0.0;
  for (var dy: u32 = 0u; dy < 3u; dy = dy + 1u) {
    for (var dx: u32 = 0u; dx < 3u; dx = dx + 1u) {
      let yy = (y + dy + p.H - 1u) % p.H;
      let xx = (x + dx + p.W - 1u) % p.W;
      pre = max(pre, old[sidx(3u, yy, xx)]);
      post = max(post, tmp[sidx(3u, yy, xx)]);
    }
  }
  var alive = 0.0;
  if (pre > 0.1 && post > 0.1) { alive = 1.0; }
  for (var c: u32 = 0u; c < 16u; c = c + 1u) {
    next[sidx(c, y, x)] = tmp[sidx(c, y, x)] * alive;
  }
}
