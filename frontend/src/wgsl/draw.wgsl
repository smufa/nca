struct DrawParams {
  H: u32, W: u32,
  r: i32, g: i32, b: i32, a: i32,
  bg: f32, pad: f32,
};
@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<uniform> dp: DrawParams;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  let x = u32(fc.x);
  let y = u32(fc.y);
  if (x >= dp.W || y >= dp.H) { discard; }
  let plane = dp.H * dp.W;
  let i = y * dp.W + x;
  var alpha = 1.0;
  if (dp.a >= 0) { alpha = clamp(state[u32(dp.a) * plane + i], 0.0, 1.0); }
  let inv = (1.0 - alpha) * dp.bg;
  return vec4<f32>(
    clamp(state[u32(dp.r) * plane + i] + inv, 0.0, 1.0),
    clamp(state[u32(dp.g) * plane + i] + inv, 0.0, 1.0),
    clamp(state[u32(dp.b) * plane + i] + inv, 0.0, 1.0),
    1.0);
}
