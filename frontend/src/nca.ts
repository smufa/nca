/**
 * Pure NCA helpers with no DOM / ORT dependencies, so they can be unit-tested
 * in Node and reused by the analysis views.
 */

export type Rng = () => number;

/** Small deterministic PRNG so runs are reproducible. */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The canonical perception basis: [identity, Sobel-x/8, Sobel-y/8]. */
export const CANONICAL_KERNELS: readonly number[][] = [
  [0, 0, 0, 0, 1, 0, 0, 0, 0],
  [-1, 0, 1, -2, 0, 2, -1, 0, 1].map((v) => v / 8),
  [-1, -2, -1, 0, 0, 0, 1, 2, 1].map((v) => v / 8),
];

/** Expand three 3x3 basis kernels into the depthwise `[3C, 1, 3, 3]` weight. */
export function buildPerceptionWeight(
  channelN: number,
  kernels: readonly number[][],
): Float32Array {
  const w = new Float32Array(3 * channelN * 9);
  for (let ci = 0; ci < channelN; ci++) {
    for (let cm = 0; cm < 3; cm++) {
      const base = (ci * 3 + cm) * 9;
      for (let k = 0; k < 9; k++) w[base + k] = kernels[cm][k];
    }
  }
  return w;
}

/** The fixed identity + Sobel perception basis. */
export function perceptionWeight(channelN: number): Float32Array {
  return buildPerceptionWeight(channelN, CANONICAL_KERNELS);
}

/**
 * The perception kernels for a model: the trained ones from `meta` when present
 * (a `--learn-perception` run), otherwise the canonical Sobel basis. Stored
 * flattened as 27 numbers, `[identity, dx, dy]`.
 */
export function modelKernels(meta: { perception_kernel?: number[] }): number[][] {
  const flat = meta.perception_kernel;
  if (!flat || flat.length !== 27) return CANONICAL_KERNELS.map((k) => k.slice());
  return [0, 1, 2].map((m) => Array.from(flat.slice(m * 9, m * 9 + 9)));
}

/**
 * Rotate the perception basis by `angle` radians (identity unchanged):
 * `dx' = cos·dx − sin·dy`, `dy' = sin·dx + cos·dy`.
 */
export function rotateKernels(angle: number): number[][] {
  const [ident, dx, dy] = CANONICAL_KERNELS;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [
    ident.slice(),
    dx.map((v, i) => c * v - s * dy[i]),
    dx.map((v, i) => s * v + c * dy[i]),
  ];
}

/** A single seed cell in the middle: RGB = 0, all other channels = 1. */
export function seedState(channelN: number, grid: number): Float32Array {
  const plane = grid * grid;
  const x = new Float32Array(channelN * plane);
  const mid = (grid >> 1) * grid + (grid >> 1);
  for (let c = 3; c < channelN; c++) x[c * plane + mid] = 1.0;
  return x;
}

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Which state channels map to R, G, B and how to composite them. */
export interface ChannelMapping {
  r: number;
  g: number;
  b: number;
  /** Alpha channel for compositing, or -1 for a direct (opaque) mapping. */
  a: number;
  /** Background level for the alpha composite: 0 = black, 1 = white. */
  bg: number;
}

/**
 * Map three arbitrary state channels to RGB and write RGBA bytes into `out`.
 * `x` is the flat NCHW state `[1, C, H, W]`.
 *
 * With `a >= 0` the selected channels are treated as premultiplied and
 * composited over `bg`: `out = channel + (1 - alpha) * bg`. With `a = -1` the
 * channels are shown directly. The default mapping `{0,1,2, a:3, bg:1}`
 * reproduces the familiar RGB-over-white view.
 */
export function mapChannels(
  x: Float32Array,
  h: number,
  w: number,
  m: ChannelMapping,
  out: Uint8ClampedArray,
): void {
  const plane = h * w;
  const rp = m.r * plane;
  const gp = m.g * plane;
  const bp = m.b * plane;
  const ap = m.a >= 0 ? m.a * plane : -1;
  for (let i = 0; i < plane; i++) {
    const a = ap >= 0 ? clamp01(x[ap + i]) : 1;
    const inv = (1 - a) * m.bg;
    const o = i * 4;
    out[o] = clamp01(x[rp + i] + inv) * 255;
    out[o + 1] = clamp01(x[gp + i] + inv) * 255;
    out[o + 2] = clamp01(x[bp + i] + inv) * 255;
    out[o + 3] = 255;
  }
}
