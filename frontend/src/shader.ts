/**
 * A native WebGPU (WGSL) implementation of one CA step.
 *
 * ORT-Web costs ~1-12 ms *per graph node* (each node is a separate dispatch), and
 * our step has ~19 nodes. This engine does the whole step in **two** compute
 * passes with no WASM bridge:
 *
 *   1. perception (depthwise 3x3, wrap) + update-rule MLP + fire/step -> `tmp`
 *   2. living-cell mask (pre & post) -> the other state buffer
 *
 * The MLP weights come from `weights.bin` (exported alongside the ONNX model).
 */

import type { ManifestModel } from './models';
import { CANONICAL_KERNELS, mulberry32, seedState } from './nca';
import type { Rng } from './nca';
import type { CaEngine, Perf } from './engine';

const WG = 8;

const PARAMS_STRUCT = /* wgsl */ `
struct Params {
  C: u32, H: u32, W: u32, hidden: u32,
  w1: u32, b1: u32, w2: u32, b2: u32,
  step: f32,
};
fn sidx(c: u32, y: u32, x: u32) -> u32 { return (c * p.H + y) * p.W + x; }
`;

const PASS1 = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<storage, read> fire: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;

@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= p.W || y >= p.H) { return; }

  // perception: for each channel, [identity, dx, dy] responses (order matches the
  // depthwise conv: out channel = 3*ci + cm)
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

  // update rule: dx = W2 * relu(W1 * perc + b1) + b2, fused (no hidden array)
  var dxv: array<f32, 16>;
  for (var c: u32 = 0u; c < 16u; c = c + 1u) { dxv[c] = weights[p.b2 + c]; }
  for (var j: u32 = 0u; j < p.hidden; j = j + 1u) {
    var h = weights[p.b1 + j];
    for (var i: u32 = 0u; i < 48u; i = i + 1u) {
      h = h + weights[p.w1 + j * 48u + i] * perc[i];
    }
    h = max(h, 0.0);
    for (var c: u32 = 0u; c < 16u; c = c + 1u) {
      dxv[c] = dxv[c] + weights[p.w2 + c * p.hidden + j] * h;
    }
  }

  let f = fire[y * p.W + x];
  for (var c: u32 = 0u; c < 16u; c = c + 1u) {
    out[sidx(c, y, x)] = state[sidx(c, y, x)] + dxv[c] * f * p.step;
  }
}
`;

const PASS2 = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<storage, read> old: array<f32>;
@group(0) @binding(1) var<storage, read> tmp: array<f32>;
@group(0) @binding(2) var<storage, read_write> next: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(${WG}, ${WG}, 1)
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
`;

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export class ShaderRunner implements CaEngine {
  readonly meta: ManifestModel['meta'];
  readonly channelN: number;
  grid: number;
  fireRate: number;
  stepCount = 0;
  x: Float32Array;
  readonly perf: Perf = { maskMs: 0, runMs: 0, syncMs: 0, steps: 0 };

  private device: GPUDevice;
  private adapterDesc: string;
  private hidden: number;
  private off: { w1: number; b1: number; w2: number; b2: number };
  private weightData: Float32Array;

  private stateA!: GPUBuffer;
  private stateB!: GPUBuffer;
  private tmpBuf!: GPUBuffer;
  private fireBuf!: GPUBuffer;
  private weightBuf!: GPUBuffer;
  private kernelBuf!: GPUBuffer;
  private paramsBuf!: GPUBuffer;
  private staging!: GPUBuffer;

  private pipeline1!: GPUComputePipeline;
  private pipeline2!: GPUComputePipeline;
  private bg1!: [GPUBindGroup, GPUBindGroup];
  private bg2!: [GPUBindGroup, GPUBindGroup];

  private cur = 0;
  private stateBytes = 0;
  private mask: Float32Array;
  private kernels: number[][] = CANONICAL_KERNELS.map((k) => k.slice());
  private rng: Rng;
  private seed: number;
  private stepping = false;
  private queuedEdits: Array<(x: Float32Array) => void> = [];

  private constructor(
    device: GPUDevice,
    adapterDesc: string,
    model: ManifestModel,
    weights: Float32Array,
    seed: number,
  ) {
    this.device = device;
    this.adapterDesc = adapterDesc;
    this.meta = model.meta;
    this.channelN = model.meta.channel_n;
    this.grid = model.meta.grid;
    this.fireRate = model.meta.fire_rate;
    this.seed = seed;
    this.hidden = model.meta.weights.hidden_n;
    this.off = model.meta.weights.offsets;
    this.weightData = weights;
    this.mask = new Float32Array(this.grid * this.grid);
    this.rng = mulberry32(seed);
    this.x = seedState(this.channelN, this.grid);
  }

  static async create(model: ManifestModel, seed = 1): Promise<ShaderRunner> {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) throw new Error('no WebGPU adapter');
    const device = await adapter.requestDevice();
    const info = (adapter as unknown as { info?: Record<string, string> }).info;
    const desc = info
      ? [info.vendor, info.architecture, info.device, info.description]
          .filter(Boolean)
          .join(' ') || 'unknown'
      : 'unknown';

    const res = await fetch(`/models/${model.meta.id}/${model.meta.weights.file}`);
    if (!res.ok) throw new Error(`weights fetch failed: ${res.status}`);
    const weights = new Float32Array(await res.arrayBuffer());

    const runner = new ShaderRunner(device, desc, model, weights, seed);
    runner.init();
    return runner;
  }

  get mode(): string {
    return 'shader';
  }

  get adapter(): string {
    return this.adapterDesc;
  }

  private init(): void {
    const d = this.device;
    this.pipeline1 = d.createComputePipeline({
      layout: 'auto',
      compute: { module: d.createShaderModule({ code: PASS1 }), entryPoint: 'main' },
    });
    this.pipeline2 = d.createComputePipeline({
      layout: 'auto',
      compute: { module: d.createShaderModule({ code: PASS2 }), entryPoint: 'main' },
    });
    this.weightBuf = d.createBuffer({ size: this.weightData.byteLength, usage: STORAGE });
    d.queue.writeBuffer(this.weightBuf, 0, this.weightData);
    this.kernelBuf = d.createBuffer({
      size: 27 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.paramsBuf = d.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.allocGrid();
    this.writeKernel();
  }

  /** (Re)allocate the size-dependent buffers and bind groups. */
  private allocGrid(): void {
    const d = this.device;
    const { channelN: C, grid: H } = this;
    const W = H;
    this.stateBytes = C * H * W * 4;
    for (const b of [this.stateA, this.stateB, this.tmpBuf, this.fireBuf, this.staging]) {
      b?.destroy();
    }
    const size = Math.ceil(this.stateBytes / 16) * 16;
    const mk = (usage: number) => d.createBuffer({ size, usage });
    this.stateA = mk(STORAGE);
    this.stateB = mk(STORAGE);
    this.tmpBuf = mk(STORAGE);
    this.fireBuf = d.createBuffer({
      size: Math.ceil((H * W * 4) / 16) * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.staging = d.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const l1 = this.pipeline1.getBindGroupLayout(0);
    const l2 = this.pipeline2.getBindGroupLayout(0);
    const bg1 = (state: GPUBuffer): GPUBindGroup =>
      d.createBindGroup({
        layout: l1,
        entries: [
          { binding: 0, resource: { buffer: state } },
          { binding: 1, resource: { buffer: this.fireBuf } },
          { binding: 2, resource: { buffer: this.weightBuf } },
          { binding: 3, resource: { buffer: this.kernelBuf } },
          { binding: 4, resource: { buffer: this.tmpBuf } },
          { binding: 5, resource: { buffer: this.paramsBuf } },
        ],
      });
    const bg2 = (old: GPUBuffer, next: GPUBuffer): GPUBindGroup =>
      d.createBindGroup({
        layout: l2,
        entries: [
          { binding: 0, resource: { buffer: old } },
          { binding: 1, resource: { buffer: this.tmpBuf } },
          { binding: 2, resource: { buffer: next } },
          { binding: 3, resource: { buffer: this.paramsBuf } },
        ],
      });
    // index 0: cur = A, 1: cur = B
    this.bg1 = [bg1(this.stateA), bg1(this.stateB)];
    this.bg2 = [bg2(this.stateA, this.stateB), bg2(this.stateB, this.stateA)];

    this.writeParams();
    this.cur = 0;
    this.uploadState();
  }

  private writeParams(): void {
    const buf = new ArrayBuffer(48);
    new Uint32Array(buf, 0, 8).set([
      this.channelN,
      this.grid,
      this.grid,
      this.hidden,
      this.off.w1,
      this.off.b1,
      this.off.w2,
      this.off.b2,
    ]);
    new Float32Array(buf, 32, 1)[0] = 1.0;
    this.device.queue.writeBuffer(this.paramsBuf, 0, buf);
  }

  private writeKernel(): void {
    const k = new Float32Array(27);
    for (let m = 0; m < 3; m++) for (let i = 0; i < 9; i++) k[m * 9 + i] = this.kernels[m][i];
    this.device.queue.writeBuffer(this.kernelBuf, 0, k);
  }

  private uploadState(): void {
    const buf = this.cur === 0 ? this.stateA : this.stateB;
    this.device.queue.writeBuffer(buf, 0, this.x);
  }

  reset(seed?: number): void {
    if (seed !== undefined) this.seed = seed;
    this.rng = mulberry32(this.seed);
    this.x = seedState(this.channelN, this.grid);
    this.queuedEdits = [];
    this.cur = 0;
    this.stepCount = 0;
    this.uploadState();
  }

  edit(fn: (x: Float32Array) => void): void {
    if (this.stepping) {
      this.queuedEdits.push(fn);
      return;
    }
    fn(this.x);
    this.uploadState();
  }

  private flushEdits(): void {
    if (this.queuedEdits.length === 0) return;
    // pending edits are applied to the CPU mirror after this step's `sync()`
    const queued = this.queuedEdits;
    this.queuedEdits = [];
    for (const fn of queued) fn(this.x);
    this.uploadState();
  }

  async sync(): Promise<void> {
    const t0 = performance.now();
    const buf = this.cur === 0 ? this.stateA : this.stateB;
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, this.staging, 0, this.stateBytes);
    this.device.queue.submit([enc.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ);
    this.x = new Float32Array(this.staging.getMappedRange(0, this.stateBytes).slice(0));
    this.staging.unmap();
    this.perf.syncMs += performance.now() - t0;
  }

  async step(): Promise<void> {
    const { grid: H } = this;
    const plane = H * H;

    const tMask = performance.now();
    for (let i = 0; i < plane; i++) {
      this.mask[i] = this.rng() < this.fireRate ? 1 : 0;
    }
    this.device.queue.writeBuffer(this.fireBuf, 0, this.mask);
    this.perf.maskMs += performance.now() - tMask;

    const tRun = performance.now();
    this.stepping = true;
    try {
      const enc = this.device.createCommandEncoder();
      const pass1 = enc.beginComputePass();
      pass1.setPipeline(this.pipeline1);
      pass1.setBindGroup(0, this.bg1[this.cur]);
      pass1.dispatchWorkgroups(Math.ceil(H / WG), Math.ceil(H / WG));
      pass1.end();
      const pass2 = enc.beginComputePass();
      pass2.setPipeline(this.pipeline2);
      pass2.setBindGroup(0, this.bg2[this.cur]);
      pass2.dispatchWorkgroups(Math.ceil(H / WG), Math.ceil(H / WG));
      pass2.end();
      this.device.queue.submit([enc.finish()]);
      this.cur ^= 1;
    } finally {
      this.stepping = false;
    }
    // edits made during the step are applied to the CPU mirror (which the caller
    // refreshes via sync()); upload them next step.
    if (this.queuedEdits.length) {
      await this.sync();
      this.flushEdits();
    }
    this.perf.runMs += performance.now() - tRun;
    this.perf.steps++;
    this.stepCount++;
  }

  setGrid(size: number): void {
    if (size === this.grid) return;
    this.grid = size;
    this.mask = new Float32Array(size * size);
    this.queuedEdits = [];
    this.allocGrid();
    this.reset();
  }

  setKernels(ident: number[], dx: number[], dy: number[]): void {
    this.kernels = [ident.slice(), dx.slice(), dy.slice()];
    this.writeKernel();
  }

  async dispose(): Promise<void> {
    for (const b of [
      this.stateA,
      this.stateB,
      this.tmpBuf,
      this.fireBuf,
      this.weightBuf,
      this.kernelBuf,
      this.paramsBuf,
      this.staging,
    ]) {
      b?.destroy();
    }
    this.device.destroy();
  }
}
