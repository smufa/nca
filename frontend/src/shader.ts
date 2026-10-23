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
 * The shaders live in `wgsl/*.wgsl` so the same source is used by the headless
 * test harness in `scripts/shader_bench.py`. The MLP weights come from
 * `weights.bin` (exported alongside the ONNX model).
 */

import type { ManifestModel } from './models';
import { CANONICAL_KERNELS, modelKernels, mulberry32, seedState } from './nca';
import type { Rng } from './nca';
import type { BrushOp, CaEngine, Capabilities, Perf, Support } from './engine';
import { checkSpec } from './engine';
import type { ChannelMapping } from './nca';

import PASS1 from './wgsl/pass1.wgsl?raw';
import PASS2 from './wgsl/pass2.wgsl?raw';
import DRAW_WGSL from './wgsl/draw.wgsl?raw';
import BRUSH_WGSL from './wgsl/brush.wgsl?raw';

const WG = 8;
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

interface GpuViewRes {
  context: GPUCanvasContext;
  uniform: GPUBuffer;
  bindGroups: [GPUBindGroup, GPUBindGroup];
}

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
  private layers = 1;
  private wmid = 0;
  private off: {
    w1: number;
    b1: number;
    w2: number;
    b2: number;
    wmid?: number;
    bmid?: number;
    mid_stride?: number;
  };
  private weightData: Float32Array;

  private stateA!: GPUBuffer;
  private stateB!: GPUBuffer;
  private tmpBuf!: GPUBuffer;
  private fireBuf!: GPUBuffer;
  private weightBuf!: GPUBuffer;
  private weight4Buf!: GPUBuffer;
  private kernelBuf!: GPUBuffer;
  private condBuf!: GPUBuffer;
  private paramsBuf!: GPUBuffer;
  private staging!: GPUBuffer;

  private pipeline1!: GPUComputePipeline;
  private pipeline2!: GPUComputePipeline;
  private brushPipeline!: GPUComputePipeline;
  private brushUniform!: GPUBuffer;
  private brushBindGroups!: [GPUBindGroup, GPUBindGroup];
  private drawPipeline!: GPURenderPipeline;
  private gpuViews = new Map<HTMLCanvasElement, GpuViewRes>();
  private bg1!: [GPUBindGroup, GPUBindGroup];
  private bg2!: [GPUBindGroup, GPUBindGroup];

  private cur = 0;
  private stateBytes = 0;
  private mask: Float32Array;
  private kernels: number[][] = CANONICAL_KERNELS.map((k) => k.slice());
  /** One-hot target selector for a conditional model (length condN, may be 0). */
  private condN = 0;
  private cond: Float32Array;
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
    this.layers = model.meta.weights.hidden_layers ?? 1;
    this.wmid = this.off.wmid ?? 0;
    this.kernels = modelKernels(model.meta);
    this.condN = model.meta.cond_n ?? 0;
    this.cond = new Float32Array(Math.max(this.condN, 1));
    if (this.condN) this.cond[0] = 1;
    this.weightData = weights;
    this.mask = new Float32Array(this.grid * this.grid);
    this.rng = mulberry32(seed);
    this.x = seedState(this.channelN, this.grid);
  }

  static async create(model: ManifestModel, seed = 1): Promise<ShaderRunner> {
    const support = ShaderRunner.supports(model);
    if (!support.ok) {
      throw new Error(`shader engine cannot run this model: ${support.reason}`);
    }
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

  /**
   * Limits of the WGSL step, each traceable to a declaration in `wgsl/pass1.wgsl`:
   * `perc` is `array<f32, 64>` and `perc4` is `array<vec4<f32>, 16>`, so
   * `3C + CONDN <= 64`; the hidden vector is `array<vec4<f32>, HMAX=64>` because
   * WGSL array sizes cannot be pipeline overrides, so `hidden/4 <= 64`; `LH` only
   * has offsets for one hidden->hidden block in `weights.bin`, so 2 layers.
   *
   * Grid: three storage buffers of `C*H*W*4` bytes must fit the default
   * `maxStorageBufferBindingSize` (128 MiB), which is a *pixel* budget here --
   * with C=16 that is ~700k pixels, i.e. up to 836x836.
   */
  static readonly CAPABILITIES: Capabilities = {
    kind: 'shader',
    label: 'WebGPU shader',
    cond: true,
    gpuBrush: true,
    gpuRender: true,
    maxC: 16,
    maxHidden: 256,
    maxLayers: 2,
    maxCond: 16,
    maxPerc: 64,
    maxGridPx: Math.floor((128 * 1024 * 1024) / (3 * 16 * 4)),
  };

  static supports(model: ManifestModel): Support {
    return checkSpec(ShaderRunner.CAPABILITIES, model.meta);
  }

  get capabilities(): Capabilities {
    return ShaderRunner.CAPABILITIES;
  }

  get mode(): string {
    return 'shader';
  }

  get adapter(): string {
    return this.adapterDesc;
  }

  private init(): void {
    const d = this.device;
    // Size-dependent constants are pipeline overrides so naga keeps the MLP
    // loops unrolled (dynamic bounds cost ~1.9x).
    const C = this.channelN;
    const p4 = Math.ceil((3 * C + this.condN) / 4);
    const hg = this.hidden / 4;
    // WGSL array sizes must be const-expressions, so HMAX is baked into the source
    // per model rather than set as an override: sizing it to the actual hidden
    // width keeps the deep path's hidden arrays (h, h2) as small as possible, and
    // they are the main register-pressure source.
    const src = PASS1.replace('HMAX: u32 = 64u', `HMAX: u32 = ${Math.max(hg, 1)}u`);
    const constants = {
      C,
      NP4: p4,
      W1S4: p4,
      HG: hg,
      CONDN: this.condN,
      LH: this.layers,
    };
    this.pipeline1 = d.createComputePipeline({
      layout: 'auto',
      compute: {
        module: d.createShaderModule({ code: src }),
        entryPoint: 'main',
        constants,
      },
    });
    this.pipeline2 = d.createComputePipeline({
      layout: 'auto',
      compute: {
        module: d.createShaderModule({ code: PASS2 }),
        entryPoint: 'main',
        constants: { C },
      },
    });
    this.brushPipeline = d.createComputePipeline({
      layout: 'auto',
      compute: { module: d.createShaderModule({ code: BRUSH_WGSL }), entryPoint: 'main' },
    });
    this.brushUniform = d.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const drawModule = d.createShaderModule({ code: DRAW_WGSL });
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.drawPipeline = d.createRenderPipeline({
      layout: 'auto',
      vertex: { module: drawModule, entryPoint: 'vs' },
      fragment: { module: drawModule, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });

    this.weightBuf = d.createBuffer({ size: this.weightData.byteLength, usage: STORAGE });
    d.queue.writeBuffer(this.weightBuf, 0, this.weightData);
    // same data viewed as vec4 (4x fewer weight loads in the MLP)
    this.weight4Buf = d.createBuffer({ size: this.weightData.byteLength, usage: STORAGE });
    d.queue.writeBuffer(this.weight4Buf, 0, this.weightData);
    this.kernelBuf = d.createBuffer({
      size: 27 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // Always bound, even for an unconditional model (CONDN=0 reads nothing).
    this.condBuf = d.createBuffer({
      size: Math.max(16, this.condN * 4),
      usage: STORAGE,
    });
    this.writeCond();
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
          { binding: 6, resource: { buffer: this.weight4Buf } },
          { binding: 7, resource: { buffer: this.condBuf } },
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

    const bl = this.brushPipeline.getBindGroupLayout(0);
    const bgBrush = (state: GPUBuffer): GPUBindGroup =>
      d.createBindGroup({
        layout: bl,
        entries: [
          { binding: 0, resource: { buffer: state } },
          { binding: 1, resource: { buffer: this.brushUniform } },
        ],
      });
    this.brushBindGroups = [bgBrush(this.stateA), bgBrush(this.stateB)];
    this.gpuViews.clear(); // cached draw bind groups reference the old buffers

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
    new Uint32Array(buf, 36, 1)[0] = this.wmid;
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

  getKernels(): number[][] {
    return this.kernels.map((k) => k.slice());
  }

  /**
   * Select which target a conditional model should grow.
   *
   * The state is deliberately left alone: switching the one-hot mid-run should
   * *convert* whatever is on the grid towards the new target, which is the whole
   * point of a conditional model. Use `reset()` to go back to the seed.
   */
  setCond(index: number): void {
    if (!this.condN) return;
    this.cond.fill(0);
    this.cond[Math.max(0, Math.min(this.condN - 1, index | 0))] = 1;
    this.writeCond();
  }

  /** Set the conditioning vector directly (arbitrary mixture, not just one-hot). */
  setCondVector(values: number[]): void {
    if (!this.condN) return;
    for (let i = 0; i < this.condN; i++) this.cond[i] = values[i] ?? 0;
    this.writeCond();
  }

  private writeCond(): void {
    if (this.condN) this.device.queue.writeBuffer(this.condBuf, 0, this.cond);
  }

  /** Render the current GPU state straight to a canvas (no download). */
  renderTo(canvas: HTMLCanvasElement, mapping: ChannelMapping): void {
    const d = this.device;
    let gv = this.gpuViews.get(canvas);
    if (!gv) {
      const context = canvas.getContext('webgpu') as GPUCanvasContext | null;
      if (!context) throw new Error('no WebGPU canvas context');
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device: d, format, alphaMode: 'opaque' });
      const uniform = d.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const layout = this.drawPipeline.getBindGroupLayout(0);
      const bg = (state: GPUBuffer): GPUBindGroup =>
        d.createBindGroup({
          layout,
          entries: [
            { binding: 0, resource: { buffer: state } },
            { binding: 1, resource: { buffer: uniform } },
          ],
        });
      gv = { context, uniform, bindGroups: [bg(this.stateA), bg(this.stateB)] };
      this.gpuViews.set(canvas, gv);
    }

    const buf = new ArrayBuffer(32);
    const u = new Uint32Array(buf);
    const s = new Int32Array(buf);
    u[0] = this.grid;
    u[1] = this.grid;
    s[2] = mapping.r;
    s[3] = mapping.g;
    s[4] = mapping.b;
    s[5] = mapping.a;
    new Float32Array(buf, 24, 1)[0] = mapping.bg;
    d.queue.writeBuffer(gv.uniform, 0, buf);

    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: gv.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(this.drawPipeline);
    pass.setBindGroup(0, gv.bindGroups[this.cur]);
    pass.draw(3);
    pass.end();
    d.queue.submit([enc.finish()]);
  }

  /** Apply a brush stroke directly on the GPU state. */
  brush(
    op: BrushOp,
    cx: number,
    cy: number,
    radius: number,
    factor: number,
    channels: number[],
  ): void {
    let mask = 0;
    for (const c of channels) if (c >= 0 && c < 32) mask |= 1 << c;
    const buf = new ArrayBuffer(48);
    const u = new Uint32Array(buf);
    const s = new Int32Array(buf);
    u[0] = this.grid;
    u[1] = this.grid;
    s[2] = cx;
    s[3] = cy;
    s[4] = radius;
    u[5] = op === 'one' ? 0 : op === 'zero' ? 1 : 2;
    new Float32Array(buf, 24, 1)[0] = factor;
    u[7] = mask >>> 0;
    u[8] = this.channelN;
    this.device.queue.writeBuffer(this.brushUniform, 0, buf);

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.brushPipeline);
    pass.setBindGroup(0, this.brushBindGroups[this.cur]);
    pass.dispatchWorkgroups(Math.ceil(this.grid / WG), Math.ceil(this.grid / WG));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  async dispose(): Promise<void> {
    for (const b of [
      this.stateA,
      this.stateB,
      this.tmpBuf,
      this.fireBuf,
      this.weightBuf,
      this.weight4Buf,
      this.kernelBuf,
      this.condBuf,
      this.paramsBuf,
      this.staging,
    ]) {
      b?.destroy();
    }
    this.device.destroy();
  }
}
