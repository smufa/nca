import * as ort from 'onnxruntime-web/webgpu';
import type { ManifestModel } from './models';
import { buildPerceptionWeight, CANONICAL_KERNELS, modelKernels, mulberry32, seedState } from './nca';
import type { Rng } from './nca';
import type { CaEngine } from './engine';

// Single-threaded WASM core is enough: the heavy lifting runs on the WebGPU EP,
// and this avoids needing cross-origin-isolation (SharedArrayBuffer) headers.
ort.env.wasm.numThreads = 1;

// `?log=info` / `?log=verbose` surfaces ORT's logs (e.g. nodes not assigned to
// the WebGPU EP). Useful for diagnosing slow WebGPU runs.
const _log = new URLSearchParams(location.search).get('log');
if (_log) ort.env.logLevel = _log === 'verbose' ? 'verbose' : 'info';

const STATE_USAGE =
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

/**
 * Runs one trained NCA.
 *
 * When WebGPU IO binding is available the state lives in a pair of ping-pong
 * GPU buffers, so a step never moves the state across the bus: the fire mask is
 * written into a small GPU buffer, the output is written straight into the other
 * state buffer (`session.run(feeds, fetches)`), and we only download the state
 * for rendering (`sync`). If that setup fails we fall back to CPU tensors.
 */
export class CaRunner implements CaEngine {
  readonly meta: ManifestModel['meta'];
  readonly channelN: number;
  grid: number;
  fireRate: number;
  stepCount = 0;
  x: Float32Array;
  /** Cumulative step timings (ms) + step count, for the perf readout. */
  readonly perf = { maskMs: 0, runMs: 0, syncMs: 0, steps: 0 };

  private session: ort.InferenceSession;
  private weight: Float32Array;
  private mask: Float32Array;
  private kernels: number[][] = CANONICAL_KERNELS.map((k) => k.slice());
  private rng: Rng;
  private seed: number;
  private stepping = false;
  private queuedEdits: Array<(x: Float32Array) => void> = [];

  // GPU IO-binding state
  private useGpu = false;
  private device: GPUDevice | null = null;
  private stateBytes = 0;
  private bufA: GPUBuffer | null = null;
  private bufB: GPUBuffer | null = null;
  private maskBuf: GPUBuffer | null = null;
  private weightBuf: GPUBuffer | null = null;
  private staging: GPUBuffer | null = null;
  private tensorA: ort.Tensor | null = null;
  private tensorB: ort.Tensor | null = null;
  private maskTensor: ort.Tensor | null = null;
  private weightTensor: ort.Tensor | null = null;
  private stepTensor: ort.Tensor | null = null;
  private cur = 0;
  private adapterDesc = '';

  private constructor(
    session: ort.InferenceSession,
    meta: ManifestModel['meta'],
    seed: number,
  ) {
    this.session = session;
    this.meta = meta;
    this.channelN = meta.channel_n;
    this.grid = meta.grid;
    this.fireRate = meta.fire_rate;
    this.seed = seed;
    this.kernels = modelKernels(meta);
    this.weight = buildPerceptionWeight(this.channelN, this.kernels);
    this.mask = new Float32Array(this.grid * this.grid);
    this.rng = mulberry32(seed);
    this.x = seedState(this.channelN, this.grid);
  }

  static async create(
    model: ManifestModel,
    ep: 'webgpu' | 'wasm' = 'webgpu',
    seed = 1,
  ): Promise<CaRunner> {
    const session = await ort.InferenceSession.create(`/models/${model.path}`, {
      executionProviders: ep === 'wasm' ? ['wasm'] : ['webgpu'],
    });
    const runner = new CaRunner(session, model.meta, seed);
    if (ep === 'webgpu') await runner.setupGpu();
    return runner;
  }

  /** 'gpu' when IO binding is active, otherwise 'cpu'. */
  get mode(): 'gpu' | 'cpu' {
    return this.useGpu ? 'gpu' : 'cpu';
  }

  /** WebGPU adapter description, for diagnostics. */
  get adapter(): string {
    return this.adapterDesc || 'unknown';
  }

  /**
   * Replace the perception basis `[identity, dx, dy]` and re-upload it.
   * The ONNX graph takes the kernel as an input, so no session rebuild needed.
   */
  setKernels(ident: number[], dx: number[], dy: number[]): void {
    this.kernels = [ident.slice(), dx.slice(), dy.slice()];
    this.weight = buildPerceptionWeight(this.channelN, this.kernels);
    if (this.useGpu && this.device && this.weightBuf) {
      this.device.queue.writeBuffer(this.weightBuf, 0, this.weight);
    }
  }

  getKernels(): number[][] {
    return this.kernels.map((k) => k.slice());
  }

  /** Resolve the GPU device and allocate the buffers (best-effort). */
  private async setupGpu(): Promise<void> {
    try {
      const device = await ort.env.webgpu.device;
      if (!device) return;
      this.device = device;

      const adapter = ort.env.webgpu.adapter as
        | { info?: unknown; adapterInfo?: unknown }
        | undefined;
      const info = (adapter?.info ?? adapter?.adapterInfo) as
        | {
            vendor?: string;
            architecture?: string;
            device?: string;
            description?: string;
          }
        | undefined;
      this.adapterDesc = info
        ? [info.vendor, info.architecture, info.device, info.description]
            .filter(Boolean)
            .join(' ') || 'unknown'
        : 'unknown';
      console.info('[nca] WebGPU adapter:', this.adapterDesc);

      this.allocGpu();
      this.reset();
    } catch (e) {
      console.warn('GPU IO binding unavailable; falling back to CPU tensors', e);
      this.freeGpu();
    }
  }

  /** Allocate state/mask/weight buffers + tensors for the current grid. */
  private allocGpu(): void {
    const device = this.device;
    if (!device) return;
    const { channelN: C, grid: H } = this;
    const W = H;
    this.stateBytes = C * H * W * 4;

    const alloc = (bytes: number, usage: number): GPUBuffer =>
      device.createBuffer({ size: Math.ceil(bytes / 16) * 16, usage });

    this.bufA = alloc(this.stateBytes, STATE_USAGE);
    this.bufB = alloc(this.stateBytes, STATE_USAGE);
    this.maskBuf = alloc(H * W * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.weightBuf = alloc(3 * C * 9 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.staging = alloc(
      this.stateBytes,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    );

    const dims = [1, C, H, W];
    this.tensorA = ort.Tensor.fromGpuBuffer(this.bufA, { dataType: 'float32', dims });
    this.tensorB = ort.Tensor.fromGpuBuffer(this.bufB, { dataType: 'float32', dims });
    this.maskTensor = ort.Tensor.fromGpuBuffer(this.maskBuf, {
      dataType: 'float32',
      dims: [1, 1, H, W],
    });
    this.weightTensor = ort.Tensor.fromGpuBuffer(this.weightBuf, {
      dataType: 'float32',
      dims: [3 * C, 1, 3, 3],
    });
    this.stepTensor = new ort.Tensor('float32', new Float32Array([1]), []);

    device.queue.writeBuffer(this.weightBuf, 0, this.weight);
    this.useGpu = true;
  }

  private freeGpu(): void {
    this.bufA?.destroy();
    this.bufB?.destroy();
    this.maskBuf?.destroy();
    this.weightBuf?.destroy();
    this.staging?.destroy();
    this.bufA = null;
    this.bufB = null;
    this.maskBuf = null;
    this.weightBuf = null;
    this.staging = null;
    this.tensorA = null;
    this.tensorB = null;
    this.maskTensor = null;
    this.weightTensor = null;
    this.stepTensor = null;
    this.useGpu = false;
  }

  /**
   * Run the CA on a different grid size. The ONNX model has dynamic H/W, so no
   * session rebuild is needed - we just reallocate buffers and reseed.
   */
  setGrid(size: number): void {
    if (size === this.grid) return;
    this.grid = size;
    this.mask = new Float32Array(size * size);
    this.queuedEdits = [];
    if (this.device) {
      try {
        this.freeGpu();
        this.allocGpu();
      } catch (e) {
        console.warn('failed to resize GPU buffers; using CPU tensors', e);
        this.freeGpu();
      }
    }
    this.reset();
  }

  /** Restore the single-cell seed state. */
  reset(seed?: number): void {
    if (seed !== undefined) this.seed = seed;
    this.rng = mulberry32(this.seed);
    this.x = seedState(this.channelN, this.grid);
    this.queuedEdits = [];
    this.cur = 0;
    this.stepCount = 0;
    this.uploadState();
  }

  /**
   * Apply an edit to the state.
   *
   * While a step is in flight the state buffers are being swapped, so the edit
   * is queued and applied to the fresh state once the step finishes; otherwise
   * it would be lost.
   */
  edit(fn: (x: Float32Array) => void): void {
    if (this.stepping) {
      this.queuedEdits.push(fn);
      return;
    }
    fn(this.x);
    this.uploadState();
  }

  private async flushEdits(): Promise<void> {
    if (this.queuedEdits.length === 0) return;
    const queued = this.queuedEdits;
    this.queuedEdits = [];
    if (this.useGpu) await this.sync(); // refresh the CPU mirror from the GPU
    for (const fn of queued) fn(this.x);
    this.uploadState();
  }

  private uploadState(): void {
    if (!this.useGpu || !this.device) return;
    const buf = this.cur === 0 ? this.bufA : this.bufB;
    if (buf) this.device.queue.writeBuffer(buf, 0, this.x);
  }

  /** Download the current GPU state into the CPU mirror (for rendering). */
  async sync(): Promise<void> {
    if (!this.useGpu || !this.device || !this.staging) return;
    const buf = this.cur === 0 ? this.bufA : this.bufB;
    if (!buf) return;

    const t0 = performance.now();
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buf, 0, this.staging, 0, this.stateBytes);
    this.device.queue.submit([encoder.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ);
    this.x = new Float32Array(
      this.staging.getMappedRange(0, this.stateBytes).slice(0),
    );
    this.staging.unmap();
    this.perf.syncMs += performance.now() - t0;
  }

  /** Advance the CA by one step. */
  async step(): Promise<void> {
    const { channelN: C, grid: H } = this;
    const W = H;
    const plane = H * W;

    const tMask = performance.now();
    for (let i = 0; i < plane; i++) {
      this.mask[i] = this.rng() < this.fireRate ? 1 : 0;
    }
    const gpu =
      this.useGpu &&
      this.device !== null &&
      this.maskBuf !== null &&
      this.maskTensor !== null;
    if (gpu) this.device!.queue.writeBuffer(this.maskBuf!, 0, this.mask);
    this.perf.maskMs += performance.now() - tMask;

    this.stepping = true;
    const tRun = performance.now();
    try {
      if (gpu) {
        const inTensor = this.cur === 0 ? this.tensorA : this.tensorB;
        const outTensor = this.cur === 0 ? this.tensorB : this.tensorA;
        const feeds: Record<string, ort.Tensor> = {
          x: inTensor!,
          perception_weight: this.weightTensor!,
          fire_mask: this.maskTensor!,
          step_size: this.stepTensor!,
        };
        // Pre-allocated output: ORT writes straight into our other state buffer.
        await this.session.run(feeds, { x_next: outTensor! });
        this.cur ^= 1;
      } else {
        const feeds: Record<string, ort.Tensor> = {
          x: new ort.Tensor('float32', this.x, [1, C, H, W]),
          perception_weight: new ort.Tensor('float32', this.weight, [3 * C, 1, 3, 3]),
          fire_mask: new ort.Tensor('float32', this.mask, [1, 1, H, W]),
          step_size: new ort.Tensor('float32', new Float32Array([1]), []),
        };
        const out = await this.session.run(feeds);
        this.x = (out['x_next'] as ort.Tensor).data as Float32Array;
      }
    } finally {
      this.stepping = false;
    }
    this.perf.runMs += performance.now() - tRun;

    await this.flushEdits();
    this.perf.steps++;
    this.stepCount++;
  }

  async dispose(): Promise<void> {
    this.freeGpu();
    await this.session.release();
  }
}
