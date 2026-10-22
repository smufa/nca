import * as ort from 'onnxruntime-web/wasm';
import type { ManifestModel } from './models';
import {
  buildPerceptionWeight,
  CANONICAL_KERNELS,
  modelKernels,
  mulberry32,
  seedState,
} from './nca';
import type { Rng } from './nca';
import type { CaEngine } from './engine';

// Single-threaded WASM core: enough for the compatibility path, and it avoids
// needing cross-origin-isolation (SharedArrayBuffer) headers.
ort.env.wasm.numThreads = 1;

// `?log=info` / `?log=verbose` surfaces ORT's logs.
const _log = new URLSearchParams(location.search).get('log');
if (_log) ort.env.logLevel = _log === 'verbose' ? 'verbose' : 'info';

/**
 * The compatibility backend: one trained NCA run through onnxruntime-web's WASM
 * EP on CPU tensors.
 *
 * The state is a plain `Float32Array` (`x`) that both gets fed in and comes back
 * out, so there is nothing to upload or download - `sync()` is a no-op. Use the
 * `shader` engine for speed; this exists for models whose size the WGSL step
 * cannot represent (see `ShaderRunner.supports`).
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
  /** Number of targets for a conditional model (0 = unconditional). */
  private condN = 0;
  private condTensor: ort.Tensor | null = null;
  private rng: Rng;
  private seed: number;
  private stepping = false;
  private queuedEdits: Array<(x: Float32Array) => void> = [];

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
    this.condN = meta.cond_n ?? 0;
    if (this.condN) this.setCond(0);
    this.mask = new Float32Array(this.grid * this.grid);
    this.rng = mulberry32(seed);
    this.x = seedState(this.channelN, this.grid);
  }

  static async create(model: ManifestModel, seed = 1): Promise<CaRunner> {
    const session = await ort.InferenceSession.create(`/models/${model.path}`, {
      executionProviders: ['wasm'],
    });
    return new CaRunner(session, model.meta, seed);
  }

  get mode(): string {
    return 'cpu';
  }

  get adapter(): string {
    return 'onnxruntime-web wasm';
  }

  /**
   * Select which target a conditional model should grow.
   *
   * The state is deliberately left alone: switching the one-hot mid-run should
   * *convert* whatever is on the grid towards the new target. Use `reset()` to go
   * back to the seed.
   */
  setCond(index: number): void {
    if (!this.condN) return;
    const i = Math.max(0, Math.min(this.condN - 1, index | 0));
    const data = new Float32Array(this.condN);
    data[i] = 1;
    this.condTensor = new ort.Tensor('float32', data, [1, this.condN, 1, 1]);
  }

  /**
   * Replace the perception basis `[identity, dx, dy]`.
   * The ONNX graph takes the kernel as an input, so no session rebuild is needed.
   */
  setKernels(ident: number[], dx: number[], dy: number[]): void {
    this.kernels = [ident.slice(), dx.slice(), dy.slice()];
    this.weight = buildPerceptionWeight(this.channelN, this.kernels);
  }

  getKernels(): number[][] {
    return this.kernels.map((k) => k.slice());
  }

  /**
   * Run the CA on a different grid size. The ONNX model has dynamic H/W, so no
   * session rebuild is needed - just reseed.
   */
  setGrid(size: number): void {
    if (size === this.grid) return;
    this.grid = size;
    this.mask = new Float32Array(size * size);
    this.queuedEdits = [];
    this.reset();
  }

  /** Restore the single-cell seed state. */
  reset(seed?: number): void {
    if (seed !== undefined) this.seed = seed;
    this.rng = mulberry32(this.seed);
    this.x = seedState(this.channelN, this.grid);
    this.queuedEdits = [];
    this.stepCount = 0;
  }

  /**
   * Apply an edit to the state.
   *
   * While a step is in flight the state array is swapped for the result, so the
   * edit is queued and applied to the fresh state once the step finishes.
   */
  edit(fn: (x: Float32Array) => void): void {
    if (this.stepping) {
      this.queuedEdits.push(fn);
      return;
    }
    fn(this.x);
  }

  private flushEdits(): void {
    if (this.queuedEdits.length === 0) return;
    const queued = this.queuedEdits;
    this.queuedEdits = [];
    for (const fn of queued) fn(this.x);
  }

  /** No-op: `x` is the state, so there is never anything to download. */
  async sync(): Promise<void> {}

  /** Advance the CA by one step. */
  async step(): Promise<void> {
    const { channelN: C, grid: H } = this;
    const W = H;
    const plane = H * W;

    const tMask = performance.now();
    for (let i = 0; i < plane; i++) {
      this.mask[i] = this.rng() < this.fireRate ? 1 : 0;
    }
    this.perf.maskMs += performance.now() - tMask;

    this.stepping = true;
    const tRun = performance.now();
    try {
      const feeds: Record<string, ort.Tensor> = {
        x: new ort.Tensor('float32', this.x, [1, C, H, W]),
        perception_weight: new ort.Tensor('float32', this.weight, [3 * C, 1, 3, 3]),
        fire_mask: new ort.Tensor('float32', this.mask, [1, 1, H, W]),
        step_size: new ort.Tensor('float32', new Float32Array([1]), []),
      };
      if (this.condTensor) feeds['cond'] = this.condTensor;
      const out = await this.session.run(feeds);
      this.x = (out['x_next'] as ort.Tensor).data as Float32Array;
    } finally {
      this.stepping = false;
    }
    this.perf.runMs += performance.now() - tRun;

    this.flushEdits();
    this.perf.steps++;
    this.stepCount++;
  }

  async dispose(): Promise<void> {
    await this.session.release();
  }
}
