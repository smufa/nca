import * as ort from 'onnxruntime-web/webgpu';
import type { ManifestModel } from './models';
import { mulberry32, perceptionWeight, seedState } from './nca';
import type { Rng } from './nca';

// Single-threaded WASM core is enough: the heavy lifting runs on the WebGPU EP,
// and this avoids needing cross-origin-isolation (SharedArrayBuffer) headers.
ort.env.wasm.numThreads = 1;

/**
 * Runs one trained NCA on the GPU. The ONNX graph is a single deterministic
 * step; the stochastic update mask is generated here each step.
 */
export class CaRunner {
  readonly meta: ManifestModel['meta'];
  readonly channelN: number;
  readonly grid: number;
  fireRate: number;
  stepCount = 0;
  x: Float32Array;

  private session: ort.InferenceSession;
  private weight: Float32Array;
  private mask: Float32Array;
  private rng: Rng;
  private seed: number;

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
    this.weight = perceptionWeight(this.channelN);
    this.mask = new Float32Array(this.grid * this.grid);
    this.rng = mulberry32(seed);
    this.x = seedState(this.channelN, this.grid);
  }

  static async create(model: ManifestModel, seed = 1): Promise<CaRunner> {
    const session = await ort.InferenceSession.create(`/models/${model.path}`, {
      executionProviders: ['webgpu'],
    });
    return new CaRunner(session, model.meta, seed);
  }

  /** Restore the single-cell seed state. */
  reset(seed?: number): void {
    if (seed !== undefined) this.seed = seed;
    this.rng = mulberry32(this.seed);
    this.x = seedState(this.channelN, this.grid);
    this.stepCount = 0;
  }

  /** Advance the CA by one step. */
  async step(): Promise<void> {
    const { channelN: C, grid: H } = this;
    const W = H;
    const plane = H * W;

    for (let i = 0; i < plane; i++) {
      this.mask[i] = this.rng() < this.fireRate ? 1 : 0;
    }

    const feeds: Record<string, ort.Tensor> = {
      x: new ort.Tensor('float32', this.x, [1, C, H, W]),
      perception_weight: new ort.Tensor('float32', this.weight, [3 * C, 1, 3, 3]),
      fire_mask: new ort.Tensor('float32', this.mask, [1, 1, H, W]),
      step_size: new ort.Tensor('float32', new Float32Array([1]), []),
    };

    const out = await this.session.run(feeds);
    this.x = (out['x_next'] as ort.Tensor).data as Float32Array;
    this.stepCount++;
  }

  async dispose(): Promise<void> {
    await this.session.release();
  }
}
