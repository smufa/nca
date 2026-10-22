/** Shared interface for the CA backends (ORT and the native WGSL engine). */
import type { ModelMeta } from './models';
import type { ChannelMapping } from './nca';

export interface Perf {
  maskMs: number;
  runMs: number;
  syncMs: number;
  steps: number;
}

export type BrushOp = 'one' | 'zero' | 'mul';

export interface CaEngine {
  readonly meta: ModelMeta;
  readonly channelN: number;
  grid: number;
  fireRate: number;
  stepCount: number;
  /** CPU mirror of the state, `[C*H*W]` (refreshed by `sync()`). */
  x: Float32Array;
  readonly perf: Perf;
  /** 'gpu' (IO binding) / 'shader' (WGSL) / 'cpu'. */
  readonly mode: string;
  readonly adapter: string;

  step(): Promise<void>;
  sync(): Promise<void>;
  reset(seed?: number): void;
  edit(fn: (x: Float32Array) => void): void;
  setGrid(size: number): void;
  setKernels(ident: number[], dx: number[], dy: number[]): void;
  /** The perception kernels the engine is currently using. */
  getKernels(): number[][];
  dispose(): Promise<void>;

  /**
   * If present, views render via WebGPU straight from the GPU state (no
   * download). Engines without it use the Canvas2D + `x` path.
   */
  renderTo?(canvas: HTMLCanvasElement, mapping: ChannelMapping): void;

  /**
   * If present, brush strokes are applied on the GPU (no full-state upload).
   * `channels` are the channel indices to edit.
   */
  brush?(
    op: BrushOp,
    cx: number,
    cy: number,
    radius: number,
    factor: number,
    channels: number[],
  ): void;
}
