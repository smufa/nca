/** Shared interface for the CA backends (ORT and the native WGSL engine). */
import type { ModelMeta } from './models';

export interface Perf {
  maskMs: number;
  runMs: number;
  syncMs: number;
  steps: number;
}

export interface CaEngine {
  readonly meta: ModelMeta;
  readonly channelN: number;
  grid: number;
  fireRate: number;
  stepCount: number;
  /** CPU mirror of the state, `[C*H*W]`, used for rendering and editing. */
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
  dispose(): Promise<void>;
}
