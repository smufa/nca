/** Shared interface for the CA backends (ORT and the native WGSL engine). */
import type { ModelMeta } from './models';
import type { ChannelMapping } from './nca';

/**
 * What an engine can run. Advertised as data so the UI never feature-detects
 * with `?.` and a third backend is a new object, not a new branch in `main.ts`.
 *
 * `Infinity` means "no engine limit" (the ORT path runs any shape the graph
 * contains); a finite value names a real hardware/shader bound and must be
 * justified in the engine's comment.
 */
export interface Capabilities {
  readonly kind: 'ort' | 'shader';
  readonly label: string;
  /** Can select a conditioning vector (`setCond`/`setCondVector`). */
  readonly cond: boolean;
  /** Brush strokes applied on the GPU (no full-state upload). */
  readonly gpuBrush: boolean;
  /** Rendering straight from GPU state (`renderTo`). */
  readonly gpuRender: boolean;
  /**
   * Work-skipping: the update rule runs only for cells the fire mask selects
   * (`compact` + `finalize` + `step_list` instead of `pass1`). Needs one hidden
   * layer, because only the shallow rule has a list-driven entry point.
   */
  readonly workSkip: boolean;
  readonly maxC: number;
  readonly maxHidden: number;
  readonly maxLayers: number;
  readonly maxCond: number;
  /** Max `3*C + cond_n`, i.e. the perception vector the step must hold. */
  readonly maxPerc: number;
  /** State buffers are `3 * C * pixels * 4` bytes; this is a pixel budget. */
  readonly maxGridPx: number;
}

/** The answer to "can this engine run this model", with the reason when it cannot. */
export type Support = { ok: true } | { ok: false; reason: string };

/**
 * The one place a spec is checked against an engine's capabilities.
 *
 * Both engines call this with their own `Capabilities`; every failed check names
 * the value and the bound, so the UI can say *why* a model fell back to WASM
 * instead of silently switching.
 */
export function checkSpec(caps: Capabilities, meta: ModelMeta): Support {
  const c = meta.channel_n;
  const cond = meta.cond_n ?? 0;
  const hidden = meta.hidden_n;
  const layers = meta.hidden_layers ?? 1;
  const perc = 3 * c + cond;
  const gridPx = meta.grid * meta.grid;
  if (c < 4) return { ok: false, reason: `C=${c} < 4 (shader needs 4 channels: RGB + alpha)` };
  if (c > caps.maxC) return { ok: false, reason: `C=${c} > ${caps.maxC}` };
  if (hidden % 4 !== 0) return { ok: false, reason: `hidden=${hidden} is not a multiple of 4 (vec4 lanes)` };
  if (hidden > caps.maxHidden) return { ok: false, reason: `hidden=${hidden} > ${caps.maxHidden}` };
  if (layers > caps.maxLayers) return { ok: false, reason: `${layers} hidden layers > ${caps.maxLayers}` };
  if (cond > caps.maxCond) return { ok: false, reason: `cond_n=${cond} > ${caps.maxCond}` };
  if (perc > caps.maxPerc) {
    return { ok: false, reason: `3*C+cond_n=${perc} > ${caps.maxPerc} (perception buffer)` };
  }
  if (gridPx > caps.maxGridPx) {
    const mb = (3 * c * gridPx * 4) / (1024 * 1024);
    return { ok: false, reason: `grid ${meta.grid}² needs ${mb.toFixed(0)} MB of state buffers` };
  }
  return { ok: true };
}

export interface Perf {
  maskMs: number;
  runMs: number;
  syncMs: number;
  steps: number;
}

export type BrushOp = 'one' | 'zero' | 'mul';

export interface CaEngine {
  readonly meta: ModelMeta;
  /** What this engine can run; branch on this instead of probing `?.`. */
  readonly capabilities: Capabilities;
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
  /**
   * Conditional models: choose which target to grow (index into meta.targets).
   * Both engines implement these; `capabilities.cond` says whether a model *has*
   * a conditioning input, not whether the engine can set one.
   */
  setCond(index: number): void;
  /**
   * Conditional models: set the conditioning vector directly, for values that are
   * not one-hot (e.g. 0.5 of one target + 0.5 of another).
   */
  setCondVector(values: number[]): void;
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
