/**
 * Every DOM element the UI touches, looked up once at module load.
 *
 * `index.html` is the contract: a missing id is a programming error, so `el()`
 * throws instead of returning null (which would only fail later, further away).
 */

export function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
}

export const viewsEl = el<HTMLDivElement>('views');
export const statusEl = el<HTMLSpanElement>('status');
export const perfEl = el<HTMLSpanElement>('perf');
export const modelSelect = el<HTMLSelectElement>('model');
export const targetField = el<HTMLDivElement>('targetField');
export const condSlidersEl = el<HTMLDivElement>('condSliders');
export const condValEl = el<HTMLElement>('condVal');
export const condResetBtn = el<HTMLButtonElement>('condReset');
export const engineSelect = el<HTMLSelectElement>('engine');
export const gridSizeSelect = el<HTMLSelectElement>('gridSize');
export const playBtn = el<HTMLButtonElement>('play');
export const stepBtn = el<HTMLButtonElement>('step');
export const resetBtn = el<HTMLButtonElement>('reset');
export const speedInput = el<HTMLInputElement>('speed');
export const speedVal = el<HTMLElement>('speedVal');
export const infoEl = el<HTMLDListElement>('info');
export const addViewBtn = el<HTMLButtonElement>('addView');
export const angleInput = el<HTMLInputElement>('angle');
export const angleVal = el<HTMLElement>('angleVal');
export const linkedInput = el<HTMLInputElement>('linked');
export const autoRotateInput = el<HTMLInputElement>('autoRotate');
export const kernelGridsEl = el<HTMLDivElement>('kernelGrids');
export const resetKernelsBtn = el<HTMLButtonElement>('resetKernels');
export const toolsEl = el<HTMLDivElement>('tools');
export const brushRadiusInput = el<HTMLInputElement>('brushRadius');
export const brushRadiusVal = el<HTMLElement>('brushRadiusVal');
export const brushFactorInput = el<HTMLInputElement>('brushFactor');
export const brushAlphaInput = el<HTMLInputElement>('brushAlpha');
export const viewSizeInput = el<HTMLInputElement>('viewSize');
export const viewSizeVal = el<HTMLElement>('viewSizeVal');
