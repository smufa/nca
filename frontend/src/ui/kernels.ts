/**
 * Kernel lab: the perception kernel is a model *input*, so the UI can rewrite it
 * (rotate it, edit the Sobel taps) without re-exporting anything.
 *
 * Three 3×3 kernels in `[identity, dx, dy]` order, matching the ONNX graph's
 * perception weight. "Linked" means rotate a canonical Sobel pair by an angle;
 * unlocking the chain exposes the 27 taps for hand editing.
 */

import { CANONICAL_KERNELS, rotateKernels } from '../nca';
import { app } from './app';
import {
  angleInput,
  angleVal,
  autoRotateInput,
  kernelGridsEl,
  linkedInput,
  resetKernelsBtn,
} from './dom';

const KERNEL_NAMES = ['identity', 'dx', 'dy'];
const kernelInputs: HTMLInputElement[][] = [[], [], []];
let autoAngle = 0;

function swatch(inp: HTMLInputElement, v: number): void {
  const a = Math.min(1, Math.abs(v)) * 0.55;
  inp.style.background =
    v > 0
      ? `rgba(91,140,255,${a})`
      : v < 0
        ? `rgba(248,113,113,${a})`
        : 'var(--panel-2)';
}

function setKernelInputs(kernels: number[][]): void {
  for (let k = 0; k < 3; k++) {
    for (let i = 0; i < 9; i++) {
      const v = kernels[k][i];
      kernelInputs[k][i].value = String(v);
      swatch(kernelInputs[k][i], v);
    }
  }
}

function readKernelInputs(): number[][] {
  return kernelInputs.map((arr) => arr.map((inp) => Number(inp.value) || 0));
}

export function setLinked(linked: boolean): void {
  linkedInput.checked = linked;
  for (const arr of kernelInputs) for (const inp of arr) inp.disabled = linked;
}

export function applyKernels(kernels: number[][]): void {
  setKernelInputs(kernels);
  if (app.runner) app.runner.setKernels(kernels[0], kernels[1], kernels[2]);
}

function onKernelEdit(): void {
  if (linkedInput.checked) setLinked(false);
  const kernels = readKernelInputs();
  for (let k = 0; k < 3; k++) {
    for (let i = 0; i < 9; i++) swatch(kernelInputs[k][i], kernels[k][i]);
  }
  if (app.runner) app.runner.setKernels(kernels[0], kernels[1], kernels[2]);
}

export function resetKernelsUI(): void {
  autoAngle = 0;
  angleInput.value = '0';
  angleVal.textContent = '0°';
  setLinked(true);
  // Start from the kernels the model actually uses (learned, for --learn-perception).
  const kernels = app.runner
    ? app.runner.getKernels()
    : CANONICAL_KERNELS.map((k) => k.slice());
  applyKernels(kernels);
}

export function buildKernelEditors(): void {
  kernelGridsEl.innerHTML = '';
  for (const arr of kernelInputs) arr.length = 0;
  for (let k = 0; k < 3; k++) {
    const block = document.createElement('div');
    block.className = 'kernel';
    const title = document.createElement('div');
    title.className = 'kernel-title';
    title.textContent = KERNEL_NAMES[k];
    block.appendChild(title);
    const grid = document.createElement('div');
    grid.className = 'kgrid';
    for (let i = 0; i < 9; i++) {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.step = '0.125';
      inp.value = '0';
      inp.disabled = true;
      inp.addEventListener('input', onKernelEdit);
      grid.appendChild(inp);
      kernelInputs[k].push(inp);
    }
    block.appendChild(grid);
    kernelGridsEl.appendChild(block);
  }
}

/** One frame of the auto-rotate animation (`null` while it is off). */
export function autoRotate(dt: number): void {
  if (!autoRotateInput.checked) return;
  autoAngle = (autoAngle + dt * 45) % 360;
  if (!linkedInput.checked) setLinked(true);
  angleInput.value = String(Math.round(autoAngle));
  angleVal.textContent = `${Math.round(autoAngle)}°`;
  applyKernels(rotateKernels((autoAngle * Math.PI) / 180));
}

export function wireKernelControls(): void {
  linkedInput.addEventListener('change', () => {
    setLinked(linkedInput.checked);
    if (linkedInput.checked) {
      const deg = Number(angleInput.value);
      applyKernels(rotateKernels((deg * Math.PI) / 180));
    }
  });

  angleInput.addEventListener('input', () => {
    if (!linkedInput.checked) return;
    const deg = Number(angleInput.value);
    angleVal.textContent = `${deg}°`;
    applyKernels(rotateKernels((deg * Math.PI) / 180));
  });

  resetKernelsBtn.addEventListener('click', resetKernelsUI);
}
