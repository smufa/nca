/**
 * The frame loop and the transport controls (play / step / reset / grid / speed).
 *
 * The loop is the only place that runs steps on a timer; everything interactive
 * (brush, kernels, conditioning) mutates state synchronously and asks for a redraw.
 * `app.busy` guards every entry point so a step can never overlap a session swap.
 */

import { addSteps, app, countFrame, maybeUpdatePerf, setBusy, setStatus } from './app';
import { tickStroke } from './brush';
import {
  addViewBtn,
  engineSelect,
  gridSizeSelect,
  modelSelect,
  playBtn,
  resetBtn,
  speedInput,
  speedVal,
  stepBtn,
  viewSizeInput,
  viewSizeVal,
} from './dom';
import { updateInfo } from './info';
import { autoRotate, buildKernelEditors, resetKernelsUI } from './kernels';
import { loadModels, markPickerSupport, selectModel, selectedModel } from './picker';
import { createView, drawAll, nextHiddenMapping, resizeViews } from './views';

function setPlaying(playing: boolean): void {
  app.playing = playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  app.lastT = performance.now();
  app.accumulator = 0;
}

async function singleStep(): Promise<void> {
  const runner = app.runner;
  if (!runner || app.busy) return;
  setBusy(true);
  try {
    await runner.step();
    if (!runner.capabilities.gpuRender) await runner.sync();
  } catch (e) {
    console.error(e);
    setStatus(`step failed: ${(e as Error).message}`, 'err');
  } finally {
    setBusy(false);
  }
  drawAll();
  updateInfo();
}

function applyViewSize(px: number): void {
  document.documentElement.style.setProperty('--view-size', `${px}px`);
  viewSizeVal.textContent = String(px);
}

async function frame(t: number): Promise<void> {
  requestAnimationFrame(frame);
  countFrame();
  maybeUpdatePerf();
  tickStroke();
  const dt = Math.min((t - app.lastT) / 1000, 0.1);
  app.lastT = t;

  autoRotate(dt);

  const runner = app.runner;
  if (!app.playing || !runner || app.busy) return;

  app.accumulator += dt * Number(speedInput.value);
  let n = Math.floor(app.accumulator);
  if (n <= 0) return;
  app.accumulator -= n;
  n = Math.min(n, 24);

  setBusy(true);
  const t0 = performance.now();
  let done = 0;
  try {
    for (let i = 0; i < n; i++) {
      await runner.step();
      done++;
    }
    if (!runner.capabilities.gpuRender) await runner.sync();
  } catch (e) {
    console.error(e);
    setStatus(`step failed: ${(e as Error).message}`, 'err');
    setPlaying(false);
  } finally {
    setBusy(false);
  }
  addSteps(done, performance.now() - t0);
  drawAll();
  updateInfo();
}

export function wireControls(): void {
  playBtn.addEventListener('click', () => {
    if (!app.runner) return;
    setPlaying(!app.playing);
  });

  stepBtn.addEventListener('click', () => void singleStep());

  resetBtn.addEventListener('click', () => {
    const runner = app.runner;
    if (!runner) return;
    runner.reset();
    drawAll();
    updateInfo();
  });

  addViewBtn.addEventListener('click', () => {
    if (!app.runner) return;
    createView(nextHiddenMapping());
    drawAll();
    updateInfo();
  });

  speedInput.addEventListener('input', () => {
    speedVal.textContent = speedInput.value;
  });

  modelSelect.addEventListener('change', () => {
    // The engine/model pair decides what is runnable, so re-mark before reloading.
    markPickerSupport();
    const m = selectedModel();
    if (m) void selectModel(m);
  });

  engineSelect.value = app.ep;
  engineSelect.addEventListener('change', () => {
    app.ep = engineSelect.value as typeof app.ep;
    localStorage.setItem('nca.ep', app.ep);
    markPickerSupport();
    const m = selectedModel();
    if (m) void selectModel(m);
  });

  gridSizeSelect.addEventListener('change', () => {
    const runner = app.runner;
    if (!runner) return;
    runner.setGrid(Number(gridSizeSelect.value));
    resizeViews(runner.grid);
    drawAll();
    updateInfo();
  });

  viewSizeInput.addEventListener('input', () => {
    applyViewSize(Number(viewSizeInput.value));
  });

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      playBtn.click();
    } else if (e.key === 's') {
      stepBtn.click();
    } else if (e.key === 'r') {
      resetBtn.click();
    } else if (e.key === 'v') {
      addViewBtn.click();
    }
  });
}

/** Boot the playground: build the static widgets, load the manifest, start drawing. */
export function start(): void {
  if (!('gpu' in navigator)) {
    setStatus('WebGPU is not available in this browser', 'err');
    return;
  }
  buildKernelEditors();
  resetKernelsUI();
  applyViewSize(Number(viewSizeInput.value));
  void loadModels();
  requestAnimationFrame(frame);
}
