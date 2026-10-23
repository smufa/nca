/**
 * Brush: the interaction half of the playground.
 *
 * A stroke paints a disc of cells in the channels the view displays (plus alpha,
 * when enabled) with one of three ops: set 1, set 0, scale by a factor. Two
 * engines, one behaviour: `runner.edit` ships the touched indices to WASM, or the
 * shader engine's GPU brush pass writes them in place - no state upload either way.
 *
 * Strokes are coordinates, not pixels: both the WASM path and `wgsl/brush.wgsl`
 * index `y*grid + x` in a channel-first layout.
 */

import { app } from './app';
import {
  brushAlphaInput,
  brushFactorInput,
  brushRadiusInput,
  brushRadiusVal,
  toolsEl,
} from './dom';
import { drawAll, type View } from './views';

export interface Stroke {
  view: View;
  pointerId: number;
  lastX: number;
  lastY: number;
  curX: number;
  curY: number;
}

let stroke: Stroke | null = null;

export function brushTargets(view: View): number[] {
  const runner = app.runner;
  if (!runner) return [];
  const set = new Set<number>([view.mapping.r, view.mapping.g, view.mapping.b]);
  if (app.brushAlpha) set.add(3);
  return [...set].filter((c) => c >= 0 && c < runner.channelN);
}

export function paintDot(view: View, cx: number, cy: number): void {
  const runner = app.runner;
  const tool = app.tool;
  if (!runner || tool === 'none') return;
  const targets = brushTargets(view);
  if (targets.length === 0) return;
  const factor = app.brushFactor;
  const radius = app.brushRadius;

  // GPU engines apply the stroke in a compute pass - no state upload.
  if (runner.capabilities.gpuBrush) {
    runner.brush!(tool, cx, cy, radius, factor, targets);
    return;
  }

  const grid = runner.grid;
  const plane = grid * grid;
  const r2 = radius * radius;
  const y0 = Math.max(0, cy - radius);
  const y1 = Math.min(grid - 1, cy + radius);
  const x0 = Math.max(0, cx - radius);
  const x1 = Math.min(grid - 1, cx + radius);

  const indices: number[] = [];
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx;
      const dy = y - cy;
      if (dx * dx + dy * dy > r2) continue;
      const i = y * grid + x;
      for (const c of targets) indices.push(c * plane + i);
    }
  }
  if (indices.length === 0) return;

  runner.edit((x) => {
    for (const idx of indices) {
      if (tool === 'one') x[idx] = 1;
      else if (tool === 'zero') x[idx] = 0;
      else x[idx] *= factor;
    }
  });
}

/** Interpolate a stroke between two cells so fast drags leave no gaps. */
export function paintLine(view: View, x0: number, y0: number, x1: number, y1: number): void {
  const n = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    paintDot(view, Math.round(x0 + (x1 - x0) * t), Math.round(y0 + (y1 - y0) * t));
  }
}

/** Re-apply the brush at the held position every frame. */
export function tickStroke(): void {
  if (!stroke || !app.runner) return;
  paintDot(stroke.view, stroke.curX, stroke.curY);
  drawAll();
}

export function beginStroke(view: View, cx: number, cy: number, pointerId: number): void {
  stroke = { view, pointerId, lastX: cx, lastY: cy, curX: cx, curY: cy };
  paintDot(view, cx, cy);
  drawAll();
}

export function moveStroke(view: View, cx: number, cy: number, pointerId: number): void {
  if (!stroke || stroke.view !== view || stroke.pointerId !== pointerId || !app.runner) return;
  paintLine(view, stroke.lastX, stroke.lastY, cx, cy);
  stroke.lastX = cx;
  stroke.lastY = cy;
  stroke.curX = cx;
  stroke.curY = cy;
  drawAll();
}

export function endStroke(view: View, pointerId: number): void {
  if (stroke && stroke.view === view && stroke.pointerId === pointerId) stroke = null;
}

export function updateCursors(): void {
  for (const view of app.views) {
    view.canvas.style.cursor = app.tool === 'none' ? 'default' : 'crosshair';
  }
}

export function wireBrushControls(): void {
  app.brushRadius = Number(brushRadiusInput.value);
  app.brushFactor = Number(brushFactorInput.value);
  app.brushAlpha = brushAlphaInput.checked;
  brushRadiusVal.textContent = String(app.brushRadius);

  for (const btn of toolsEl.querySelectorAll<HTMLButtonElement>('button[data-tool]')) {
    btn.addEventListener('click', () => {
      app.tool = (btn.dataset.tool as typeof app.tool) ?? 'none';
      for (const b of toolsEl.querySelectorAll('button[data-tool]')) {
        b.classList.toggle('active', b === btn);
      }
      updateCursors();
    });
  }

  brushRadiusInput.addEventListener('input', () => {
    app.brushRadius = Number(brushRadiusInput.value);
    brushRadiusVal.textContent = String(app.brushRadius);
  });
  brushFactorInput.addEventListener('input', () => {
    app.brushFactor = Number(brushFactorInput.value);
  });
  brushAlphaInput.addEventListener('change', () => {
    app.brushAlpha = brushAlphaInput.checked;
  });
}
