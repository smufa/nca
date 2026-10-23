/**
 * Render panels: one canvas per view, each with its own channel mapping.
 *
 * A view is the whole presentation contract: which state channels become R/G/B, an
 * optional alpha channel (premultiplied; "off" means opaque) and the background to
 * composite over. `nca.ts:mapChannels` and `wgsl/draw.wgsl` are the two
 * implementations of that mapping, so the two engines draw the same picture.
 */

import type { ChannelMapping } from '../nca';
import { mapChannels } from '../nca';
import { app } from './app';
import { beginStroke, endStroke, moveStroke } from './brush';
import { fillChannelSelect } from './channels';
import { viewsEl } from './dom';

/** A render panel: its own canvas + channel mapping selectors. */
export interface View {
  root: HTMLDivElement;
  canvas: HTMLCanvasElement;
  ctx2d: CanvasRenderingContext2D | null;
  image: ImageData | null;
  mapping: ChannelMapping;
  selects: Record<'r' | 'g' | 'b' | 'a' | 'bg', HTMLSelectElement>;
}

export function drawView(view: View): void {
  const runner = app.runner;
  if (!runner) return;
  if (runner.capabilities.gpuRender) {
    runner.renderTo!(view.canvas, view.mapping);
    return;
  }
  if (!view.ctx2d || !view.image) return;
  mapChannels(runner.x, runner.grid, runner.grid, view.mapping, view.image.data);
  view.ctx2d.putImageData(view.image, 0, 0);
}

export function drawAll(): void {
  for (const view of app.views) drawView(view);
}

/** Resize every view's canvas/image after a grid change (mappings preserved). */
export function resizeViews(grid: number): void {
  for (const view of app.views) {
    view.canvas.width = grid;
    view.canvas.height = grid;
    if (view.ctx2d) view.image = view.ctx2d.createImageData(grid, grid);
  }
}

export function createView(mapping: ChannelMapping): View {
  const runner = app.runner;
  if (!runner) throw new Error('no model loaded');
  const channelN = runner.channelN;
  const grid = runner.grid;

  const root = document.createElement('div');
  root.className = 'view';

  const head = document.createElement('div');
  head.className = 'view-head';
  const channels = document.createElement('div');
  channels.className = 'channels';
  head.appendChild(channels);

  const mkChannel = (label: string, includeOff: boolean): HTMLSelectElement => {
    const wrap = document.createElement('label');
    wrap.textContent = label;
    const sel = document.createElement('select');
    fillChannelSelect(sel, channelN, includeOff);
    wrap.appendChild(sel);
    channels.appendChild(wrap);
    return sel;
  };

  const singleWrap = document.createElement('label');
  singleWrap.className = 'single';
  singleWrap.title = 'Show a single channel as grayscale';
  const singleBox = document.createElement('input');
  singleBox.type = 'checkbox';
  singleWrap.appendChild(singleBox);
  singleWrap.appendChild(document.createTextNode('1ch'));
  channels.appendChild(singleWrap);

  const rSel = mkChannel('R', false);
  const gSel = mkChannel('G', false);
  const bSel = mkChannel('B', false);
  const aSel = mkChannel('A', true);

  const bgWrap = document.createElement('label');
  bgWrap.textContent = 'bg';
  const bgSel = document.createElement('select');
  for (const [value, text] of [
    ['1', 'white'],
    ['0', 'black'],
  ]) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    bgSel.appendChild(opt);
  }
  bgWrap.appendChild(bgSel);
  channels.appendChild(bgWrap);

  const removeBtn = document.createElement('button');
  removeBtn.className = 'remove';
  removeBtn.textContent = '×';
  removeBtn.title = 'Remove view';
  head.appendChild(removeBtn);

  const canvas = document.createElement('canvas');
  canvas.width = grid;
  canvas.height = grid;
  const onGpu = runner.capabilities.gpuRender;
  let ctx2d: CanvasRenderingContext2D | null = null;
  let image: ImageData | null = null;
  if (!onGpu) {
    ctx2d = canvas.getContext('2d');
    if (!ctx2d) throw new Error('2d canvas context unavailable');
    image = ctx2d.createImageData(grid, grid);
  }

  root.appendChild(head);
  root.appendChild(canvas);
  viewsEl.appendChild(root);

  rSel.value = String(mapping.r);
  gSel.value = String(mapping.g);
  bSel.value = String(mapping.b);
  aSel.value = String(mapping.a);
  bgSel.value = String(mapping.bg);

  const view: View = {
    root,
    canvas,
    ctx2d,
    image,
    mapping,
    selects: { r: rSel, g: gSel, b: bSel, a: aSel, bg: bgSel },
  };

  const sync = (): void => {
    // Single-channel mode: R drives G and B, which are disabled.
    if (singleBox.checked) {
      gSel.value = rSel.value;
      bSel.value = rSel.value;
    }
    gSel.disabled = singleBox.checked;
    bSel.disabled = singleBox.checked;
    view.mapping = {
      r: Number(rSel.value),
      g: Number(gSel.value),
      b: Number(bSel.value),
      a: Number(aSel.value),
      bg: Number(bgSel.value),
    };
    drawView(view);
  };
  for (const sel of [rSel, gSel, bSel, aSel, bgSel, singleBox]) {
    sel.addEventListener('change', sync);
  }

  removeBtn.addEventListener('click', () => {
    if (app.views.length <= 1) return; // always keep one view
    app.views = app.views.filter((v) => v !== view);
    root.remove();
  });

  canvas.style.touchAction = 'none';
  canvas.style.cursor = app.tool === 'none' ? 'default' : 'crosshair';

  let painting = false;

  const toCell = (e: PointerEvent): [number, number] => {
    const rect = canvas.getBoundingClientRect();
    const g = app.runner!.grid;
    const cx = Math.floor(((e.clientX - rect.left) / rect.width) * g);
    const cy = Math.floor(((e.clientY - rect.top) / rect.height) * g);
    return [Math.max(0, Math.min(g - 1, cx)), Math.max(0, Math.min(g - 1, cy))];
  };

  canvas.addEventListener('pointerdown', (e) => {
    if (app.tool === 'none' || !app.runner) return;
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    painting = true;
    beginStroke(view, ...toCell(e), e.pointerId);
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!painting) return;
    moveStroke(view, ...toCell(e), e.pointerId);
  });

  const onEnd = (e: PointerEvent): void => {
    if (!painting) return;
    painting = false;
    endStroke(view, e.pointerId);
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  };
  canvas.addEventListener('pointerup', onEnd);
  canvas.addEventListener('pointercancel', onEnd);

  app.views.push(view);
  return view;
}

/** Reset the views for a newly loaded model; the first view is RGB over white. */
export function rebuildViews(): void {
  viewsEl.innerHTML = '';
  app.views = [];
  const runner = app.runner;
  if (!runner) return;
  createView({ r: 0, g: 1, b: 2, a: runner.channelN > 3 ? 3 : -1, bg: 1 });
}

/** Default mapping for a new view: the next hidden channel as a grayscale map. */
export function nextHiddenMapping(): ChannelMapping {
  const channelN = app.runner!.channelN;
  const start = 4;
  if (channelN <= start) return { r: 0, g: 0, b: 0, a: -1, bg: 0 };
  const ch = start + ((app.views.length - 1) % (channelN - start));
  return { r: ch, g: ch, b: ch, a: -1, bg: 0 };
}
