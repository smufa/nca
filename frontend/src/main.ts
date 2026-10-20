import './style.css';
import { loadManifest } from './models';
import type { Manifest, ManifestModel } from './models';
import { CaRunner } from './runner';
import { mapChannels } from './nca';
import type { ChannelMapping } from './nca';

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
}

const viewsEl = el<HTMLDivElement>('views');
const statusEl = el<HTMLSpanElement>('status');
const modelSelect = el<HTMLSelectElement>('model');
const playBtn = el<HTMLButtonElement>('play');
const stepBtn = el<HTMLButtonElement>('step');
const resetBtn = el<HTMLButtonElement>('reset');
const speedInput = el<HTMLInputElement>('speed');
const speedVal = el<HTMLElement>('speedVal');
const infoEl = el<HTMLDListElement>('info');
const addViewBtn = el<HTMLButtonElement>('addView');
const toolsEl = el<HTMLDivElement>('tools');
const brushRadiusInput = el<HTMLInputElement>('brushRadius');
const brushRadiusVal = el<HTMLElement>('brushRadiusVal');
const brushFactorInput = el<HTMLInputElement>('brushFactor');
const brushAlphaInput = el<HTMLInputElement>('brushAlpha');

/** A render panel: its own canvas + channel mapping selectors. */
interface View {
  root: HTMLDivElement;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  image: ImageData;
  mapping: ChannelMapping;
  selects: Record<'r' | 'g' | 'b' | 'a' | 'bg', HTMLSelectElement>;
}

let manifest: Manifest | null = null;
let runner: CaRunner | null = null;
let views: View[] = [];
let playing = false;
let busy = false;
let accumulator = 0;
let lastT = performance.now();

type Tool = 'none' | 'one' | 'zero' | 'mul';
let tool: Tool = 'none';
let brushRadius = Number(brushRadiusInput.value);
let brushFactor = Number(brushFactorInput.value);
let brushAlpha = brushAlphaInput.checked;

function setStatus(text: string, kind: 'ok' | 'warn' | 'err' = 'ok'): void {
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`;
}

const CHANNEL_NAMES = ['R', 'G', 'B', 'A'];

function channelLabel(i: number): string {
  return i < CHANNEL_NAMES.length ? `${i} ${CHANNEL_NAMES[i]}` : String(i);
}

function fillChannelSelect(
  sel: HTMLSelectElement,
  channelN: number,
  includeOff: boolean,
): void {
  sel.innerHTML = '';
  if (includeOff) {
    const off = document.createElement('option');
    off.value = '-1';
    off.textContent = 'off';
    sel.appendChild(off);
  }
  for (let i = 0; i < channelN; i++) {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = channelLabel(i);
    sel.appendChild(opt);
  }
}

function drawView(view: View): void {
  if (!runner) return;
  mapChannels(runner.x, runner.grid, runner.grid, view.mapping, view.image.data);
  view.ctx.putImageData(view.image, 0, 0);
}

function drawAll(): void {
  for (const view of views) drawView(view);
}

/** Channels a brush stroke affects in a view: the ones it displays, plus alpha. */
function brushTargets(view: View): number[] {
  if (!runner) return [];
  const set = new Set<number>([view.mapping.r, view.mapping.g, view.mapping.b]);
  if (brushAlpha) set.add(3);
  return [...set].filter((c) => c >= 0 && c < runner!.channelN);
}

/** Apply the active tool to a disc of `brushRadius` cells centred on (cx, cy). */
function paintDot(view: View, cx: number, cy: number, painted: Set<number>): void {
  if (!runner || tool === 'none') return;
  const grid = runner.grid;
  const plane = grid * grid;
  const targets = brushTargets(view);
  if (targets.length === 0) return;

  const r2 = brushRadius * brushRadius;
  const y0 = Math.max(0, cy - brushRadius);
  const y1 = Math.min(grid - 1, cy + brushRadius);
  const x0 = Math.max(0, cx - brushRadius);
  const x1 = Math.min(grid - 1, cx + brushRadius);

  const indices: number[] = [];
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx;
      const dy = y - cy;
      if (dx * dx + dy * dy > r2) continue;
      const i = y * grid + x;
      if (painted.has(i)) continue; // at most once per cell per stroke
      painted.add(i);
      for (const c of targets) indices.push(c * plane + i);
    }
  }
  if (indices.length === 0) return;

  const op = tool;
  const factor = brushFactor;
  runner.edit((x) => {
    for (const idx of indices) {
      if (op === 'one') x[idx] = 1;
      else if (op === 'zero') x[idx] = 0;
      else x[idx] *= factor;
    }
  });
}

/** Interpolate a stroke between two cells so fast drags leave no gaps. */
function paintLine(
  view: View,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  painted: Set<number>,
): void {
  const n = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    paintDot(
      view,
      Math.round(x0 + (x1 - x0) * t),
      Math.round(y0 + (y1 - y0) * t),
      painted,
    );
  }
}

function createView(mapping: ChannelMapping): View {
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
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2d canvas context unavailable');

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
    ctx,
    image: ctx.createImageData(grid, grid),
    mapping,
    selects: { r: rSel, g: gSel, b: bSel, a: aSel, bg: bgSel },
  };

  const sync = (): void => {
    view.mapping = {
      r: Number(rSel.value),
      g: Number(gSel.value),
      b: Number(bSel.value),
      a: Number(aSel.value),
      bg: Number(bgSel.value),
    };
    drawView(view);
  };
  for (const sel of [rSel, gSel, bSel, aSel, bgSel]) {
    sel.addEventListener('change', sync);
  }

  removeBtn.addEventListener('click', () => {
    if (views.length <= 1) return; // always keep one view
    views = views.filter((v) => v !== view);
    root.remove();
  });

  canvas.style.touchAction = 'none';
  canvas.style.cursor = tool === 'none' ? 'default' : 'crosshair';

  let painting = false;
  let lastCell: [number, number] | null = null;
  let painted = new Set<number>();

  const toCell = (e: PointerEvent): [number, number] => {
    const rect = canvas.getBoundingClientRect();
    const g = runner!.grid;
    const cx = Math.floor(((e.clientX - rect.left) / rect.width) * g);
    const cy = Math.floor(((e.clientY - rect.top) / rect.height) * g);
    return [Math.max(0, Math.min(g - 1, cx)), Math.max(0, Math.min(g - 1, cy))];
  };

  canvas.addEventListener('pointerdown', (e) => {
    if (tool === 'none' || !runner) return;
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    painting = true;
    painted = new Set<number>();
    const [cx, cy] = toCell(e);
    lastCell = [cx, cy];
    paintDot(view, cx, cy, painted);
    drawAll();
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!painting || !runner) return;
    const [cx, cy] = toCell(e);
    if (lastCell) paintLine(view, lastCell[0], lastCell[1], cx, cy, painted);
    else paintDot(view, cx, cy, painted);
    lastCell = [cx, cy];
    drawAll();
  });

  const endStroke = (e: PointerEvent): void => {
    if (!painting) return;
    painting = false;
    lastCell = null;
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  };
  canvas.addEventListener('pointerup', endStroke);
  canvas.addEventListener('pointercancel', endStroke);

  views.push(view);
  return view;
}

/** Reset the views for a newly loaded model; the first view is RGB over white. */
function rebuildViews(): void {
  viewsEl.innerHTML = '';
  views = [];
  if (!runner) return;
  createView({ r: 0, g: 1, b: 2, a: runner.channelN > 3 ? 3 : -1, bg: 1 });
}

/** Default mapping for a new view: the next hidden channel as a grayscale map. */
function nextHiddenMapping(): ChannelMapping {
  const channelN = runner!.channelN;
  const start = 4;
  if (channelN <= start) return { r: 0, g: 0, b: 0, a: -1, bg: 0 };
  const ch = start + ((views.length - 1) % (channelN - start));
  return { r: ch, g: ch, b: ch, a: -1, bg: 0 };
}

function updateInfo(): void {
  if (!runner) return;
  const m = runner.meta;
  const parity = m.parity_max_abs_diff
    ? Object.entries(m.parity_max_abs_diff)
        .map(([k, v]) => `${k}² ${v.toExponential(1)}`)
        .join(' · ')
    : '—';
  infoEl.innerHTML = [
    ['step', String(runner.stepCount)],
    ['grid', `${runner.grid}×${runner.grid}`],
    ['channels', String(runner.channelN)],
    ['params', m.num_parameters.toLocaleString()],
    ['experiment', m.experiment],
    ['fire rate', String(runner.fireRate)],
    ['views', String(views.length)],
    ['parity', parity],
  ]
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`)
    .join('');
}

async function selectModel(model: ManifestModel): Promise<void> {
  playing = false;
  playBtn.textContent = 'Play';
  setStatus(`loading ${model.id}…`, 'warn');
  if (runner) {
    await runner.dispose();
    runner = null;
  }
  try {
    runner = await CaRunner.create(model);
  } catch (e) {
    setStatus(`failed to load: ${(e as Error).message}`, 'err');
    console.error(e);
    return;
  }
  rebuildViews();
  drawAll();
  updateInfo();
  setStatus('ready · WebGPU', 'ok');
}

async function loadModels(): Promise<void> {
  try {
    manifest = await loadManifest();
  } catch (e) {
    setStatus('could not load /manifest.json', 'err');
    console.error(e);
    return;
  }
  modelSelect.innerHTML = '';
  for (const m of manifest.models) {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = `${m.id} (${m.meta.experiment})`;
    modelSelect.appendChild(opt);
  }
  const first = manifest.models[0];
  if (first) {
    modelSelect.value = first.id;
    await selectModel(first);
  } else {
    setStatus('no models in manifest', 'warn');
  }
}

function updateCursors(): void {
  for (const v of views) {
    v.canvas.style.cursor = tool === 'none' ? 'default' : 'crosshair';
  }
}

toolsEl.addEventListener('click', (e) => {
  const btn = (e.target as HTMLElement).closest('button');
  if (!btn) return;
  tool = (btn.dataset.tool as Tool) ?? 'none';
  for (const b of toolsEl.querySelectorAll('button')) {
    b.classList.toggle('active', b === btn);
  }
  updateCursors();
});

brushRadiusInput.addEventListener('input', () => {
  brushRadius = Number(brushRadiusInput.value);
  brushRadiusVal.textContent = String(brushRadius);
});

brushFactorInput.addEventListener('input', () => {
  brushFactor = Number(brushFactorInput.value);
});

brushAlphaInput.addEventListener('change', () => {
  brushAlpha = brushAlphaInput.checked;
});

addViewBtn.addEventListener('click', () => {
  if (!runner) return;
  createView(nextHiddenMapping());
  drawAll();
  updateInfo();
});

playBtn.addEventListener('click', () => {
  if (!runner) return;
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  lastT = performance.now();
  accumulator = 0;
});

stepBtn.addEventListener('click', async () => {
  if (!runner || busy) return;
  busy = true;
  await runner.step();
  busy = false;
  drawAll();
  updateInfo();
});

resetBtn.addEventListener('click', () => {
  if (!runner) return;
  runner.reset();
  drawAll();
  updateInfo();
});

speedInput.addEventListener('input', () => {
  speedVal.textContent = speedInput.value;
});

modelSelect.addEventListener('change', () => {
  const m = manifest?.models.find((x) => x.id === modelSelect.value);
  if (m) void selectModel(m);
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

async function frame(t: number): Promise<void> {
  requestAnimationFrame(frame);
  const dt = Math.min((t - lastT) / 1000, 0.1);
  lastT = t;
  if (!playing || !runner || busy) return;

  accumulator += dt * Number(speedInput.value);
  let n = Math.floor(accumulator);
  if (n <= 0) return;
  accumulator -= n;
  n = Math.min(n, 24);

  busy = true;
  for (let i = 0; i < n; i++) await runner.step();
  busy = false;
  drawAll();
  updateInfo();
}

if (!('gpu' in navigator)) {
  setStatus('WebGPU is not available in this browser', 'err');
} else {
  void loadModels();
  requestAnimationFrame(frame);
}
