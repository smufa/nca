import './style.css';
import { GROUP_ORDER, loadManifest, modelGroup } from './models';
import type { Manifest, ManifestModel } from './models';
import { CaRunner } from './runner';
import { ShaderRunner } from './shader';
import type { CaEngine } from './engine';
import { CANONICAL_KERNELS, mapChannels, rotateKernels } from './nca';
import type { ChannelMapping } from './nca';

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
}

const viewsEl = el<HTMLDivElement>('views');
const statusEl = el<HTMLSpanElement>('status');
const perfEl = el<HTMLSpanElement>('perf');
const modelSelect = el<HTMLSelectElement>('model');
const targetField = el<HTMLDivElement>('targetField');
const condSlidersEl = el<HTMLDivElement>('condSliders');
const condValEl = el<HTMLElement>('condVal');
const condResetBtn = el<HTMLButtonElement>('condReset');
/** Sliders for the conditioning vector, one per one-hot component. */
let condSliders: HTMLInputElement[] = [];
/** Last conditioning vector, so switching engine/grid does not lose a mix. */
let condState: number[] | null = null;
const engineSelect = el<HTMLSelectElement>('engine');
const gridSizeSelect = el<HTMLSelectElement>('gridSize');
const playBtn = el<HTMLButtonElement>('play');
const stepBtn = el<HTMLButtonElement>('step');
const resetBtn = el<HTMLButtonElement>('reset');
const speedInput = el<HTMLInputElement>('speed');
const speedVal = el<HTMLElement>('speedVal');
const infoEl = el<HTMLDListElement>('info');
const addViewBtn = el<HTMLButtonElement>('addView');
const angleInput = el<HTMLInputElement>('angle');
const angleVal = el<HTMLElement>('angleVal');
const linkedInput = el<HTMLInputElement>('linked');
const autoRotateInput = el<HTMLInputElement>('autoRotate');
const kernelGridsEl = el<HTMLDivElement>('kernelGrids');
const resetKernelsBtn = el<HTMLButtonElement>('resetKernels');
const toolsEl = el<HTMLDivElement>('tools');
const brushRadiusInput = el<HTMLInputElement>('brushRadius');
const brushRadiusVal = el<HTMLElement>('brushRadiusVal');
const brushFactorInput = el<HTMLInputElement>('brushFactor');
const brushAlphaInput = el<HTMLInputElement>('brushAlpha');
const viewSizeInput = el<HTMLInputElement>('viewSize');
const viewSizeVal = el<HTMLElement>('viewSizeVal');

/** A render panel: its own canvas + channel mapping selectors. */
interface View {
  root: HTMLDivElement;
  canvas: HTMLCanvasElement;
  ctx2d: CanvasRenderingContext2D | null;
  image: ImageData | null;
  mapping: ChannelMapping;
  selects: Record<'r' | 'g' | 'b' | 'a' | 'bg', HTMLSelectElement>;
}

/** An in-progress brush stroke. Applied every frame while the button is held. */
interface Stroke {
  view: View;
  pointerId: number;
  lastX: number;
  lastY: number;
  curX: number;
  curY: number;
}
let stroke: Stroke | null = null;

let manifest: Manifest | null = null;
let runner: CaEngine | null = null;
let views: View[] = [];
let playing = false;
let busy = false;
let idleWaiters: Array<() => void> = [];
let selectToken = 0;
let accumulator = 0;
let lastT = performance.now();

type Ep = 'wasm' | 'shader';

/** Engine from ?ep= (persisted) or localStorage; default WASM (works everywhere). */
function initialEp(): Ep {
  const url = new URLSearchParams(location.search).get('ep');
  if (url === 'wasm' || url === 'shader') {
    localStorage.setItem('nca.ep', url);
    return url;
  }
  const stored = localStorage.getItem('nca.ep');
  return stored === 'shader' ? 'shader' : 'wasm';
}

let ep: Ep = initialEp();

function setBusy(value: boolean): void {
  busy = value;
  if (!value) {
    const waiters = idleWaiters;
    idleWaiters = [];
    for (const w of waiters) w();
  }
}

/** Resolves once no step is in flight (used before releasing a session). */
function whenIdle(): Promise<void> {
  if (!busy) return Promise.resolve();
  return new Promise((resolve) => idleWaiters.push(resolve));
}

let perfFrames = 0;
let perfSteps = 0;
let perfStepMs = 0;
let perfTime = performance.now();
let prevPerf = { maskMs: 0, runMs: 0, syncMs: 0, steps: 0 };

function maybeUpdatePerf(): void {
  const now = performance.now();
  const dt = now - perfTime;
  if (dt < 500) return;
  const parts = [
    `${((perfFrames * 1000) / dt).toFixed(0)} fps`,
    `${((perfSteps * 1000) / dt).toFixed(1)} steps/s`,
  ];
  if (perfSteps > 0) parts.push(`${(perfStepMs / perfSteps).toFixed(1)} ms/step`);
  if (runner) {
    const p = runner.perf;
    const n = p.steps - prevPerf.steps;
    if (n > 0) {
      parts.push(
        `run ${((p.runMs - prevPerf.runMs) / n).toFixed(1)}` +
          ` · sync ${((p.syncMs - prevPerf.syncMs) / n).toFixed(1)}` +
          ` · mask ${((p.maskMs - prevPerf.maskMs) / n).toFixed(1)} ms`,
      );
    }
    prevPerf = { ...p };
  }
  perfEl.textContent = parts.join(' · ');
  perfFrames = 0;
  perfSteps = 0;
  perfStepMs = 0;
  perfTime = now;
}

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
  if (runner.renderTo) {
    runner.renderTo(view.canvas, view.mapping);
    return;
  }
  if (!view.ctx2d || !view.image) return;
  mapChannels(runner.x, runner.grid, runner.grid, view.mapping, view.image.data);
  view.ctx2d.putImageData(view.image, 0, 0);
}

function drawAll(): void {
  for (const view of views) drawView(view);
}

/** Resize every view's canvas/image after a grid change (mappings preserved). */
function resizeViews(grid: number): void {
  for (const view of views) {
    view.canvas.width = grid;
    view.canvas.height = grid;
    if (view.ctx2d) view.image = view.ctx2d.createImageData(grid, grid);
  }
}

/** Channels a brush stroke affects in a view: the ones it displays, plus alpha. */
function brushTargets(view: View): number[] {
  if (!runner) return [];
  const set = new Set<number>([view.mapping.r, view.mapping.g, view.mapping.b]);
  if (brushAlpha) set.add(3);
  return [...set].filter((c) => c >= 0 && c < runner!.channelN);
}

/** Apply the active tool to a disc of `brushRadius` cells centred on (cx, cy). */
function paintDot(view: View, cx: number, cy: number): void {
  if (!runner || tool === 'none') return;
  const targets = brushTargets(view);
  if (targets.length === 0) return;
  const op = tool;
  const factor = brushFactor;

  // GPU engines apply the stroke in a compute pass - no state upload.
  if (runner.brush) {
    runner.brush(op, cx, cy, brushRadius, factor, targets);
    return;
  }

  const grid = runner.grid;
  const plane = grid * grid;
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
      for (const c of targets) indices.push(c * plane + i);
    }
  }
  if (indices.length === 0) return;

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
): void {
  const n = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    paintDot(
      view,
      Math.round(x0 + (x1 - x0) * t),
      Math.round(y0 + (y1 - y0) * t),
    );
  }
}

/** Re-apply the brush at the held position every frame. */
function tickStroke(): void {
  if (!stroke || !runner) return;
  paintDot(stroke.view, stroke.curX, stroke.curY);
  drawAll();
}

// --- Kernel lab ---------------------------------------------------------

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

function setLinked(linked: boolean): void {
  linkedInput.checked = linked;
  for (const arr of kernelInputs) for (const inp of arr) inp.disabled = linked;
}

function applyKernels(kernels: number[][]): void {
  setKernelInputs(kernels);
  if (runner) runner.setKernels(kernels[0], kernels[1], kernels[2]);
}

function onKernelEdit(): void {
  if (linkedInput.checked) setLinked(false);
  const kernels = readKernelInputs();
  for (let k = 0; k < 3; k++) {
    for (let i = 0; i < 9; i++) swatch(kernelInputs[k][i], kernels[k][i]);
  }
  if (runner) runner.setKernels(kernels[0], kernels[1], kernels[2]);
}

/** Render an emoji target spec as its glyph (falling back to the raw spec). */
function prettySpec(spec: string): string {
  if (!spec.startsWith('emoji:')) return spec;
  const v = spec.slice('emoji:'.length);
  if (!/^[0-9a-f-]+$/i.test(v)) return v;
  try {
    return v
      .split('-')
      .map((c) => String.fromCodePoint(parseInt(c, 16)))
      .join('');
  } catch {
    return spec;
  }
}

/**
 * Build the conditioning-vector editor for a conditional model, or hide it.
 *
 * One slider per component, so the vector does not have to be one-hot: setting
 * 0.5/0.5 asks the CA for a state between two targets, which is out of
 * distribution and is the point of being able to move them freely.
 */
function setupCond(model: ManifestModel): void {
  const n = model.meta.cond_n ?? 0;
  condSlidersEl.innerHTML = '';
  condSliders = [];
  targetField.style.display = n ? '' : 'none';
  if (!n) return;
  const specs = model.meta.targets ?? [];
  // Reuse the previous vector when the width matches (engine/grid switches), so a
  // hand-tuned mix survives; otherwise start from a pure one-hot on component 0.
  const keep = condState && condState.length === n ? condState : null;
  for (let i = 0; i < n; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const label = document.createElement('label');
    label.textContent = specs[i] ? prettySpec(specs[i]) : String(i + 1);
    label.title = specs[i] ?? `target ${i + 1}`;

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '-1';
    slider.max = '1';
    slider.step = '0.05';
    slider.value = keep ? String(keep[i]) : i === 0 ? '1' : '0';
    slider.dataset.index = String(i);

    const out = document.createElement('output');
    out.textContent = slider.value;

    slider.addEventListener('input', () => {
      out.textContent = Number(slider.value).toFixed(2);
      applyCondVector();
    });

    row.append(label, slider, out);
    condSlidersEl.appendChild(row);
    condSliders.push(slider);
  }
  updateCondLabel();
}

function readCondVector(): number[] {
  return condSliders.map((s) => Number(s.value) || 0);
}

function updateCondLabel(): void {
  const v = readCondVector();
  condState = v;
  condValEl.textContent = `[${v.map((x) => x.toFixed(2)).join(', ')}]`;
}

/** Push the slider values to the engine without touching the state. */
function applyCondVector(): void {
  updateCondLabel();
  const v = readCondVector();
  void whenIdle().then(() => {
    if (runner?.setCondVector) runner.setCondVector(v);
    else runner?.setCond?.(v.indexOf(Math.max(...v)));
  });
}

/** Reset the conditioning vector to a pure one-hot on component 0. */
function resetCondVector(): void {
  condSliders.forEach((s, i) => {
    s.value = i === 0 ? '1' : '0';
    const out = s.nextElementSibling as HTMLOutputElement | null;
    if (out) out.textContent = s.value;
  });
  applyCondVector();
  drawAll();
}

function resetKernelsUI(): void {
  autoAngle = 0;
  angleInput.value = '0';
  angleVal.textContent = '0°';
  setLinked(true);
  // Start from the kernels the model actually uses (learned, for --learn-perception).
  const kernels = runner ? runner.getKernels() : CANONICAL_KERNELS.map((k) => k.slice());
  applyKernels(kernels);
}

function buildKernelEditors(): void {
  kernelGridsEl.innerHTML = '';
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
  const onGpu = !!runner.renderTo;
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
    if (views.length <= 1) return; // always keep one view
    views = views.filter((v) => v !== view);
    root.remove();
  });

  canvas.style.touchAction = 'none';
  canvas.style.cursor = tool === 'none' ? 'default' : 'crosshair';

  let painting = false;

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
    const [cx, cy] = toCell(e);
    stroke = { view, pointerId: e.pointerId, lastX: cx, lastY: cy, curX: cx, curY: cy };
    paintDot(view, cx, cy);
    drawAll();
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!painting || !stroke || stroke.view !== view || !runner) return;
    const [cx, cy] = toCell(e);
    paintLine(view, stroke.lastX, stroke.lastY, cx, cy);
    stroke.lastX = cx;
    stroke.lastY = cy;
    stroke.curX = cx;
    stroke.curY = cy;
    drawAll();
  });

  const endStroke = (e: PointerEvent): void => {
    if (!painting) return;
    painting = false;
    if (stroke && stroke.view === view && stroke.pointerId === e.pointerId) stroke = null;
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
    ['engine', ep],
    ['fire rate', String(runner.fireRate)],
    ['io', runner.mode],
    ['gpu', runner.adapter],
    ['views', String(views.length)],
    ['parity', parity],
  ]
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`)
    .join('');
}

async function selectModel(model: ManifestModel): Promise<void> {
  const token = ++selectToken;
  playing = false;
  playBtn.textContent = 'Play';
  rememberModel(model.id);
  setStatus(`loading ${model.id}…`, 'warn');

  // Let any in-flight step finish before releasing the old session, otherwise
  // ORT unmaps a GPU buffer mid-download ("Buffer was unmapped before mapping
  // was resolved").
  await whenIdle();
  if (token !== selectToken) return; // superseded by a newer selection

  if (runner) {
    await runner.dispose();
    runner = null;
  }
  prevPerf = { maskMs: 0, runMs: 0, syncMs: 0, steps: 0 };
  let shaderFallback = false;
  try {
    const shaderOk = ep !== 'shader' || ShaderRunner.supports(model);
    shaderFallback = !shaderOk;
    runner = !shaderOk
      ? await CaRunner.create(model)
      : ep === 'shader'
        ? await ShaderRunner.create(model)
        : await CaRunner.create(model);
  } catch (e) {
    setStatus(`failed to load: ${(e as Error).message}`, 'err');
    console.error(e);
    return;
  }
  setupCond(model);
  // A fresh engine starts from its own default, so push the current mix into it.
  if (condSliders.length && runner.setCondVector) {
    runner.setCondVector(readCondVector());
  }
  rebuildViews();
  resetKernelsUI();
  gridSizeSelect.value = String(runner.grid);
  drawAll();
  updateInfo();
  const label =
    runner.mode === 'gpu' ? 'WebGPU' : runner.mode === 'shader' ? 'WebGPU shader' : 'WASM';
  if (shaderFallback) {
    setStatus(`ready · ${label} (shader unsupported for this model size)`, 'warn');
  } else {
    setStatus(`ready · ${label}`, 'ok');
  }
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
  const wanted = new URLSearchParams(location.search).get('model');
  const fromUrl = wanted ? manifest.models.find((m) => m.id === wanted) : undefined;
  const entries = manifest.models.map((m, i) => ({ m, g: modelGroup(m), i }));
  const names = [...GROUP_ORDER];
  for (const e of entries) if (!names.includes(e.g.group)) names.push(e.g.group);
  for (const name of names) {
    const items = entries
      .filter((e) => e.g.group === name)
      .sort((a, b) => a.g.order - b.g.order || a.i - b.i);
    if (items.length === 0) continue;
    const og = document.createElement('optgroup');
    og.label = name;
    for (const { m, g } of items) {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = g.label;
      opt.title = `${m.id} · ${m.meta.experiment} · C=${m.meta.channel_n} h=${m.meta.hidden_n}`;
      og.appendChild(opt);
    }
    modelSelect.appendChild(og);
  }
  const first = fromUrl ?? manifest.models[0];
  if (wanted && !fromUrl) {
    // A stale link (the id grammar changed once, see docs/HOUSEKEEPING.md). Say so
    // instead of silently showing a different model.
    setStatus(`unknown model in ?model=${wanted}`, 'err');
  }
  if (first) {
    modelSelect.value = first.id;
    await selectModel(first);
  } else {
    setStatus('no models in manifest', 'warn');
  }
}

/** `?model=<id>` deep link; keeps `?ep=` and any other param intact. */
function rememberModel(modelId: string): void {
  const params = new URLSearchParams(location.search);
  params.set('model', modelId);
  history.replaceState(null, '', `${location.pathname}?${params}`);
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

function applyViewSize(px: number): void {
  document.documentElement.style.setProperty('--view-size', `${px}px`);
  viewSizeVal.textContent = String(px);
}

viewSizeInput.addEventListener('input', () => {
  applyViewSize(Number(viewSizeInput.value));
});

angleInput.addEventListener('input', () => {
  if (!linkedInput.checked) return;
  const deg = Number(angleInput.value);
  angleVal.textContent = `${deg}°`;
  applyKernels(rotateKernels((deg * Math.PI) / 180));
});

linkedInput.addEventListener('change', () => {
  setLinked(linkedInput.checked);
  if (linkedInput.checked) {
    const deg = Number(angleInput.value);
    applyKernels(rotateKernels((deg * Math.PI) / 180));
  }
});

resetKernelsBtn.addEventListener('click', resetKernelsUI);

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
  setBusy(true);
  try {
    await runner.step();
    if (!runner.renderTo) await runner.sync();
  } catch (e) {
    console.error(e);
    setStatus(`step failed: ${(e as Error).message}`, 'err');
  } finally {
    setBusy(false);
  }
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

condResetBtn.addEventListener('click', () => resetCondVector());

engineSelect.value = ep;
engineSelect.addEventListener('change', () => {
  ep = engineSelect.value as Ep;
  localStorage.setItem('nca.ep', ep);
  const m = manifest?.models.find((x) => x.id === modelSelect.value);
  if (m) void selectModel(m);
});

gridSizeSelect.addEventListener('change', () => {
  if (!runner) return;
  runner.setGrid(Number(gridSizeSelect.value));
  resizeViews(runner.grid);
  drawAll();
  updateInfo();
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
  perfFrames++;
  maybeUpdatePerf();
  tickStroke();
  const dt = Math.min((t - lastT) / 1000, 0.1);
  lastT = t;

  if (autoRotateInput.checked) {
    autoAngle = (autoAngle + dt * 45) % 360;
    if (!linkedInput.checked) setLinked(true);
    angleInput.value = String(Math.round(autoAngle));
    angleVal.textContent = `${Math.round(autoAngle)}°`;
    applyKernels(rotateKernels((autoAngle * Math.PI) / 180));
  }

  if (!playing || !runner || busy) return;

  accumulator += dt * Number(speedInput.value);
  let n = Math.floor(accumulator);
  if (n <= 0) return;
  accumulator -= n;
  n = Math.min(n, 24);

  setBusy(true);
  const t0 = performance.now();
  let done = 0;
  try {
    for (let i = 0; i < n; i++) {
      await runner.step();
      done++;
    }
    if (!runner.renderTo) await runner.sync();
  } catch (e) {
    console.error(e);
    setStatus(`step failed: ${(e as Error).message}`, 'err');
    playing = false;
    playBtn.textContent = 'Play';
  } finally {
    setBusy(false);
  }
  perfSteps += done;
  perfStepMs += performance.now() - t0;
  drawAll();
  updateInfo();
}

if (!('gpu' in navigator)) {
  setStatus('WebGPU is not available in this browser', 'err');
} else {
  buildKernelEditors();
  resetKernelsUI();
  applyViewSize(Number(viewSizeInput.value));
  void loadModels();
  requestAnimationFrame(frame);
}

// Console helpers: `nca.bench(200)` measures raw session.run cost (no render),
// `nca.benchSync(50)` measures the GPU->CPU download on its own.
(window as unknown as Record<string, unknown>).nca = {
  bench: async (n = 100): Promise<number | undefined> => {
    if (!runner) {
      console.warn('[nca] no model loaded');
      return undefined;
    }
    const t0 = performance.now();
    for (let i = 0; i < n; i++) await runner.step();
    const ms = (performance.now() - t0) / n;
    console.info(`[nca] run: ${ms.toFixed(2)} ms/step · ${(1000 / ms).toFixed(0)} steps/s`);
    return ms;
  },
  benchSync: async (n = 50): Promise<number | undefined> => {
    if (!runner) {
      console.warn('[nca] no model loaded');
      return undefined;
    }
    const t0 = performance.now();
    for (let i = 0; i < n; i++) await runner.sync();
    const ms = (performance.now() - t0) / n;
    console.info(`[nca] sync: ${ms.toFixed(2)} ms`);
    return ms;
  },
  gpuInfo: async (): Promise<unknown> => {
    const adapter = await navigator.gpu.requestAdapter();
    const a = adapter as
      | { info?: unknown; adapterInfo?: unknown; features?: GPUFeatureName[] }
      | null;
    const info = a?.info ?? a?.adapterInfo ?? null;
    const features = a?.features ? [...a.features] : [];
    console.info('[nca] adapter:', adapter, '\ninfo:', info, '\nfeatures:', features);
    return { adapter, info, features };
  },
  state: () => runner,
};
