/**
 * Conditioning vector editor for multi-target models (`--targets`).
 *
 * One slider per component, so the vector does not have to be one-hot: setting
 * 0.5/0.5 asks the CA for a state between two targets, which is out of
 * distribution and is the point of being able to move them freely.
 */

import type { ManifestModel } from '../models';
import { app, whenIdle } from './app';
import { condResetBtn, condSlidersEl, condValEl, targetField } from './dom';
import { drawAll } from './views';

let condSliders: HTMLInputElement[] = [];
let condState: number[] | null = null;

/** Render an emoji target spec as its glyph (falling back to the raw spec). */
export function prettySpec(spec: string): string {
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

export function setupCond(model: ManifestModel): void {
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

export function readCondVector(): number[] {
  return condSliders.map((s) => Number(s.value) || 0);
}

export function hasCond(): boolean {
  return condSliders.length > 0;
}

function updateCondLabel(): void {
  const v = readCondVector();
  condState = v;
  condValEl.textContent = `[${v.map((x) => x.toFixed(2)).join(', ')}]`;
}

/** Push the slider values to the engine without touching the state. */
export function applyCondVector(): void {
  updateCondLabel();
  const v = readCondVector();
  void whenIdle().then(() => {
    const runner = app.runner;
    if (runner?.capabilities.cond) runner.setCondVector(v);
  });
}

/** Reset the conditioning vector to a pure one-hot on component 0. */
export function resetCondVector(): void {
  condSliders.forEach((s, i) => {
    s.value = i === 0 ? '1' : '0';
    const out = s.nextElementSibling as HTMLOutputElement | null;
    if (out) out.textContent = s.value;
  });
  applyCondVector();
  drawAll();
}

export function wireCondControls(): void {
  condResetBtn.addEventListener('click', () => resetCondVector());
}
