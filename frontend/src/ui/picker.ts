/**
 * Model picker: manifest load, grouped options, support marking, session swap.
 *
 * Loading a model is the only place that can release an engine, which is why it
 * waits for `whenIdle()` first: ORT unmaps a GPU buffer mid-download otherwise
 * ("Buffer was unmapped before mapping was resolved"). It is also guarded by a
 * token so a fast double-click cannot interleave two loads.
 */

import { checkSpec } from '../engine';
import {
  GROUP_ORDER,
  loadManifest,
  modelGroup,
  type Manifest,
  type ManifestModel,
} from '../models';
import { CaRunner } from '../runner';
import { ShaderRunner } from '../shader';
import { app, resetPerfBaseline, setStatus, whenIdle } from './app';
import { hasCond, readCondVector, setupCond } from './cond';
import { gridSizeSelect, modelSelect, playBtn } from './dom';
import { updateInfo } from './info';
import { fmtLoss } from './sparkline';
import { resetKernelsUI } from './kernels';
import { drawAll, rebuildViews } from './views';

async function selectModel(model: ManifestModel): Promise<void> {
  const token = ++app.selectToken;
  app.playing = false;
  playBtn.textContent = 'Play';
  rememberModel(model.id);
  setStatus(`loading ${model.id}…`, 'warn');

  // Let any in-flight step finish before releasing the old session, otherwise
  // ORT unmaps a GPU buffer mid-download ("Buffer was unmapped before mapping
  // was resolved").
  await whenIdle();
  if (token !== app.selectToken) return; // superseded by a newer selection

  if (app.runner) {
    await app.runner.dispose();
    app.runner = null;
  }
  resetPerfBaseline();
  // Advertised capabilities, not `?.` probing: the reason a shader model falls
  // back to WASM is data the UI can show ("hidden=64 > 32").
  const support = app.ep === 'shader' ? ShaderRunner.supports(model) : ({ ok: true } as const);
  const fallbackReason = support.ok ? null : support.reason;
  try {
    app.runner =
      app.ep === 'shader' && support.ok
        ? await ShaderRunner.create(model)
        : await CaRunner.create(model);
  } catch (e) {
    setStatus(`failed to load: ${(e as Error).message}`, 'err');
    console.error(e);
    return;
  }
  setupCond(model);
  // A fresh engine starts from its own default, so push the current mix into it.
  if (hasCond() && app.runner.capabilities.cond) {
    app.runner.setCondVector(readCondVector());
  }
  rebuildViews();
  resetKernelsUI();
  gridSizeSelect.value = String(app.runner.grid);
  drawAll();
  updateInfo();
  const label = app.runner.capabilities.label;
  const skip = app.runner.capabilities.workSkip ? ' · work-skip' : '';
  if (fallbackReason) {
    setStatus(`ready · ${label}${skip} (shader cannot run this model: ${fallbackReason})`, 'warn');
  } else {
    setStatus(`ready · ${label}${skip}`, 'ok');
  }
  markPickerSupport();
}

/**
 * Mark the models the selected engine cannot run, with the reason as the option's
 * tooltip. Re-run whenever the engine switch changes, since "runs on shader" is a
 * property of the engine/model pair.
 */
export function markPickerSupport(): void {
  const manifest = app.manifest;
  if (!manifest) return;
  const caps = app.ep === 'shader' ? ShaderRunner.CAPABILITIES : CaRunner.CAPABILITIES;
  for (const opt of Array.from(modelSelect.options)) {
    const model = manifest.models.find((m) => m.id === opt.value);
    if (!model) continue;
    const support = checkSpec(caps, model.meta);
    opt.disabled = !support.ok;
    opt.title = support.ok
      ? `${model.id} · C=${model.meta.channel_n} h=${model.meta.hidden_n}`
      : `${model.id} · ${caps.label} cannot run it: ${support.reason}`;
  }
}

export async function loadModels(): Promise<void> {
  let manifest: Manifest;
  try {
    manifest = await loadManifest();
  } catch (e) {
    setStatus('could not load /manifest.json', 'err');
    console.error(e);
    return;
  }
  app.manifest = manifest;
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
      const curve = m.meta.loss;
      const conv = curve ? ` · min ${fmtLoss(curve.min)} @ ${Math.round(curve.min_at * 100)}%` : '';
      opt.title = `${m.id} · ${m.meta.experiment} · C=${m.meta.channel_n} h=${m.meta.hidden_n}${conv}`;
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

/** Model for the current select value, if it is in the manifest. */
export function selectedModel(): ManifestModel | undefined {
  return app.manifest?.models.find((x) => x.id === modelSelect.value);
}

export { selectModel };
