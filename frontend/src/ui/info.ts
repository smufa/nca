/** The model-information panel. Nothing here affects the run. */

import { app } from './app';
import { infoEl, lossBlock, lossCanvas, lossCaption } from './dom';
import { drawLossCurve, fmtLoss } from './sparkline';
import type { ModelMeta } from '../models';

/** The model the sparkline was last drawn for (updateInfo runs every frame). */
let drawnFor: ModelMeta | undefined;

function updateLoss(m: ModelMeta): void {
  if (m === drawnFor) return;
  drawnFor = m;
  const curve = m.loss;
  lossBlock.hidden = !curve;
  if (!curve) return;
  drawLossCurve(lossCanvas, curve);
  lossCaption.textContent = `loss ${fmtLoss(curve.final)} → min ${fmtLoss(curve.min)} @ ${Math.round(curve.min_at * 100)}%`;
  lossBlock.title =
    `${curve.logged} logged values, bucketed to ${curve.points.length} samples ` +
    `(log scale); the dot marks the minimum`;
}

export function updateInfo(): void {
  const runner = app.runner;
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
    ['engine', app.ep],
    ['fire rate', String(runner.fireRate)],
    ['io', runner.mode],
    ['gpu', runner.adapter],
    ['views', String(app.views.length)],
    ['parity', parity],
  ]
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`)
    .join('');
  updateLoss(m);
}
