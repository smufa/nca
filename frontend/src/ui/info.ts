/** The model-information panel. Nothing here affects the run. */

import { app } from './app';
import { infoEl } from './dom';

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
}
