/**
 * The mutable UI state, in one place.
 *
 * This is a hub, not a state machine: the fields are the ones that more than one
 * module reads (`runner` and `views` are touched by almost all of them). Anything
 * only one module needs stays local to that module.
 *
 * The `busy`/`whenIdle` pair is the important invariant: an engine must never be
 * released or reconfigured while a step is in flight (ORT unmaps a GPU buffer
 * mid-download: "Buffer was unmapped before mapping was resolved").
 */

import type { CaEngine } from '../engine';
import type { Manifest } from '../models';
import { perfEl, statusEl } from './dom';
import type { View } from './views';

export type Ep = 'wasm' | 'shader';
export type Tool = 'none' | 'one' | 'zero' | 'mul';

export const app = {
  manifest: null as Manifest | null,
  runner: null as CaEngine | null,
  views: [] as View[],
  playing: false,
  busy: false,
  tool: 'none' as Tool,
  brushRadius: 3,
  brushFactor: 0.5,
  brushAlpha: true,
  ep: 'wasm' as Ep,
  /** Bumped per model load so a superseded load can bail out. */
  selectToken: 0,
  accumulator: 0,
  lastT: performance.now(),
};

let idleWaiters: Array<() => void> = [];

export function setBusy(value: boolean): void {
  app.busy = value;
  if (!value) {
    const waiters = idleWaiters;
    idleWaiters = [];
    for (const w of waiters) w();
  }
}

/** Resolves once no step is in flight (used before releasing a session). */
export function whenIdle(): Promise<void> {
  if (!app.busy) return Promise.resolve();
  return new Promise((resolve) => idleWaiters.push(resolve));
}

export type StatusKind = 'ok' | 'warn' | 'err';

export function setStatus(text: string, kind: StatusKind = 'ok'): void {
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`;
}

interface PerfCounters {
  maskMs: number;
  runMs: number;
  syncMs: number;
  steps: number;
}

const perf = {
  frames: 0,
  steps: 0,
  stepMs: 0,
  time: performance.now(),
  prev: { maskMs: 0, runMs: 0, syncMs: 0, steps: 0 } as PerfCounters,
};

export function countFrame(): void {
  perf.frames++;
}

export function addSteps(n: number, ms: number): void {
  perf.steps += n;
  perf.stepMs += ms;
}

/** Forget the engine's cumulative counters (they belong to the old session). */
export function resetPerfBaseline(): void {
  perf.prev = { maskMs: 0, runMs: 0, syncMs: 0, steps: 0 };
}

/** Rewrite the perf line at most twice a second. */
export function maybeUpdatePerf(): void {
  const now = performance.now();
  const dt = now - perf.time;
  if (dt < 500) return;
  const parts = [
    `${((perf.frames * 1000) / dt).toFixed(0)} fps`,
    `${((perf.steps * 1000) / dt).toFixed(1)} steps/s`,
  ];
  if (perf.steps > 0) parts.push(`${(perf.stepMs / perf.steps).toFixed(1)} ms/step`);
  if (app.runner) {
    const p = app.runner.perf;
    const n = p.steps - perf.prev.steps;
    if (n > 0) {
      parts.push(
        `run ${((p.runMs - perf.prev.runMs) / n).toFixed(1)}` +
          ` · sync ${((p.syncMs - perf.prev.syncMs) / n).toFixed(1)}` +
          ` · mask ${((p.maskMs - perf.prev.maskMs) / n).toFixed(1)} ms`,
      );
    }
    perf.prev = { ...p };
  }
  perfEl.textContent = parts.join(' · ');
  perf.frames = 0;
  perf.steps = 0;
  perf.stepMs = 0;
  perf.time = now;
}

/** Engine from `?ep=` (persisted) or localStorage; default WASM (works everywhere). */
export function initialEp(): Ep {
  const url = new URLSearchParams(location.search).get('ep');
  if (url === 'wasm' || url === 'shader') {
    localStorage.setItem('nca.ep', url);
    return url;
  }
  const stored = localStorage.getItem('nca.ep');
  return stored === 'shader' ? 'shader' : 'wasm';
}
