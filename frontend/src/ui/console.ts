/**
 * `window.nca` console helpers: measure a run without the render loop in the way.
 *
 *   nca.bench(200)     -> ms/step of raw session.run (no render, no sync)
 *   nca.benchSync(50)  -> ms of one GPU->CPU download (the WASM engine's cost)
 *   nca.gpuInfo()      -> adapter, info, features
 */

import { app } from './app';

export function installConsoleHelpers(): void {
  (window as unknown as Record<string, unknown>).nca = {
    bench: async (n = 100): Promise<number | undefined> => {
      const runner = app.runner;
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
      const runner = app.runner;
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
  };
}
