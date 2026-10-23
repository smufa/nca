/**
 * Sparklines for the model panel.
 *
 * A run's loss falls by orders of magnitude, so a linear y axis shows a cliff
 * followed by a flat line; log10 is what makes the tail (where the interesting
 * question lives: did it keep improving, or plateau, or drift?) readable.
 */

import type { LossCurve } from '../models';

const PAD = 2;

/** `3.5e-4` style, but short: the caption has a narrow column. */
export function fmtLoss(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (v === 0) return '0';
  if (v >= 0.01) return v.toFixed(3);
  return v.toExponential(1);
}

export function drawLossCurve(canvas: HTMLCanvasElement, curve: LossCurve): void {
  const points = curve.points;
  if (!points.length) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 200;
  const h = canvas.clientHeight || 44;
  const width = Math.max(1, Math.round(w * dpr));
  const height = Math.max(1, Math.round(h * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, width, height);

  const lo = Math.min(...points);
  const hi = Math.max(...points);
  // A flat curve (or one point) would divide by zero: give it a decade of headroom.
  const span = hi - lo || Math.max(1e-6, hi * 0.1);
  const y = (v: number): number => {
    const t = (Math.log10(Math.max(v, 1e-12)) - Math.log10(Math.max(lo, 1e-12))) / Math.log10((lo + span) / Math.max(lo, 1e-12));
    return height - PAD - Math.min(1, Math.max(0, t)) * (height - 2 * PAD);
  };
  const x = (i: number): number =>
    points.length === 1 ? width / 2 : PAD + (i / (points.length - 1)) * (width - 2 * PAD);

  // The curve.
  ctx.beginPath();
  points.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.strokeStyle = getComputedStyle(canvas).color || '#8ab4f8';
  ctx.lineWidth = 1.5 * dpr;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // The minimum: a dot, so "bottomed out at 40% then drifted" is visible at a glance.
  const at = Math.min(points.length - 1, Math.max(0, Math.round(curve.min_at * (points.length - 1))));
  ctx.beginPath();
  ctx.arc(x(at), y(points[at]), 2.2 * dpr, 0, Math.PI * 2);
  ctx.fillStyle = ctx.strokeStyle;
  ctx.fill();
}
