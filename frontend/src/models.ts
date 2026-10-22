/** Types mirroring `exports/<id>/meta.json` and `exports/manifest.json`. */

export interface ModelMeta {
  id: string;
  /** The id, duplicated in meta; the id *is* the name (see docs/HOUSEKEEPING.md). */
  name?: string;
  /** Which study produced the model: the picker's group key. */
  sweep?: string;
  /** Recovered training facts the id encodes (iteration, mismatch_p, ...). */
  train?: Record<string, string | number>;
  /** Presentation override; absent means "derive it from the id". */
  picker?: { group?: string; label?: string; order?: number };
  created: string;
  channel_n: number;
  hidden_n: number;
  fire_rate: number;
  learn_perception?: boolean;
  perception_kernel?: number[];
  /** >0 for a conditional model: the one-hot selector width. */
  cond_n?: number;
  /** Hidden (ReLU) blocks in the update rule; the WGSL step implements 1. */
  hidden_layers?: number;
  /** Target spec per selector index (conditional models). */
  targets?: string[];
  num_parameters: number;
  grid: number;
  dynamic_shape: boolean;
  wrap: boolean;
  opset: number;
  experiment: string;
  target: string;
  source_checkpoint: string;
  torch_version: string;
  onnxruntime_version: string;
  parity_max_abs_diff: Record<string, number> | null;
  weights: {
    file: string;
    channel_n: number;
    hidden_n: number;
    hidden_layers?: number;
    perception_n?: number;
    perception_padded_n?: number;
    offsets: { w1: number; b1: number; w2: number; b2: number };
    count: number;
  };
  size_bytes: number;
  git_rev: string | null;
}

export interface ManifestModel {
  id: string;
  path: string;
  meta: ModelMeta;
}

export interface Manifest {
  version: number;
  generated: string;
  models: ManifestModel[];
}

export async function loadManifest(url = '/models/manifest.json'): Promise<Manifest> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`manifest request failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as Manifest;
}

/** A model's place in the picker: which group, and the short label to show. */
export interface ModelGroup {
  group: string;
  label: string;
  /** Sort key inside the group (models with equal order keep manifest order). */
  order: number;
}

/** Groups are shown in this order; any other group name is appended after. */
export const GROUP_ORDER = [
  'Emoji sweep',
  'Crab · capacity',
  'Crab · checkpoints',
  'Crab · experiments',
  'Multi-target',
  'Demos',
];

/** `meta.sweep` -> picker group. A model without a known sweep lands in "Other". */
export const SWEEP_GROUPS: Record<string, string> = {
  'emoji-sweep': 'Emoji sweep',
  capacity: 'Crab · capacity',
  checkpoints: 'Crab · checkpoints',
  experiments: 'Crab · experiments',
  multitarget: 'Multi-target',
  crosstarget: 'Multi-target',
  demo: 'Demos',
};

/** `c=16.h=8` -> 160008, so the capacity list reads c-major then h. */
function orderOf(id: string): number {
  const num = (key: string): number => {
    const m = new RegExp(`[.]${key}=(\\d+)`).exec(id);
    return m ? Number(m[1]) : 0;
  };
  // Conditional sweeps: by N, then the plain run before its wide/deep variants.
  const cond = /^(?:multi|cross)(\d+)[.]/.exec(id);
  if (cond) {
    const sub = id.includes('.h=') ? 1 : id.includes('.l=') ? 2 : 0;
    return 1000 * Number(cond[1]) + sub;
  }
  const c = num('c');
  const h = num('h');
  if (c || h) return c * 10000 + h;
  return num('ckpt');
}

/** `crab.regenerating.c=16.h=8` -> "c=16 h=8", `...ckpt=2000` -> "@2000". */
function labelOf(id: string, sweep: string | undefined): string {
  const [subject, task, ...rest] = id.split('.');
  if (sweep === 'emoji-sweep') return `${subject} · ${task}`;
  const cond = /^(?:multi|cross)(\d+)$/.exec(subject);
  if (cond) {
    const kind = subject.startsWith('cross') ? 'cross' : 'matched';
    // `mismatch=0.3` is the same for the whole cross family: it is the sweep.
    const variant = rest.filter((s) => !s.startsWith('mismatch=')).join(' ');
    return `${cond[1]} emoji · ${kind}${variant ? ` · ${variant}` : ''}`;
  }
  const knobs = rest.join(' ').replace(/\bckpt=(\d+)/, '@$1');
  if (knobs) return knobs;
  return sweep === 'demo' ? subject : task;
}

/**
 * Presentation grouping for the model picker.
 *
 * The group comes from `meta.sweep` (written by the sweep that produced the model),
 * the label and the sort key from the dotted id; `meta.picker` overrides any of
 * them. Nothing here parses legacy `crab_c16_h64`-style names -- that grammar is
 * gone (see `scripts/rename_exports.py`, which holds the one-time mapping).
 */
export function modelGroup(m: ManifestModel): ModelGroup {
  const picker = m.meta.picker ?? {};
  const sweep = m.meta.sweep;
  return {
    group: picker.group ?? (sweep ? (SWEEP_GROUPS[sweep] ?? `Other · ${sweep}`) : 'Other'),
    label: picker.label ?? labelOf(m.id, sweep),
    order: picker.order ?? orderOf(m.id),
  };
}
