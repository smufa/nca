/** Types mirroring `exports/<id>/meta.json` and `exports/manifest.json`. */

export interface ModelMeta {
  id: string;
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

/**
 * Presentation grouping for the model picker, derived from the model id so no
 * manifest bookkeeping is needed. Falls back to a single "Other" group.
 */
export function modelGroup(m: ManifestModel): ModelGroup {
  const id = m.id;
  // Capacity sweep: crab_c6_h128, crab_c16_h64
  let mt = /^\w+_c(\d+)_h(\d+)$/.exec(id);
  if (mt) {
    const c = Number(mt[1]);
    const h = Number(mt[2]);
    return { group: 'Crab · capacity', label: `C=${c} h=${h}`, order: c * 10000 + h };
  }
  // Training checkpoints: crab_regen_2000, crab_regen_8000_static
  mt = /^\w+_regen_(\d+)(_static)?$/.exec(id);
  if (mt) {
    return {
      group: 'Crab · checkpoints',
      label: `@${mt[1]}${mt[2] ? ' static' : ''}`,
      order: Number(mt[1]),
    };
  }
  // Conditional multi-target runs: multi2..8 (matched pairs) / cross2..8 (a slice
  // of each batch is asked to convert one grown target into another).
  mt = /^multi(\d+)$/.exec(id);
  if (mt) {
    const n = Number(mt[1]);
    return { group: 'Multi-target', label: `${n} emoji`, order: 1000 * n };
  }
  mt = /^cross(\d+)(?:_(wide|deep))?$/.exec(id);
  if (mt) {
    const n = Number(mt[1]);
    const variant = mt[2] ? ` · ${mt[2]}` : '';
    // same N groups together, expanding (plain, wide, deep); '' sorts before the rest
    const sub = mt[2] === 'wide' ? 1 : mt[2] === 'deep' ? 2 : 0;
    // N first, then variant; scaled so all 8 wide/deep land after the plain set
    return {
      group: 'Multi-target',
      label: `${n} emoji · cross${variant}`,
      order: 1000 * n + sub,
    };
  }
  if (id === 'smile-demo') return { group: 'Demos', label: 'smile', order: 0 };
  // Regime sweep: cactus_growing, crab_regenerating, ...
  mt = /^(\w+)_(growing|persistent|regenerating)$/.exec(id);
  if (mt) return { group: 'Emoji sweep', label: `${mt[1]} · ${mt[2]}`, order: 0 };
  // One-off study runs on the crab (crab_learnperc, crab_noisy, ...). Must come
  // after the regime rule so crab_growing / crab_regenerating stay in the sweep.
  mt = /^crab_(.+)$/.exec(id);
  if (mt) {
    return { group: 'Crab · experiments', label: mt[1].replace(/_/g, ' '), order: 0 };
  }
  return { group: 'Other', label: id, order: 0 };
}
