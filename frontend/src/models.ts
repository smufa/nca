/** Types mirroring `exports/<id>/meta.json` and `exports/manifest.json`. */

export interface ModelMeta {
  id: string;
  created: string;
  channel_n: number;
  hidden_n: number;
  fire_rate: number;
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
