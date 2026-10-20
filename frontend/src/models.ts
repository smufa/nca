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
  opset: number;
  experiment: string;
  target: string;
  source_checkpoint: string;
  torch_version: string;
  onnxruntime_version: string;
  parity_max_abs_diff: Record<string, number> | null;
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

export async function loadManifest(url = '/manifest.json'): Promise<Manifest> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`manifest request failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as Manifest;
}
