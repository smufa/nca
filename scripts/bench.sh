#!/usr/bin/env bash
# Run the headless WGSL correctness/benchmark harness with the right Vulkan env.
#
#   bash scripts/bench.sh crab_regenerating --steps 150 --reference
#   bash scripts/bench.sh crab_regenerating --grid 256
#   bash scripts/bench.sh crab_regenerating --shader scripts/variants/pass1_vec4.wgsl
#
# On NixOS the Vulkan loader and ICDs are not on the default library path, so wgpu
# reports "No suitable graphics adapter found". Set VK_ICD_FILENAMES to
# .../lvp_icd.x86_64.json to force the software (lavapipe) adapter instead.
set -euo pipefail
cd "$(dirname "$0")/.."

LOADER="$(ls /nix/store/*vulkan-loader*/lib/libvulkan.so.1 2>/dev/null | head -1 || true)"
if [[ -n "$LOADER" ]]; then
  LOADER_DIR="$(dirname "$LOADER")"
  export LD_LIBRARY_PATH="${LOADER_DIR}:/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/run/opengl-driver/share/vulkan/icd.d/intel_icd.x86_64.json}"

exec uv run --with wgpu --with onnxruntime python scripts/shader_bench.py "$@"
