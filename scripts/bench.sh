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

# Exit code 3 means "no usable GPU here" (not a failure): verify.sh turns it into a
# SKIP. Two things look like that -- no wgpu installed, and no adapter (below).
SKIP=3

LOADER="$(ls /nix/store/*vulkan-loader*/lib/libvulkan.so.1 2>/dev/null | head -1 || true)"
if [[ -n "$LOADER" ]]; then
  LOADER_DIR="$(dirname "$LOADER")"
  export LD_LIBRARY_PATH="${LOADER_DIR}:/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/run/opengl-driver/share/vulkan/icd.d/intel_icd.x86_64.json}"

if command -v uv >/dev/null 2>&1; then
  RUN=(uv run --with wgpu --with onnxruntime python scripts/shader_bench.py)
elif ./.venv/bin/python -c "import wgpu" >/dev/null 2>&1; then
  # No uv: only usable if the venv already has wgpu (e.g. a long-lived cluster venv).
  RUN=(./.venv/bin/python scripts/shader_bench.py)
else
  echo "bench.sh: needs \`uv\` (it pulls wgpu into an ephemeral env) - skipping." >&2
  echo "          pip install wgpu, or add a \`bench\` dependency group and re-run." >&2
  exit "$SKIP"
fi

# A missing adapter is a property of the machine, not a failure of the shader, and
# it is the common case inside CI/sandboxes -- so map it to SKIP instead of letting
# verify.sh report a red build for an unimplemented test.
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
set +e
"${RUN[@]}" "$@" 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
set -e
if [[ "$status" -ne 0 ]] && grep -qE "No suitable graphics adapter|No adapter found|no suitable adapter" "$log"; then
  echo "bench.sh: no usable Vulkan adapter on this machine - skipping." >&2
  exit "$SKIP"
fi
exit "$status"
