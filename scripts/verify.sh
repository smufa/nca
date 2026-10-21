#!/usr/bin/env bash
# One-command sanity check: Python compiles, the frontend builds, and the WGSL CA
# step agrees with the ONNX model. Run this after touching the model, the
# exporter, the shaders or the frontend.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== python compile =="
uv run python -m compileall -q nca main.py scripts/shader_bench.py && echo ok

echo "== frontend build =="
( cd frontend && npm run build >/dev/null ) && echo ok

echo "== WGSL step vs ONNX (100 steps) =="
bash scripts/bench.sh crab_regenerating --steps 100 --reference

echo "all checks passed"
