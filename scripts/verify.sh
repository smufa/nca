#!/usr/bin/env bash
# One-command sanity check: Python compiles, the frontend builds, and the WGSL CA
# step agrees with the ONNX model. Run this after touching the model, the
# exporter, the shaders or the frontend.
set -euo pipefail
cd "$(dirname "$0")/.."

# `uv` is the documented runner, but it is not always on PATH (NixOS shells, CI
# images without it). Fall back to the project venv so this script always works.
if command -v uv >/dev/null 2>&1; then
    PY=(uv run python)
else
    PY=(./.venv/bin/python)
fi

echo "== python compile =="
"${PY[@]}" -m compileall -q nca main.py scripts/shader_bench.py && echo ok

echo "== tests =="
"${PY[@]}" -m pytest -q tests

echo "== exports/*/meta.json vs nca/spec.py =="
"${PY[@]}" -m scripts.check_meta

echo "== frontend build =="
( cd frontend && npm run build >/dev/null ) && echo ok

echo "== WGSL step vs ONNX (100 steps) =="
if bash scripts/bench.sh crab_regenerating --steps 100 --reference; then
    :
else
    status=$?
    if [[ $status -eq 3 ]]; then
        echo "SKIPPED: no wgpu (see scripts/bench.sh; run this step where uv is available)"
    else
        exit "$status"
    fi
fi

echo "all checks passed"
