#!/usr/bin/env bash
# gw#429: runs ON the rented pod (via ssh). Clones python-gen-worker at a
# pinned SHA, installs the real torch/CUDA stack, and runs the GPU-marked
# tests plus the torch-dependent suites default (torch-less) CI can't
# exercise. Two independent junit files so one crash doesn't hide the other.
set -euo pipefail

: "${REPO_SHA:?REPO_SHA is required}"
REPO_URL="${REPO_URL:-https://github.com/cozy-creator/python-gen-worker.git}"
WORKDIR="${WORKDIR:-/workspace/gen-worker}"

nvidia-smi || echo "WARNING: nvidia-smi failed — pod may not have GPU access" >&2

rm -rf "$WORKDIR"
git clone --quiet "$REPO_URL" "$WORKDIR"
cd "$WORKDIR"
git checkout --quiet "$REPO_SHA"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --extra dev --extra torch --extra images

status=0

uv run pytest tests/ -m gpu -v --junitxml="$WORKDIR/gpu-marked-junit.xml" \
  || status=1

uv run pytest \
  tests/test_fp8_and_emergency_loading.py \
  tests/test_compile_cache.py \
  tests/convert/ \
  -v --junitxml="$WORKDIR/torch-suite-junit.xml" \
  || status=1

exit "$status"
