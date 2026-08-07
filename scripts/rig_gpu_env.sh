#!/usr/bin/env bash
# pgw#983 — build the rig's GPU interpreter environment, reproducibly.
#
# This box's driver is 570.211.01 (CUDA 12.8) and the repo pins
# torch==2.13.0+cu130, which needs a 580-series driver — so the DEFAULT rig
# runs cardless and supplies a synthetic `sm` (pgw#983). This script builds an
# ISOLATED second interpreter that can use the card for real.
#
# WHY cu126 AND NOT cu128. The cu128 index has NO torch 2.13.0 — it stops at
# 2.11.0. `torch` is one of the axes `aot_serve.verify_declared` checks
# STRICTLY, so a cu128 venv would mint cells two minor versions off the fleet:
# further away, not closer. cu126 carries 2.13.0, and CUDA minor-version
# compatibility runs a cu126 build on a 12.8 driver. Only the `cuda` axis
# differs from the fleet (12.6 vs 13.0), and that is stated wherever a
# GPU-minted cell is reported.
#
# WHAT IT DOES NOT DO: no driver change, no system package, nothing outside
# the venv. The repo's own `.venv` is untouched and stays the default.
#
#   ./scripts/rig_gpu_env.sh          # build/refresh, then print the exports
#   eval "$(./scripts/rig_gpu_env.sh --export-only)"
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The venv is ~6 GB, so every worktree SHARES the canonical checkout's copy
# rather than building its own. `--git-common-dir` names the main .git even
# from inside a linked worktree; its parent is that checkout.
MAIN="$(cd "$(git -C "$REPO" rev-parse --git-common-dir)/.." && pwd)"
VENV="${GEN_WORKER_RIG_GPU_VENV:-$MAIN/.venv-cu126}"
SP="$VENV/lib/python3.12/site-packages"
SHIM="$VENV/cuda-home"
INDEX="https://download.pytorch.org/whl/cu126"

export_only=0
[ "${1:-}" = "--export-only" ] && export_only=1

build() {
  if [ ! -x "$VENV/bin/python" ]; then
    echo "rig-gpu: creating $VENV" >&2
    uv venv "$VENV" --python 3.12 >&2
  fi
  if ! "$VENV/bin/python" -c "import torch" 2>/dev/null; then
    echo "rig-gpu: installing torch 2.13.0+cu126 (~6 GB)" >&2
    UV_HTTP_TIMEOUT=600 uv pip install --python "$VENV" --index-url "$INDEX" \
      "torch==2.13.0" >&2
    # The AOTI CUDA compile needs CUDA HEADERS and ptxas. The torch wheels
    # ship runtime libs only, so these two carry the rest. `nvcc` itself is
    # NOT required — inductor compiles the wrapper with g++ and the kernels
    # through triton/ptxas — but CUDA_HOME must point at a tree that LOOKS
    # like a toolkit, which is what the shim below is.
    UV_HTTP_TIMEOUT=600 uv pip install --python "$VENV" \
      --index-url https://pypi.org/simple \
      "nvidia-cuda-nvcc-cu12==12.6.*" "nvidia-cuda-cccl-cu12==12.6.*" >&2
    # The SDK's own runtime deps, from PyPI so torch is not re-resolved to a
    # different CUDA build.
    UV_HTTP_TIMEOUT=600 uv pip install --python "$VENV" \
      --index-url https://pypi.org/simple \
      "grpcio>=1.82.1" "msgspec>=0.18.6" "protobuf>=7.35.0" "requests>=2.32.0" \
      "boto3>=1.41.0" "psutil>=7.0.0" "pyyaml>=6.0.0" "blake3>=1.0.0" \
      "huggingface-hub>=0.26.0" "gguf>=0.10.0" "tomli-w>=1.0.0" \
      "c2pa-python>=0.36" numpy safetensors pillow pytest >&2
  fi
  # A CUDA_HOME-shaped tree assembled from the wheels. Rebuilt every run: it
  # is only symlinks, and a stale one is a compile error three legs later.
  rm -rf "$SHIM"
  mkdir -p "$SHIM/include"
  ln -sfn "$SP/nvidia/cuda_runtime/lib" "$SHIM/lib64"
  ln -sfn "$SP/nvidia/cuda_nvcc/bin" "$SHIM/bin"
  ln -sfn "$SP/nvidia/cuda_nvcc/nvvm" "$SHIM/nvvm"
  # Headers come from THREE wheels and inductor needs all of them: the runtime
  # headers include `crt/host_defines.h` (nvcc wheel) and `nv/target` (cccl
  # wheel). Each was a separate compile failure, in that order.
  for d in "$SP/nvidia/cuda_runtime/include" "$SP/nvidia/cuda_nvcc/include" \
           "$SP/nvidia/cuda_cccl/include"; do
    [ -d "$d" ] || continue
    for f in "$d"/*; do ln -sfn "$f" "$SHIM/include/$(basename "$f")"; done
  done
}

[ "$export_only" = 1 ] || build
[ -d "$SHIM/include" ] || build

echo "export GEN_WORKER_RIG_GPU_PYTHON='$VENV/bin/python'"
echo "export CUDA_HOME='$SHIM'"
