#!/usr/bin/env bash
# Reproducible native-extension build (pgw#860): compile csrc/ against the
# fleet torch pin inside the pinned devel toolchain image. No host toolkit,
# no GPU needed — compile + registration smoke only.
#
#   scripts/native/build_native_kernels.sh [out_dir]
#
# The devel image is the toolchain twin of the fleet base
# (pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime, tensorhub baseimage.go);
# both carry the same torch 2.13.0+cu130. Bump BOTH pins together.
set -euo pipefail

IMAGE="${COZY_NATIVE_BUILD_IMAGE:-pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel@sha256:c74b2049d204a55233abace509fdb1f6bdbe2d169bee382ed2f06cac00ba76c4}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${REPO}/dist-native}"
mkdir -p "${OUT}"

docker run --rm \
  -v "${REPO}:/repo:ro" -v "${OUT}:/out" -w /repo \
  "${IMAGE}" \
  bash -c 'command -v ninja >/dev/null \
             || pip install --quiet --break-system-packages ninja; \
           python csrc/build.py /out'

echo "[build_native_kernels] done: ${OUT}/libcozy_kernels.so"
