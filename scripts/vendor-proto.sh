#!/usr/bin/env bash
# pgw#944 / th#1562 — resync the VENDORED worker<->orchestrator contract from
# tensorhub, which owns the canonical copy, and regenerate src/gen_worker/pb.
#
# proto/worker_scheduler.proto in this repo is NOT hand-edited. It is a
# byte-for-byte copy of
#   tensorhub:internal/orchestrator/grpc/proto/worker_scheduler.proto
# (tensorhub pins the bytes it expects in internal/wirecontract/peers.lock).
#
#   scripts/vendor-proto.sh ~/cozy/tensorhub
#
# Then commit proto/, src/gen_worker/pb/ and land this repo's PR BEFORE
# tensorhub's — tensorhub's gate reads this repo's branch.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: scripts/vendor-proto.sh <path-to-tensorhub-checkout>" >&2
  exit 2
fi

hub="$1"
src="$hub/internal/orchestrator/grpc/proto/worker_scheduler.proto"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$src" ]; then
  echo "vendor-proto: not a tensorhub checkout — missing $src" >&2
  exit 2
fi

# The digest tensorhub publishes must actually describe the file it publishes;
# vendoring a self-inconsistent upstream would just launder the drift.

cp "$src" "$here/proto/worker_scheduler.proto"
echo "vendor-proto: vendored $(sha256sum "$src" | cut -d' ' -f1)"

cd "$here"
uv run --extra dev python -m grpc_tools.protoc \
  -Iproto \
  --python_out=src/gen_worker/pb \
  --grpc_python_out=src/gen_worker/pb \
  --pyi_out=src/gen_worker/pb \
  proto/worker_scheduler.proto
echo "vendor-proto: regenerated src/gen_worker/pb"

