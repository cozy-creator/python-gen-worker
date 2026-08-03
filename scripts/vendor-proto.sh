#!/usr/bin/env bash
# pgw#944 / th#1562 — resync the VENDORED worker<->orchestrator contract from
# tensorhub, which owns the canonical copy, and regenerate src/gen_worker/pb.
#
# proto/worker_scheduler.proto in this repo is NOT hand-edited. It is a
# byte-for-byte copy of
#   tensorhub:internal/orchestrator/grpc/proto/worker_scheduler.proto
# and scripts/proto-drift-check.sh fails if it stops being one.
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
src_digest="$hub/internal/orchestrator/grpc/proto/PROTO_DIGEST"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for f in "$src" "$src_digest"; do
  if [ ! -f "$f" ]; then
    echo "vendor-proto: not a tensorhub checkout — missing $f" >&2
    exit 2
  fi
done

# The digest tensorhub publishes must actually describe the file it publishes;
# vendoring a self-inconsistent upstream would just launder the drift.
upstream_actual="$(sha256sum "$src" | cut -d' ' -f1)"
upstream_recorded="$(grep -Eo '^[0-9a-f]{64}' "$src_digest" | head -1)"
if [ "$upstream_actual" != "$upstream_recorded" ]; then
  echo "vendor-proto: upstream is inconsistent — $src does not match its own PROTO_DIGEST" >&2
  echo "  recorded: $upstream_recorded" >&2
  echo "  actual:   $upstream_actual" >&2
  exit 1
fi

cp "$src" "$here/proto/worker_scheduler.proto"
cp "$src_digest" "$here/proto/PROTO_DIGEST"
echo "vendor-proto: vendored $upstream_actual"

cd "$here"
uv run --extra dev python -m grpc_tools.protoc \
  -Iproto \
  --python_out=src/gen_worker/pb \
  --grpc_python_out=src/gen_worker/pb \
  --pyi_out=src/gen_worker/pb \
  proto/worker_scheduler.proto
echo "vendor-proto: regenerated src/gen_worker/pb"

PROTO_SKIP_PEER=1 scripts/proto-drift-check.sh
