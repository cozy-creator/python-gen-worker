#!/usr/bin/env bash
# th#1562 / pgw#944 — the worker<->orchestrator .proto lives in two repos and
# must never diverge. tensorhub's copy is CANONICAL; python-gen-worker vendors
# it byte-for-byte. This script is committed VERBATIM to both repos.
#
# Layer 1 (always, offline, both repos): the local copy must hash to the digest
#   recorded in PROTO_DIGEST beside it. Both repos commit the SAME digest, so a
#   one-sided hand-edit fails here — in the repo where it happened, with no
#   network and no credentials. This is the layer that catches the mistake in
#   time, which is why it has no external dependency of any kind.
#
# Layer 2 (only where the peer is readable): the peer repo's copy must be
#   byte-equal to ours. python-gen-worker is a PUBLIC repo, so tensorhub fetches
#   it over plain https with no token. tensorhub is PRIVATE, so the reverse
#   fetch does not exist — layer 1 is python-gen-worker's whole gate, by design
#   rather than by omission. Layer 2 closes the one hole layer 1 leaves: an edit
#   that updates proto AND PROTO_DIGEST in only one of the two repos.
#
#   env:
#     PROTO_PEER_REF=<branch>  peer branch to compare against (default: dev)
#     PROTO_SKIP_PEER=1        run layer 1 only (offline local runs)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Per-repo location of the contract. Exactly one of these exists in a checkout.
for candidate in \
  "$here/internal/orchestrator/grpc/proto" \
  "$here/proto"
do
  if [ -f "$candidate/worker_scheduler.proto" ]; then
    proto_dir="$candidate"
    break
  fi
done

if [ -z "${proto_dir:-}" ]; then
  echo "proto-drift-check: no worker_scheduler.proto found under $here" >&2
  exit 2
fi

proto="$proto_dir/worker_scheduler.proto"
digest_file="$proto_dir/PROTO_DIGEST"

if [ ! -f "$digest_file" ]; then
  echo "proto-drift-check: missing $digest_file" >&2
  exit 2
fi

actual="$(sha256sum "$proto" | cut -d' ' -f1)"
recorded="$(grep -Eo '^[0-9a-f]{64}' "$digest_file" | head -1)"

if [ -z "$recorded" ]; then
  echo "proto-drift-check: $digest_file contains no sha256" >&2
  exit 2
fi

# ---- Layer 1: local copy vs recorded digest -------------------------------
if [ "$actual" != "$recorded" ]; then
  cat >&2 <<EOF
proto-drift-check: FAIL — the wire contract changed without its digest.

  $proto
    recorded: $recorded
    actual:   $actual

The worker<->orchestrator .proto is shared by tensorhub and python-gen-worker
and MUST be identical in both. Changing it in one repo alone silently breaks
the other's wire compatibility.

To change the contract:
  1. edit tensorhub's canonical copy
     (internal/orchestrator/grpc/proto/worker_scheduler.proto)
  2. update PROTO_DIGEST beside it:
       sha256sum internal/orchestrator/grpc/proto/worker_scheduler.proto
  3. in python-gen-worker: scripts/vendor-proto.sh /path/to/tensorhub
  4. land BOTH repos' PRs (python-gen-worker first — tensorhub's layer-2
     check reads python-gen-worker's branch)
EOF
  exit 1
fi

echo "proto-drift-check: local copy matches PROTO_DIGEST ($actual)"

# ---- Layer 2: peer repo's copy (public fetch, no credentials) --------------
if [ "${PROTO_SKIP_PEER:-0}" = "1" ]; then
  echo "proto-drift-check: PROTO_SKIP_PEER=1 — skipping peer comparison"
  exit 0
fi

# Only tensorhub can run layer 2: python-gen-worker is public and readable
# without a token, tensorhub is private and is not.
case "$proto_dir" in
  */internal/orchestrator/grpc/proto) ;;
  *)
    echo "proto-drift-check: peer (tensorhub) is private and unreadable from here — layer 1 only"
    exit 0
    ;;
esac

peer_ref="${PROTO_PEER_REF:-dev}"
peer_url="https://raw.githubusercontent.com/cozy-creator/python-gen-worker/${peer_ref}/proto/worker_scheduler.proto"
peer_file="$(mktemp)"
trap 'rm -f "$peer_file"' EXIT

if ! curl -sSfL --retry 3 --max-time 30 -o "$peer_file" "$peer_url"; then
  echo "proto-drift-check: FAIL — could not fetch the vendored copy at $peer_url" >&2
  echo "  (a gate that passes when the fetch fails is not a gate; re-run, or" >&2
  echo "   set PROTO_SKIP_PEER=1 for a deliberate offline run)" >&2
  exit 1
fi

if ! cmp -s "$proto" "$peer_file"; then
  cat >&2 <<EOF
proto-drift-check: FAIL — python-gen-worker's vendored copy has drifted.

  canonical: $proto
  vendored:  $peer_url

$(diff -u "$proto" "$peer_file" | head -60)

Resync python-gen-worker (scripts/vendor-proto.sh) and land that PR first.
EOF
  exit 1
fi

echo "proto-drift-check: python-gen-worker@${peer_ref} vendored copy is byte-identical"
