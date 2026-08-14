#!/usr/bin/env bash
# th#1914 / pgw#1239 — Hub launch names and their worker meanings must remain
# byte-identical between public python-gen-worker and every private consumer.
# This script is committed VERBATIM in all participating repositories.
#
# Layer 1 in each repo pins its local corpus. Layer 2 runs from each private
# consumer and compares to public python-gen-worker. Running this in PGW
# without an explicit peer directory is local-only by design; a public repo
# cannot fetch its private consumers and must never call a self-comparison
# peer proof.
#
# Direction: PGW LANDS FIRST.
#
#   HUB_WORKER_BOUNDARY_PEER_REF=<ref>  PGW ref to fetch (default: master)
#   HUB_WORKER_BOUNDARY_PEER_DIR=<dir>  local PGW tests/testdata directory
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pgw_rel="tests/testdata"
hub_rel="internal/wirecontract/testdata"
trainer_rel="image_lora_finetuner/tests/testdata"
corpus="hub_worker_boundary_contracts.json"
digest="HUB_WORKER_BOUNDARY_CONTRACTS_DIGEST"

if [ -f "$here/$pgw_rel/$corpus" ]; then
  local_dir="$here/$pgw_rel"
  side="pgw"
elif [ -f "$here/$hub_rel/$corpus" ]; then
  local_dir="$here/$hub_rel"
  side="hub"
elif [ -f "$here/$trainer_rel/$corpus" ]; then
  local_dir="$here/$trainer_rel"
  side="trainer"
else
  echo "hub-worker-boundary-drift: no $corpus under $here" >&2
  exit 2
fi

for name in "$corpus" "$digest"; do
  if [ ! -f "$local_dir/$name" ]; then
    echo "hub-worker-boundary-drift: missing $local_dir/$name" >&2
    exit 2
  fi
done

recorded="$(grep -Eo '^[0-9a-f]{64}' "$local_dir/$digest" | head -1 || true)"
actual="$(sha256sum "$local_dir/$corpus" | cut -d' ' -f1)"
if [ -z "$recorded" ]; then
  echo "hub-worker-boundary-drift: $local_dir/$digest contains no sha256" >&2
  exit 2
fi
if [ "$recorded" != "$actual" ]; then
  echo "hub-worker-boundary-drift: FAIL — local corpus changed without its digest" >&2
  echo "  recorded: $recorded" >&2
  echo "  actual:   $actual" >&2
  exit 1
fi
echo "hub-worker-boundary-drift: local corpus matches $actual"

peer_dir="${HUB_WORKER_BOUNDARY_PEER_DIR:-}"
if [ "$side" = "pgw" ] && [ -z "$peer_dir" ]; then
  echo "hub-worker-boundary-drift: PGW layer 1 complete; no self-comparison"
  exit 0
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
peer_ref="${HUB_WORKER_BOUNDARY_PEER_REF:-master}"
for name in "$corpus" "$digest"; do
  if [ -n "$peer_dir" ]; then
    if [ ! -f "$peer_dir/$name" ]; then
      echo "hub-worker-boundary-drift: FAIL — $peer_dir/$name does not exist" >&2
      exit 1
    fi
    cp "$peer_dir/$name" "$work/$name"
  else
    url="https://raw.githubusercontent.com/cozy-creator/python-gen-worker/${peer_ref}/${pgw_rel}/${name}"
    if ! curl -sSfL --retry 3 --max-time 30 -o "$work/$name" "$url"; then
      echo "hub-worker-boundary-drift: FAIL — could not fetch $url" >&2
      exit 1
    fi
  fi
done

peer_recorded="$(grep -Eo '^[0-9a-f]{64}' "$work/$digest" | head -1 || true)"
peer_actual="$(sha256sum "$work/$corpus" | cut -d' ' -f1)"
if [ -z "$peer_recorded" ] || [ "$peer_recorded" != "$peer_actual" ]; then
  echo "hub-worker-boundary-drift: FAIL — PGW peer corpus does not match its digest" >&2
  echo "  recorded: ${peer_recorded:-<missing>}" >&2
  echo "  actual:   $peer_actual" >&2
  exit 1
fi

status=0
for name in "$corpus" "$digest"; do
  if ! cmp -s "$local_dir/$name" "$work/$name"; then
    echo "hub-worker-boundary-drift: FAIL — $name differs from python-gen-worker@${peer_ref}" >&2
    diff -u "$local_dir/$name" "$work/$name" | head -40 >&2 || true
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  echo "hub-worker-boundary-drift: land PGW first, then copy corpus, digest and script verbatim to private consumers" >&2
  exit 1
fi

echo "hub-worker-boundary-drift: corpus and digest agree with public python-gen-worker@${peer_ref}"
