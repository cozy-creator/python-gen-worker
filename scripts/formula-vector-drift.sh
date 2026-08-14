#!/usr/bin/env bash
# th#1914 / pgw#1236 — formula_vectors.json is a byte-identical contract
# between Tensorhub's Go formula implementation and python-gen-worker's Python
# implementation. This script is committed VERBATIM in both repositories.
#
# Layer 1 (both repos, offline): the local corpus must match the shared pinned
# digest beside it. Layer 2 (Tensorhub only): compare the local corpus and
# digest to public python-gen-worker. Tensorhub is private, so a default run in
# python-gen-worker deliberately stops after layer 1; it never self-compares
# and calls that peer parity.
#
# Direction: PGW LANDS FIRST. Tensorhub reads PGW's public tree.
#
#   FORMULA_VECTOR_PEER_REF=<branch>  PGW ref to fetch (default: master)
#   FORMULA_VECTOR_PEER_DIR=<dir>     local PGW tests/testdata directory
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hub_rel="internal/formula/testdata"
pgw_rel="tests/testdata"
corpus="formula_vectors.json"
digest="FORMULA_VECTORS_DIGEST"

if [ -f "$here/$hub_rel/$corpus" ]; then
  local_dir="$here/$hub_rel"
  side="hub"
elif [ -f "$here/$pgw_rel/$corpus" ]; then
  local_dir="$here/$pgw_rel"
  side="pgw"
else
  echo "formula-vector-drift: no $corpus under $here" >&2
  exit 2
fi

for name in "$corpus" "$digest"; do
  if [ ! -f "$local_dir/$name" ]; then
    echo "formula-vector-drift: missing $local_dir/$name" >&2
    exit 2
  fi
done

recorded="$(grep -Eo '^[0-9a-f]{64}' "$local_dir/$digest" | head -1 || true)"
actual="$(sha256sum "$local_dir/$corpus" | cut -d' ' -f1)"
if [ -z "$recorded" ]; then
  echo "formula-vector-drift: $local_dir/$digest contains no sha256" >&2
  exit 2
fi
if [ "$recorded" != "$actual" ]; then
  echo "formula-vector-drift: FAIL — local corpus changed without its digest" >&2
  echo "  recorded: $recorded" >&2
  echo "  actual:   $actual" >&2
  exit 1
fi
echo "formula-vector-drift: local corpus matches $actual"

peer_dir="${FORMULA_VECTOR_PEER_DIR:-}"
if [ "$side" = "pgw" ] && [ -z "$peer_dir" ]; then
  echo "formula-vector-drift: Tensorhub is private — PGW layer 1 complete; no peer comparison"
  exit 0
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
peer_ref="${FORMULA_VECTOR_PEER_REF:-master}"
for name in "$corpus" "$digest"; do
  if [ -n "$peer_dir" ]; then
    if [ ! -f "$peer_dir/$name" ]; then
      echo "formula-vector-drift: FAIL — $peer_dir/$name does not exist" >&2
      exit 1
    fi
    cp "$peer_dir/$name" "$work/$name"
  else
    url="https://raw.githubusercontent.com/cozy-creator/python-gen-worker/${peer_ref}/${pgw_rel}/${name}"
    if ! curl -sSfL --retry 3 --max-time 30 -o "$work/$name" "$url"; then
      echo "formula-vector-drift: FAIL — could not fetch $url" >&2
      exit 1
    fi
  fi
done

status=0
for name in "$corpus" "$digest"; do
  if ! cmp -s "$local_dir/$name" "$work/$name"; then
    echo "formula-vector-drift: FAIL — $name differs from python-gen-worker@${peer_ref}" >&2
    diff -u "$local_dir/$name" "$work/$name" | head -40 >&2 || true
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  echo "formula-vector-drift: land PGW first, then copy corpus and digest verbatim to Tensorhub" >&2
  exit 1
fi

echo "formula-vector-drift: corpus and digest agree with python-gen-worker@${peer_ref}"
