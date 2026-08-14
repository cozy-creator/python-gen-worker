#!/usr/bin/env bash
# th#1914 / pgw#1237 / cl#56 — the Cozy runner and worker must agree on the
# exact environment names and the path semantics behind them. This script is
# committed VERBATIM in python-gen-worker and cozy-local.
#
# Layer 1 in each repo pins its local corpus. Layer 2 runs from cozy-local and
# compares to public python-gen-worker. Running this in PGW without an explicit
# peer directory is local-only by design; a public repo cannot fetch its
# private consumer and must never call a self-comparison peer proof.
#
# Direction: PGW LANDS FIRST.
#
#   COZY_RUNTIME_ENV_PEER_REF=<branch>  PGW ref to fetch (default: master)
#   COZY_RUNTIME_ENV_PEER_DIR=<dir>     local PGW tests/testdata directory
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pgw_rel="tests/testdata"
cozy_rel="internal/cli/testdata"
corpus="cozy_runtime_env_vectors.json"
digest="COZY_RUNTIME_ENV_DIGEST"

if [ -f "$here/$pgw_rel/$corpus" ]; then
  local_dir="$here/$pgw_rel"
  side="pgw"
elif [ -f "$here/$cozy_rel/$corpus" ]; then
  local_dir="$here/$cozy_rel"
  side="cozy"
else
  echo "cozy-runtime-env-drift: no $corpus under $here" >&2
  exit 2
fi

for name in "$corpus" "$digest"; do
  if [ ! -f "$local_dir/$name" ]; then
    echo "cozy-runtime-env-drift: missing $local_dir/$name" >&2
    exit 2
  fi
done

recorded="$(grep -Eo '^[0-9a-f]{64}' "$local_dir/$digest" | head -1 || true)"
actual="$(sha256sum "$local_dir/$corpus" | cut -d' ' -f1)"
if [ -z "$recorded" ]; then
  echo "cozy-runtime-env-drift: $local_dir/$digest contains no sha256" >&2
  exit 2
fi
if [ "$recorded" != "$actual" ]; then
  echo "cozy-runtime-env-drift: FAIL — local corpus changed without its digest" >&2
  echo "  recorded: $recorded" >&2
  echo "  actual:   $actual" >&2
  exit 1
fi
echo "cozy-runtime-env-drift: local corpus matches $actual"

peer_dir="${COZY_RUNTIME_ENV_PEER_DIR:-}"
if [ "$side" = "pgw" ] && [ -z "$peer_dir" ]; then
  echo "cozy-runtime-env-drift: PGW layer 1 complete; no self-comparison"
  exit 0
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
peer_ref="${COZY_RUNTIME_ENV_PEER_REF:-master}"
for name in "$corpus" "$digest"; do
  if [ -n "$peer_dir" ]; then
    if [ ! -f "$peer_dir/$name" ]; then
      echo "cozy-runtime-env-drift: FAIL — $peer_dir/$name does not exist" >&2
      exit 1
    fi
    cp "$peer_dir/$name" "$work/$name"
  else
    url="https://raw.githubusercontent.com/cozy-creator/python-gen-worker/${peer_ref}/${pgw_rel}/${name}"
    if ! curl -sSfL --retry 3 --max-time 30 -o "$work/$name" "$url"; then
      echo "cozy-runtime-env-drift: FAIL — could not fetch $url" >&2
      exit 1
    fi
  fi
done

status=0
for name in "$corpus" "$digest"; do
  if ! cmp -s "$local_dir/$name" "$work/$name"; then
    echo "cozy-runtime-env-drift: FAIL — $name differs from python-gen-worker@${peer_ref}" >&2
    diff -u "$local_dir/$name" "$work/$name" | head -40 >&2 || true
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  echo "cozy-runtime-env-drift: land PGW first, then copy corpus and digest verbatim to cozy-local" >&2
  exit 1
fi

echo "cozy-runtime-env-drift: corpus and digest agree with python-gen-worker@${peer_ref}"
