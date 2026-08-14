#!/usr/bin/env bash
# th#1897 — the GRAMMARS tensorhub and python-gen-worker must answer identically
# are pinned by shared vector corpora, and this proves the two trees hold the
# same bytes. Committed VERBATIM to both repos.
#
# Why it exists, in one paragraph: pgw#1176 re-keyed the fleet — scheme, store
# schema, resolve/publish wire, pack format and the proto — and the tensorhub
# half of the validator was simply never written. Both repos were internally
# consistent. Both CIs were green. The disagreement was only observable on a
# GPU pod, 45 minutes into a compile, at the publish gate. gen-worker's own
# source had already named the owner of the missing half in a docstring; a
# docstring is not a gate.
#
# The corpora (relative to each repo's root):
#
#   tensorhub                                        python-gen-worker
#   internal/orchestrator/compilecache/testdata/     tests/testdata/
#     compiled_graph_key_vectors.json                  compiled_graph_key_vectors.json
#   internal/orchestrator/release/testdata/          tests/testdata/
#     ref_grammar_vectors.json                         ref_grammar_vectors.json
#
# Layer 1 (offline, both repos, and the one inside the REQUIRED `gates` check):
#   each repo's own tests run its implementation against its own copy, and
#   compilecache's TestCompiledGraphKeyVectorDigest_TH1897 pins that copy to the
#   digest BOTH repos commit. A one-sided hand-edit fails there — in the tree
#   where it happened, with no network and no credentials.
#
# Layer 2 (this script): the peer's copy must be byte-equal to ours.
#   python-gen-worker is PUBLIC and readable with no token; tensorhub is private
#   and the reverse fetch does not exist. The asymmetry gives the same law the
#   proto and worker-constant gates have: PGW LANDS FIRST.
#
# What layer 2 CANNOT see, said plainly: the fleet runs a WHEEL, not pgw's
# master. Agreeing corpora are necessary, not sufficient — hub and fleet still
# ship in one window.
#
#   env:
#     GRAMMAR_PEER_REF=<branch>          peer branch to read (default: master)
#     GRAMMAR_PEER_DIR=<path>            read a local checkout's tests/testdata
#                                        instead of fetching (offline runs and
#                                        this gate's own tests)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Per-repo location of each corpus: "<local path>|<peer path>". Exactly one of
# the two local candidates exists in a checkout, which is what lets one script
# serve both repos.
CORPORA=(
  "internal/orchestrator/compilecache/testdata/compiled_graph_key_vectors.json|tests/testdata/compiled_graph_key_vectors.json"
  "internal/orchestrator/release/testdata/ref_grammar_vectors.json|tests/testdata/ref_grammar_vectors.json"
)

peer_ref="${GRAMMAR_PEER_REF:-master}"
peer_dir="${GRAMMAR_PEER_DIR:-}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

status=0
checked=0

for row in "${CORPORA[@]}"; do
  local_rel="${row%%|*}"
  peer_rel="${row##*|}"
  local_path="$here/$local_rel"
  name="$(basename "$local_rel")"

  # In python-gen-worker both corpora live at the peer path; in tensorhub they
  # live at the local path. Take whichever this checkout actually has.
  if [ ! -f "$local_path" ]; then
    local_path="$here/$peer_rel"
  fi
  if [ ! -f "$local_path" ]; then
    echo "grammar-vector-drift: missing $local_rel (and $peer_rel)" >&2
    exit 2
  fi

  fetched="$work/$name"
  # A failed fetch is a FAILURE, never a skip: a gate that passes when it could
  # not read the peer is not a gate.
  if [ -n "$peer_dir" ]; then
    if [ ! -f "$peer_dir/$name" ]; then
      echo "grammar-vector-drift: FAIL — $peer_dir/$name does not exist" >&2
      exit 1
    fi
    cp "$peer_dir/$name" "$fetched"
  else
    url="https://raw.githubusercontent.com/cozy-creator/python-gen-worker/${peer_ref}/${peer_rel}"
    if ! curl -sSfL --retry 3 --max-time 30 -o "$fetched" "$url"; then
      echo "grammar-vector-drift: FAIL — could not fetch $url" >&2
      echo "  (the peer has not vendored this corpus yet, or the fetch failed;" >&2
      echo "   re-run, or point GRAMMAR_PEER_DIR at a local python-gen-worker checkout)" >&2
      status=1
      continue
    fi
  fi

  checked=$((checked + 1))
  if cmp -s "$local_path" "$fetched"; then
    continue
  fi
  echo "grammar-vector-drift: FAIL — $name differs from python-gen-worker@${peer_ref}" >&2
  diff -u "$local_path" "$fetched" | head -40 >&2 || true
  status=1
done

if [ "$checked" -eq 0 ]; then
  echo "grammar-vector-drift: nothing was compared — that is a broken gate, not a pass" >&2
  exit 2
fi

if [ "$status" -ne 0 ]; then
  cat >&2 <<EOF

A shared GRAMMAR corpus differs between this repo and python-gen-worker@${peer_ref}.

These files are the contract between two validators that must answer
identically — tensorhub's compilecache.IsCompiledGraphKey and
release.ParseCanonicalRef against torch_compiled_graphs.identity and
gen_worker.models.refs.parse_model_ref. A disagreement is not caught by either
repo's tests, is not caught by review, and surfaces as a 45-minute mint refused
at the publish gate (th#1897).

To change a grammar:
  1. land the python-gen-worker half FIRST (this gate reads its default branch):
     the implementation, the vendored corpus, and the digest beside it
  2. copy the corpus here byte-for-byte and update
     internal/orchestrator/compilecache/testdata/KEY_GRAMMAR_DIGEST
  3. release hub and fleet in ONE window
EOF
  exit 1
fi

echo "grammar-vector-drift: $checked corpora agree with python-gen-worker@${peer_ref}"
