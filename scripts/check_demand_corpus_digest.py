#!/usr/bin/env python3
"""Fail if the shared demand corpus is stale, hand-edited, or undigested.

pgw#1600. Three ways this goes red, and each is a real defect:

* the corpus is not what `scripts/gen_demand_corpus.py` produces — someone
  edited an expected byte count by hand, or the evaluator moved and the corpus
  did not;
* the digest sidecar does not match the corpus bytes — the coupled cross-repo
  bump was half-done;
* either file is missing.

The FIRST check is the important one and is why this script regenerates rather
than merely hashing: a conformance corpus whose expectations are typed in by a
human proves that the human and Go agree, not that Python and Go do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_CORPUS = ROOT / "src" / "gen_worker" / "contracts" / "demand_vectors.json"
DEFAULT_DIGEST = ROOT / "src" / "gen_worker" / "contracts" / "DEMAND_VECTORS_DIGEST"


def recorded_digest(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split()[0]
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--digest", type=Path, default=DEFAULT_DIGEST)
    args = parser.parse_args()

    for path in (args.corpus, args.digest):
        if not path.is_file():
            print(f"demand-corpus-digest: missing {path}")
            return 2

    from gen_demand_corpus import build  # noqa: PLC0415

    regenerated = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    committed = args.corpus.read_text(encoding="utf-8")
    if regenerated != committed:
        print(
            "demand-corpus-digest: FAIL — the committed corpus is not what "
            "this repository's evaluator produces.\n"
            "  Run `uv run python scripts/gen_demand_corpus.py`, review the "
            "diff (a changed byte count means the ALGEBRA moved), then record "
            "the new digest and bump tensorhub's peers.lock row."
        )
        return 1

    actual = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    recorded = recorded_digest(args.digest)
    if actual == recorded:
        print(f"demand-corpus-digest: demand_vectors.json matches {actual}")
        return 0

    print(
        "demand-corpus-digest: FAIL — the corpus changed without its digest\n"
        f"  recorded: {recorded or '<missing>'}\n"
        f"  actual:   {actual}\n\n"
        "Record the new digest here, land it, then bump tensorhub's "
        "peers.lock row to this repository's merge commit."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
