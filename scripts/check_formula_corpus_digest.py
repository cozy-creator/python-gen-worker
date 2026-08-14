#!/usr/bin/env python3
"""Fail if the local shared formula corpus does not match its pinned digest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "tests" / "testdata" / "formula_vectors.json"
DEFAULT_DIGEST = ROOT / "tests" / "testdata" / "FORMULA_VECTORS_DIGEST"


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
            print(f"formula-corpus-digest: missing {path}")
            return 2

    actual = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    recorded = recorded_digest(args.digest)
    if actual == recorded:
        print(f"formula-corpus-digest: formula_vectors.json matches {actual}")
        return 0

    print(
        "formula-corpus-digest: FAIL — the shared formula corpus changed "
        "without its digest\n"
        f"  recorded: {recorded or '<missing>'}\n"
        f"  actual:   {actual}\n\n"
        "Land the python-gen-worker semantic change and corpus first, record "
        "the new digest in both repos, then copy both files to Tensorhub."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
