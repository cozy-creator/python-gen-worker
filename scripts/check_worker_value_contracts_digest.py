#!/usr/bin/env python3
"""Fail if the local worker-value carrier misses its pinned digest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = (
    ROOT / "src" / "gen_worker" / "contracts" / "worker_value_contracts.json"
)
DEFAULT_DIGEST = (
    ROOT / "src" / "gen_worker" / "contracts" / "WORKER_VALUE_CONTRACTS_DIGEST"
)


def recorded_digest(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            value = line.split()[0]
            if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
                return value
            return ""
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--digest", type=Path, default=DEFAULT_DIGEST)
    args = parser.parse_args()

    for path in (args.corpus, args.digest):
        if not path.is_file():
            print(f"worker-value-contracts-digest: missing {path}")
            return 2

    actual = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    recorded = recorded_digest(args.digest)
    if actual == recorded:
        print(f"worker-value-contracts-digest: corpus matches {actual}")
        return 0

    print(
        "worker-value-contracts-digest: FAIL — the public worker-value corpus "
        "changed without its digest\n"
        f"  recorded: {recorded or '<missing>'}\n"
        f"  actual:   {actual}\n\n"
        "Land python-gen-worker first, then copy corpus and digest verbatim "
        "to tensorhub."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
