#!/usr/bin/env python3
"""The shared grammar corpus must match the digest both repos commit; tensorhub's compilecache.TestCompiledGraphKeyVectorDigest checks the same bytes and digest file from its side."""

from __future__ import annotations

import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "testdata" / "compiled_graph_key_vectors.json"
DIGEST_FILE = ROOT / "tests" / "testdata" / "KEY_GRAMMAR_DIGEST"


def recorded_digest() -> str:
    for line in DIGEST_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split()[0]
    return ""


def main() -> int:
    for path in (CORPUS, DIGEST_FILE):
        if not path.exists():
            print(f"th#1897 gate: missing {path.relative_to(ROOT)}", file=sys.stderr)
            return 2

    actual = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    recorded = recorded_digest()
    if recorded == actual:
        print(f"th#1897 gate: compiled_graph_key_vectors.json matches {actual[:16]}…")
        return 0

    print(
        "th#1897 gate: the shared key-grammar corpus changed without its digest\n"
        f"  recorded: {recorded}\n"
        f"  actual:   {actual}\n\n"
        "The corpus is the CONTRACT between tensorhub's "
        "compilecache.IsCompiledGraphKey and gen_worker.compiled_graph_key.is_key, "
        "vendored byte-identically in both repos. Editing it is a coupled "
        "cross-repo cut: land this repo's half first (tensorhub vendors the "
        f"file byte-for-byte), then record\n  {actual}\n"
        "in KEY_GRAMMAR_DIGEST in BOTH repos, and ship hub and fleet in one "
        "window.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
