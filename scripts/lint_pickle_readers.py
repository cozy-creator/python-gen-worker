#!/usr/bin/env python3
"""HARDCUT E5: nothing in this tree deserializes a pickle."""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, Tuple

REPO = Path(__file__).resolve().parents[1]

DEFAULT_ROOTS = (
    REPO / "src", REPO / "tests", REPO / "tests_v2", REPO / "scripts",
    REPO / "examples", REPO / "benchmarks",
)

DESERIALIZERS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("torch", re.compile(r"\btorch\.load\(|\btorch_mod\.load\(")),
    ("pickle", re.compile(r"\bpickle\.loads?\b|\bUnpickle[r]\b")),
    ("joblib", re.compile(r"\bjoblib\.load\(")),
    ("dill", re.compile(r"\bdill\.loads?\b")),
    ("cloudpickle", re.compile(r"\bcloudpickle\.loads?\b")),
    ("numpy", re.compile(r"allow_pickle\s*=\s*True")),
    ("pandas", re.compile(r"\bread_pickle\(")),
)

PROOF_MARKER = "# pickle-ban: proves-the-refusal"

_REFUSAL_TOKENS = ("PickleWeightRefused", "pickle_only")
_RAISES = "pytest.raises("


def _iter_files(roots: Tuple[Path, ...]) -> Iterator[Path]:
    for root in roots:
        if root.is_file():
            yield root
            continue
        for p in sorted(root.rglob("*.py")):
            if p.is_file() and "__pycache__" not in p.parts:
                yield p


def _proves_a_refusal(text: str) -> bool:
    return _RAISES in text and any(t in text for t in _REFUSAL_TOKENS)


def scan(roots: Tuple[Path, ...]) -> List[str]:
    findings: List[str] = []
    for path in _iter_files(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        proven = _proves_a_refusal(text)
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            hits = [name for name, rx in DESERIALIZERS if rx.search(line)]
            if not hits:
                continue
            if PROOF_MARKER in line and proven:
                continue
            rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
            why = "marked but the file proves no refusal" if PROOF_MARKER in line \
                else f"{'/'.join(hits)} deserializer"
            findings.append(f"{rel}:{lineno}: {why}: {stripped}")
    return findings


def _selftest() -> int:
    planted = {
        "src": 'import torch\nstate = torch.' 'load("x.bin")\n',
        "tests": 'import joblib\nm = joblib.' 'load("x.joblib")\n',
        "scripts": 'import numpy as np\na = np.load("x.npy", allow_pickle' '=True)\n',
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for tree, body in planted.items():
            (root / tree).mkdir()
            (root / tree / "planted.py").write_text(body)
            red = scan((root / tree,))
            if len(red) != 1:
                print(f"SELFTEST FAILED: {tree}/ planted call not caught: {red}",
                      file=sys.stderr)
                return 1

        (root / "unproven.py").write_text(
            'import torch\nx = torch.' f'load("x.bin")  {PROOF_MARKER}\n')
        if not scan((root / "unproven.py",)):
            print("SELFTEST FAILED: an unproven marker was honoured", file=sys.stderr)
            return 1

        (root / "proven.py").write_text(
            'import torch\n'
            'with pytest.raises(PickleWeightRefused):\n'
            '    x = torch.' f'load("x.bin")  {PROOF_MARKER}\n')
        if scan((root / "proven.py",)):
            print("SELFTEST FAILED: a proven refusal was flagged", file=sys.stderr)
            return 1

        (root / "clean.py").write_text(
            'import numpy as np\nimport pickle\n'
            'a = np.load("x.npy")\nblob = pickle.dumps(a)\n')
        if scan((root / "clean.py",)):
            print("SELFTEST FAILED: a safe read was flagged", file=sys.stderr)
            return 1
    print("lint_pickle_readers selftest: red in src/tests/scripts, green only on a proof")
    return 0


def main(argv: List[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    roots = tuple(Path(a).resolve() for a in argv) or DEFAULT_ROOTS
    findings = scan(roots)
    if findings:
        print("HARDCUT E5: a pickle deserializer is back. Whatever the stream "
              "is, its writer is reachable from tenant code — carry the payload "
              "as msgspec (`gen_worker/parallel/wire.py`) or mirror the source "
              "without the pickle. `weights_only=True` is not an exemption.\n",
              file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        print(f"\n{len(findings)} pickle deserializer(s)", file=sys.stderr)
        return 1
    print("lint_pickle_readers: no pickle deserializer in the tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
