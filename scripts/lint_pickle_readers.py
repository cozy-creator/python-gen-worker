#!/usr/bin/env python3
"""HARDCUT E5: nothing in this tree deserializes a pickle.

Pickles are banned platform-wide (pgw#498/#884/#1273/#1275) — reading one IS
the banned act, so a `weights_only=True` site is refused exactly like a bare
one. The remedy for a legacy source is to mirror it without the pickle.

WHY THIS IS A LINT AND NOT A TEST. Two scans already existed inside
`tests/test_untrusted_pickle_and_secret_sources_pgw498_pgw884.py`, and pgw#1264
took the `tests` job off the merge path — so the fence CI actually runs was the
narrower one, and only over `src/`. This runs in `fast gates`, over every Python
tree in the repo, and the two pytest copies are gone.

WHAT IT LOOKS FOR is the SHAPE of a deserialization, because that is what stops
one coming back: a new site is silently safe on today's torch, silently unsafe
under `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`, and no test of the surrounding
feature notices. Writing a pickle is not scanned — writing one executes
nothing.

THERE IS NO PATH ALLOWLIST, deliberately: a path exemption lets a real gadget
hide behind a filename. A site is permitted only where it is PROVEN REFUSED —
the line carries the marker below AND the same file both raises and asserts one
of the platform's pickle refusals. Delete the proof and the line goes red.

Usage:

    python scripts/lint_pickle_readers.py [PATH ...]
    python scripts/lint_pickle_readers.py --selftest

Defaults to `src/`, `tests/`, `tests_v2/`, `scripts/`, `examples/` and
`benchmarks/`.
"""

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

#: Every shape that hands bytes to a class-naming decoder. The patterns are
#: written so this file does not match itself — no self-exemption exists, and
#: none is needed.
DESERIALIZERS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("torch", re.compile(r"\btorch\.load\(|\btorch_mod\.load\(")),
    ("pickle", re.compile(r"\bpickle\.loads?\b|\bUnpickle[r]\b")),
    ("joblib", re.compile(r"\bjoblib\.load\(")),
    ("dill", re.compile(r"\bdill\.loads?\b")),
    ("cloudpickle", re.compile(r"\bcloudpickle\.loads?\b")),
    ("numpy", re.compile(r"allow_pickle\s*=\s*True")),
    ("pandas", re.compile(r"\bread_pickle\(")),
)

#: The line-level statement that this call is an INPUT to a proof of the ban.
PROOF_MARKER = "# pickle-ban: proves-the-refusal"

#: ...honoured only in a file that really does raise and assert a refusal. Both
#: names below are refusals this platform raises on a pickle by name.
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
    """RED on a planted call in every tree; GREEN only on a proven refusal."""
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

        # A marker without a proof is still refused — that is what makes the
        # exemption a proof rather than an allowlist.
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

        # numpy's default is `allow_pickle=False`; only the opt-in is a reader.
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
