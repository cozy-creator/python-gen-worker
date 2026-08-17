#!/usr/bin/env python3
"""pgw#1308: the materialization hatch may not grow silently.

Under mixed-CAS a model on disk is a PROJECTED TREE — non-tensor files are
symlinks into the object store, tensor containers are ~128 byte pointer stubs,
and the real bytes are read through tensorfs's tensor reader. Nothing is
copied. Two things can quietly undo that, and this refuses both.

**Whole-tree materialization is being retired**, and this is its CENSUS.
It wrote a complete second copy of every byte the store already held, so a
resident model occupied disk twice (pgw#1296(a)). Upstream tensorfs deleted
the symbol (tensorfs#58). This repo still has exactly one production caller
and its test scaffolding; both are NAMED in `RETIRED_RESIDUE` below with the
issue that executes them. A site that is not on that list is refused, so the
residue can only shrink -- which is the whole point of writing it down
instead of leaving "we should stop doing this" in a design doc.

**`extract()` survives as a DISCOURAGED hatch**, in Paul's words *"an escape
hatch that lets you materialize tensors, but it's not recommended (defeats the
whole purpose of this system)"*. `docs/mixed-cas-layout.md` §9 lists its whole
user set, and that list — not a size limit — is the control. So a call is
permitted only when the line NAMES the audit row it belongs to, and the row is
one §9 actually has. A new row is a decision someone makes on purpose, in the
design doc, not a call site that appeared in a diff.

WHY A LINT AND NOT A TEST. The hatch's whole failure mode is a site nobody
looked at: every individual `extract()` works perfectly and the suite stays
green while the architecture erodes underneath it. Only a census notices, and
it belongs in `fast gates` where it runs on the merge path.

WHY THE HATCH SCAN IS FILE-SCOPED. `extract` is an ordinary English verb and
this tree has unrelated ones (`guard_closure`'s, `zipfile`'s). The hatch is
reachable only through a tensorfs tensor reader, so the scan considers a file
only when it imports tensorfs at all. That is precise rather than broad: a
file with no tensorfs import cannot reach it.

This file exempts ITSELF and nothing else. A fence that cannot spell what it
refuses is a worse fence, and the exemption is not a hiding place: the
`--selftest` arm proves both patterns still fire, on planted files, every CI
run.

Usage:

    python scripts/lint_materialization_hatch.py [PATH ...]
    python scripts/lint_materialization_hatch.py --selftest

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
SELF = Path(__file__).resolve()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_scope import is_unowned  # noqa: E402

DEFAULT_ROOTS = (
    REPO / "src", REPO / "tests", REPO / "tests_v2", REPO / "scripts",
    REPO / "examples", REPO / "benchmarks",
)

#: Whole-tree materialization -- the symbol upstream deleted (tensorfs#58).
RETIRED = re.compile(r"\bmaterialize" r"_repository\b")

#: The residue, NAMED. Each entry is a path that may still mention the retired
#: symbol, with the issue that deletes it. Not an allowlist for new debt: a
#: path not on this list is refused, and the list only ever shrinks.
RETIRED_RESIDUE = {
    # The chokepoint itself. pgw#1295 replaces this one line with a projected
    # tree; it is gated on pgw#1303 (Paul's ruling on author slots that demand
    # a real model DIRECTORY), because every `from_pretrained(path)` consumer
    # downstream of it reads weights through the tree today.
    "src/gen_worker/models/cozy_snapshot.py",
    # Its test scaffolding: a LocalCAS subclass that counts and interrupts the
    # copy. Dies in the same commit as the caller it exercises.
    "tests/test_snapshot_v2_fill_pgw781.py",
}

#: The hatch itself, scanned only where tensorfs is imported.
HATCH = re.compile(r"\.extract\s*\(")

#: A file reaches the hatch only if it can reach a tensor reader.
_TENSORFS_IMPORT = re.compile(r"\b(?:from|import)\s+\S*tensorfs\b")

#: The line-level statement of WHICH §9 row this call is.
MARKER = "# mixed-cas-hatch:"

#: Every row `docs/mixed-cas-layout.md` §9 admits. Adding one here without
#: adding it there is the mistake this list exists to make visible.
ROWS = {
    # "tcg artifact export off-store" — §9.
    "tcg-artifact-export",
    # "endpoint author slots reading raw weight bytes from a directory" —
    # §9, pgw#1303. Priced or deprecated; Paul's ruling is pending, so the
    # row is named and empty rather than absent.
    "author-slot-directory",
}


def _iter_files(roots: Tuple[Path, ...]) -> Iterator[Path]:
    for root in roots:
        if root.is_file():
            yield root
            continue
        for p in sorted(root.rglob("*.py")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            if is_unowned(p) or p.resolve() == SELF:
                continue
            yield p


def _row_of(line: str) -> str:
    _, _, rest = line.partition(MARKER)
    return rest.strip().split()[0] if rest.strip() else ""


def scan(roots: Tuple[Path, ...]) -> List[str]:
    findings: List[str] = []
    for path in _iter_files(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        reaches_tensorfs = bool(_TENSORFS_IMPORT.search(text))
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        rel_key = rel.as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if RETIRED.search(line) and rel_key not in RETIRED_RESIDUE:
                findings.append(
                    f"{rel}:{lineno}: whole-tree materialization is retired "
                    f"(pgw#1295/#1296a, tensorfs#58) and this file is not in "
                    f"the named residue: {stripped}"
                )
                continue
            if not reaches_tensorfs or not HATCH.search(line):
                continue
            if MARKER not in line:
                findings.append(
                    f"{rel}:{lineno}: unnamed materialization hatch — add "
                    f"`{MARKER} <row>` naming its §9 row: {stripped}"
                )
                continue
            row = _row_of(line)
            if row not in ROWS:
                findings.append(
                    f"{rel}:{lineno}: hatch row {row!r} is not in "
                    f"docs/mixed-cas-layout.md §9 (known: {sorted(ROWS)}): {stripped}"
                )
    return findings


def _selftest() -> int:
    """RED on a reintroduced copy and on an unnamed hatch; GREEN on a named row."""

    cases: Tuple[Tuple[str, str, int], ...] = (
        (
            "retired.py",
            "from gen_worker._vendor.tensorfs import LocalCAS\n"
            "cas." "materialize" "_repository(manifest, target)\n",
            1,
        ),
        (
            "unnamed.py",
            "from gen_worker._vendor.tensorfs import open_tensors\n"
            'reader.extract("config.json", dest)\n',
            1,
        ),
        (
            "unknown_row.py",
            "from gen_worker._vendor.tensorfs import open_tensors\n"
            'reader.extract("w.safetensors", dest)  '
            f"{MARKER} because-i-said-so\n",
            1,
        ),
        (
            "named.py",
            "from gen_worker._vendor.tensorfs import open_tensors\n"
            'reader.extract("artifact.so", dest)  '
            f"{MARKER} tcg-artifact-export\n",
            0,
        ),
        # The verb is ordinary English. A file that cannot reach a tensor
        # reader cannot reach the hatch, and a broad scan would be noise.
        (
            "unrelated.py",
            "import zipfile\n"
            'zf.extract(name, target)\n',
            0,
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, body, expected in cases:
            (root / name).write_text(body)
            found = scan((root / name,))
            if len(found) != expected:
                print(
                    f"SELFTEST FAILED: {name} expected {expected} finding(s), "
                    f"got {found}",
                    file=sys.stderr,
                )
                return 1
    print(
        "lint_materialization_hatch selftest: red on a retired copy, on an "
        "unnamed hatch and on an unknown row; green on a named §9 row"
    )
    return 0


def main(argv: List[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    roots = tuple(Path(a).resolve() for a in argv) or DEFAULT_ROOTS
    findings = scan(roots)
    if findings:
        print(
            "pgw#1308: the materialization hatch grew. Under mixed-CAS a "
            "resident model occupies disk ONCE — non-tensor files are symlinks "
            "and tensor bytes are read, never copied. Read "
            "docs/mixed-cas-layout.md §9 (in the tensorfs repo) before adding "
            "a row; a call that belongs to an existing row says which one on "
            "its own line.\n",
            file=sys.stderr,
        )
        for f in findings:
            print(f, file=sys.stderr)
        print(f"\n{len(findings)} unpriced materialization(s)", file=sys.stderr)
        return 1
    print(
        "lint_materialization_hatch: no whole-tree materialization, and every "
        "hatch call names a §9 row"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
