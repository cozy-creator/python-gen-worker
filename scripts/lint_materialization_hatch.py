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

**`materialize()` survives as TIER 3 of the access ladder** — Paul's #1303
ruling, 2026-08-17. The ladder is: (1) symlink snapshot + native tensorfs
reads, (2) symlink snapshot + FUSE where FUSE exists (not on a pod), (3) this
hatch. It is PERMANENT and it is NOT separately priced; it is simply the
least-preferred tier, and this census is how the migration up the ladder is
read. `docs/mixed-cas-layout.md` §9 lists the whole user set with a tier each,
and that list — not a size limit — is the control. So a call is permitted only
when the line NAMES the audit row it belongs to, and the row is one §9
actually has. A new row is a decision someone makes on purpose, in the design
doc, not a call site that appeared in a diff.

**`extract()` is the RETIRED SPELLING and is refused.** The same ruling renamed
it (tensorfs#107), so there is ONE name upstream and here. A fence that quietly
accepted both would let the old name drift back the moment someone copies an
old snippet, and a spelling problem in this fence reads as a census of ZERO
while a live materialization runs unnoticed. Refused, not ignored, so the
failure is a message about the rename instead of a mystery.

WHY A LINT AND NOT A TEST. The hatch's whole failure mode is a site nobody
looked at: every individual call works perfectly and the suite stays green
while the architecture erodes underneath it. Only a census notices, and it
belongs in `fast gates` where it runs on the merge path.

WHY THE HATCH SCAN IS FILE-SCOPED. `materialize` and `extract` are ordinary
English verbs and this tree has unrelated ones (`guard_closure`'s,
`zipfile`'s). The hatch is reachable only through a tensorfs reader, so the
scan considers a file only when it imports tensorfs at all. That is precise
rather than broad: a file with no tensorfs import cannot reach it.

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

#: A line that DEFINES the retired symbol rather than reaching for it. Only
#: the vendored storage snapshot may do that: it is a byte-identical copy of
#: an upstream rev, fixed upstream and re-vendored, never patched here.
RETIRED_DEFINITION = re.compile(r"^(?:async\s+)?def\s+materialize" r"_repository\b")

#: The residue, NAMED. Each entry is a path that may still mention the retired
#: symbol, with the issue that deletes it. Not an allowlist for new debt: a
#: path not on this list is refused, and the list only ever shrinks.
#:
#: **IT IS EMPTY**, as of pgw#1308 step ⑥. The chokepoint projects
#: (`models/cozy_snapshot.py:_publish_snapshot`) and its test scaffolding went
#: with it. Nothing in this repo, first-party or vendored, CALLS whole-tree
#: materialization; the only surviving trace is the definition in the pinned
#: storage snapshot, declared below.
RETIRED_RESIDUE: set[str] = set()

#: Where the retired symbol may still be DEFINED, and why. Declared rather
#: than merely unseen: `_lint_scope` excludes `_vendor` from every other
#: guard, so without this line the census would read "gone" when the code is
#: still shipped in the wheel with zero callers. It leaves when the vendored
#: storage rev moves, which VENDORED.toml explains is NOT gated on this issue.
RETIRED_DEFINED_IN = {
    "src/gen_worker/_vendor/tensorfs/local.py",
}

#: The hatch itself, scanned only where tensorfs is imported.
#:
#: ONE SPELLING, as of the #1303 ruling (tensorfs#107 renamed upstream's
#: `extract()` to `materialize()`): the guard spells the one symbol its own
#: vendored snapshot exports. Two names is how a fence matches NOTHING in the
#: code it guards and reads a census of zero while a live materialization runs.
HATCH = re.compile(r"\.materialize\s*\(")

#: The RETIRED spelling. Refused, not ignored -- see the module docstring.
RETIRED_HATCH = re.compile(r"\.extract\s*\(")

#: `def materialize(...)` is a definition, not a hatch call.
HATCH_DEFINITION = re.compile(r"^(?:async\s+)?def\s+(?:extract|materialize)\b")

#: Vendored files that legitimately call the hatch, with their §9 row. A
#: vendored call cannot carry an inline marker (the snapshot is byte-identical
#: to upstream), so it is declared HERE instead of being invisible. A vendored
#: hatch call in an undeclared file is refused, and a declared file that no
#: longer contains one is refused too -- so the table cannot go stale in
#: either direction.
#
# EMPTY, and that is a result rather than an omission. pgw#1342 bumped the
# vendored torchcg to a rev whose store keeps artifacts as whole blobs, so
# `resolve` untars straight out of the store and only `export_artifact` — a
# plain `copyfile` to a destination the store does not own — writes bytes
# twice. The vendored snapshot no longer reaches the hatch at all, and the
# check below refuses a declaration that outlives its last call. The one live
# §9 `tcg-artifact-export` caller is first-party and carries an inline marker
# (`aot_delivery.py`).
VENDORED_HATCH: dict = {}

#: A file reaches the hatch only if it can reach a tensor reader.
_TENSORFS_IMPORT = re.compile(r"\b(?:from|import)\s+\S*tensorfs\b")

#: The line-level statement of WHICH §9 row this call is.
MARKER = "# mixed-cas-hatch:"

#: Every row `docs/mixed-cas-layout.md` §9 admits. Adding one here without
#: adding it there is the mistake this list exists to make visible.
ROWS = {
    # "tcg artifact export off-store" — §9, tier 3.
    "tcg-artifact-export",
    # "endpoint author slots reading raw weight bytes from a directory" —
    # §9, pgw#1303. RULED 2026-08-17: tier 3, permanent, unpriced. Our own
    # loaders that pass through this row are migration debt with a tier-1
    # target; author code that demands a directory stays here.
    "author-slot-directory",
}


def _iter_files(roots: Tuple[Path, ...]) -> Iterator[Tuple[Path, bool]]:
    """Every scannable file, with whether a guard may JUDGE it.

    `_vendor` is scanned rather than skipped, unlike every other guard here.
    That exclusion is right for architecture measurements -- vendored code is
    not evidence about ours -- and wrong for a CENSUS, whose whole job is to
    say what is present. So vendored files are read, and what they may contain
    is decided by the declaration tables above instead of by an inline marker
    they cannot carry.
    """

    for root in roots:
        if root.is_file():
            yield root, is_unowned(root)
            continue
        for p in sorted(root.rglob("*.py")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            if p.resolve() == SELF:
                continue
            yield p, is_unowned(p)


def _row_of(line: str) -> str:
    _, _, rest = line.partition(MARKER)
    return rest.strip().split()[0] if rest.strip() else ""


def scan(roots: Tuple[Path, ...]) -> List[str]:
    findings: List[str] = []
    vendored_hatch_seen: set[str] = set()
    for path, vendored in _iter_files(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        reaches_tensorfs = bool(_TENSORFS_IMPORT.search(text)) or vendored
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        rel_key = rel.as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if RETIRED.search(line):
                if RETIRED_DEFINITION.match(stripped):
                    if rel_key not in RETIRED_DEFINED_IN:
                        findings.append(
                            f"{rel}:{lineno}: whole-tree materialization is "
                            f"retired (pgw#1296a, tensorfs#58) and this file "
                            f"is not the pinned upstream snapshot that still "
                            f"defines it: {stripped}"
                        )
                    continue
                if rel_key not in RETIRED_RESIDUE:
                    findings.append(
                        f"{rel}:{lineno}: whole-tree materialization is retired "
                        f"(pgw#1296a, tensorfs#58) and this file is not in "
                        f"the named residue: {stripped}"
                    )
                    continue
            if not reaches_tensorfs:
                continue
            if RETIRED_HATCH.search(line) and not HATCH_DEFINITION.match(stripped):
                findings.append(
                    f"{rel}:{lineno}: `.extract(` is the RETIRED spelling of "
                    f"the hatch — it is `.materialize(` since tensorfs#107 "
                    f"(Paul's #1303 ruling). One name, no aliases: {stripped}"
                )
                continue
            if not HATCH.search(line):
                continue
            if HATCH_DEFINITION.match(stripped):
                continue
            if vendored:
                # A vendored line cannot carry a marker without breaking the
                # digest fence, so its row is declared in the table above.
                row = VENDORED_HATCH.get(rel_key, "")
                if not row:
                    findings.append(
                        f"{rel}:{lineno}: a vendored snapshot reaches the "
                        f"single-file materialization hatch and no §9 row is "
                        f"declared for it in VENDORED_HATCH: {stripped}"
                    )
                elif row not in ROWS:
                    findings.append(
                        f"{rel}:{lineno}: declared hatch row {row!r} is not in "
                        f"docs/mixed-cas-layout.md §9: {stripped}"
                    )
                else:
                    vendored_hatch_seen.add(rel_key)
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
    # A declaration that no longer describes anything is a census that has
    # stopped being read. It goes when its last call goes, not later.
    if any(root == REPO / "src" for root in roots):
        for declared in sorted(set(VENDORED_HATCH) - vendored_hatch_seen):
            findings.append(
                f"{declared}: declared in VENDORED_HATCH but no longer calls "
                f"the hatch — delete the declaration, the row is now empty"
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
            "from gen_worker._vendor.tensorfs import LocalCAS\n"
            "cas.materialize(entry, destination)\n",
            1,
        ),
        (
            "named.py",
            "from gen_worker._vendor.tensorfs import LocalCAS\n"
            "cas.materialize(entry, destination)  "
            f"{MARKER} tcg-artifact-export\n",
            0,
        ),
        # The RETIRED spelling, refused on sight -- and refused EVEN WITH a
        # valid row marker, because the finding is about the name, not the
        # census. `extract()` was upstream's name until tensorfs#107 renamed
        # it; accepting both is how an old snippet walks back in.
        (
            "retired_spelling.py",
            "from gen_worker._vendor.tensorfs import open_tensors\n"
            'reader.extract("config.json", dest)\n',
            1,
        ),
        (
            "retired_spelling_named.py",
            "from gen_worker._vendor.tensorfs import open_tensors\n"
            'reader.extract("artifact.so", dest)  '
            f"{MARKER} tcg-artifact-export\n",
            1,
        ),
        # Defining a helper called `materialize` is not reaching for the hatch.
        (
            "defines_one.py",
            "from gen_worker._vendor.tensorfs import read_entry\n"
            "def materialize(base, fixture):\n"
            "    return read_entry(fixture.cas, entry)\n",
            0,
        ),
        (
            "unknown_row.py",
            "from gen_worker._vendor.tensorfs import open_tensors\n"
            'reader.materialize("w.safetensors", dest)  '
            f"{MARKER} because-i-said-so\n",
            1,
        ),
        # The verbs are ordinary English. A file that cannot reach a tensorfs
        # reader cannot reach the hatch, and a broad scan would be noise.
        (
            "unrelated.py",
            "import zipfile\n"
            'zf.extract(name, target)\n',
            0,
        ),
        (
            "unrelated_materialize.py",
            "import pandas\n"
            "frame.materialize(rows)\n",
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
    # The `def` of the retired symbol is permitted ONLY in the pinned storage
    # snapshot, and refused anywhere else -- otherwise "define it here and
    # call it" walks straight through the definition exemption.
    with tempfile.TemporaryDirectory() as tmp:
        planted = Path(tmp) / "redefines.py"
        planted.write_text(
            "from gen_worker._vendor.tensorfs import LocalCAS\n"
            "def " + "materialize" + "_repository(manifest, target):\n"
            "    ...\n"
        )
        if len(scan((planted,))) != 1:
            print(
                "SELFTEST FAILED: a first-party redefinition of the retired "
                "symbol was not refused",
                file=sys.stderr,
            )
            return 1
    print(
        "lint_materialization_hatch selftest: red on a whole-tree copy, on a "
        "first-party redefinition, on an unnamed hatch, on an unknown row, and "
        "on the RETIRED `.extract(` spelling with or without a valid marker; "
        "green on a named §9 row, on a plain definition, and on the ordinary "
        "English verbs in a file that cannot reach tensorfs"
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
            "its own line. The hatch is spelled `materialize()` — `extract()` "
            "is the retired name (tensorfs#107).\n",
            file=sys.stderr,
        )
        for f in findings:
            print(f, file=sys.stderr)
        print(f"\n{len(findings)} undeclared materialization(s)", file=sys.stderr)
        return 1
    print(
        "lint_materialization_hatch: no whole-tree materialization, no retired "
        "spelling, and every hatch call names a §9 row"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
