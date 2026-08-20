#!/usr/bin/env python3
"""pgw#1547: the word `cell` is retired for compiled graphs, and stays retired.

Paul, 2026-08-20: *"we stopped using the term 'cell' a while ago. They are now
'graphs' and 'graph specializations'."* A target has ONE graph and MANY graph
SPECIALIZATIONS (tcg#56); the durable artifact is a COMPILED GRAPH keyed by a
`ck1-` cg-key. Nothing in this vocabulary is a "cell".

WHY A FENCE AND NOT A SWEEP, and this is measured rather than argued. pgw#1363
(2026-08-18) already retired `cell` across the internal surface — `cell_adopt`,
`fleet_cells`, `local_cell_store`, `CellPublisher` and their prose all moved. It
did NOT stick, and it could not have: nothing watched the word. Two days later
`_PROVEN_CELLS`/`_QUARANTINED_CELLS` were still module globals in
`compile_cache.py`, `tests/conftest.py` still cleared `_QUARANTINED_CELLS`
through a `getattr(..., set())` that would have gone SILENT the moment the name
moved, and eight mint fixtures still called a compiled graph a cell. pgw#1363's
own changelog records the same lesson from the other side: a stale
`CELLS_DIRNAME` reference reddened master because only CI could see it. A rename
this wide half-lands unless a gate holds the line — the identical argument
`lint_retired_sdk_spellings.py` and `lint_retired_graph_class_spellings.py`
already make in this directory.

WHAT IT LOOKS FOR is the substring `cell` NOT PRECEDED BY A LETTER, in the swept
files, case-insensitively. It is a TEXT sweep, not an AST one, because prose is
the failure mode: a docstring calling a compiled graph "the cell" sends the next
reader to a vocabulary the fleet does not speak, and no type checker sees it.

WHY THE LOOKBEHIND, and it is load-bearing rather than cosmetic. `cancelled`,
`CancelledError`, `cancellation`, `cancelling` and `raise_if_cancelled` all
contain the letters `cell` — 84 live occurrences in `src/gen_worker` alone,
every one correct. They are excluded STRUCTURALLY by `(?<![A-Za-z])` (the `c` of
`can-cell-ed` is a letter) rather than by an allowlist that would have to grow
every time someone cancels something.

THE OTHER `cell` SENSES, which are DIFFERENT WORDS and are deliberately out of
scope. Measured before this fence was written:

* `cell_contents` / `__closure__` — the *Python* closure-cell API
  (`serving/weightless_program.py` reads it to unwrap a patched function).
  A language feature, allowlisted by exact token below;
* STATISTICS cells — `posture_wire_vectors.json`'s "one cell with two samples",
  "shatters every cell into singletons". An aggregation bucket, not an artifact.
  That corpus is digest-gated and excluded structurally;
* BENCH-MATRIX cells — `scripts/svdq_bench/`'s `--cells mat,corr,bb,bf`, the
  rows of a benchmark table. Same eval sense as the H3 program's "15s cell".

THE DEFERRED SET IS NOW THREE HISTORICAL NOTES, and that is the whole story of
this fence. When it first landed it carried 83 line-proofs for spellings the hub
owned and the fleet had persisted. Paul then ruled the hardcut (2026-08-20:
*"I want it fully renamed. You can go ahead and break everything. No legacy
support, hardcut. We are pre-launch it's fine."*), so pgw#1547's migration
retired all 83: the proto fields moved by NAME with their numbers unchanged
(PROTO_DIGEST records why that keeps the binary wire byte-identical), the
`jit_cell`/`aot_cell` serving modes, the `cell_*` boot phases and the
`cell_read_*` claims moved with the code that emits them, the on-disk layout
became `<TENSORHUB_CACHE_DIR>/compiled-graph-store/graphs/<ck1>/graph.tar.gz`,
and `GEN_WORKER_LOCAL_CELLS_DIR` was DELETED rather than renamed — it configured
a second root for a store the CAS knob already roots.

What is left carries a proof only because it is a RECORD: three lines that name
the deleted env or the exact names of tests pgw#1181 deleted. Respelling any of
them would falsify the record. So the marker's meaning has changed from
"deferred, pending a coordinated lane" to "historical, deliberately", and the
count should stay at three unless something is genuinely being recorded.

EXCLUSIONS, all structural rather than an allowlist of findings:

* this file, which DEFINES the retired spelling and so must spell it;
* `src/gen_worker/_vendor/` — byte-identical upstream snapshots;
* `src/gen_worker/pb/` and `proto/` — generated protobuf and the wire contract
  it is generated from, gated on `PROTO_DIGEST`;
* `src/gen_worker/contracts/` and `tests/testdata/` — the shared conformance
  corpora. These are vendored BYTE-IDENTICALLY into tensorhub and gated by
  `check_*_digest.py`; editing prose here reddens two repos at once and is a
  coordinated proto-lane move, not a rename;
* `CHANGELOG.md` and `changelog.d/` — released text records what was true when
  it was written;
* `scripts/svdq_bench/` — the bench-matrix sense above.

Everything else is recognised by a PROOF AT THE LINE: a `cell-spelling:` marker
with a reason after the colon. A bare marker is not a proof.

Usage:

    python scripts/lint_retired_cell_spellings.py [PATH ...]
    python scripts/lint_retired_cell_spellings.py --selftest
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lint_side  # noqa: E402

#: `cell` not preceded by a letter. The lookbehind is what keeps `cancelled`
#: and its family out — see the module docstring.
PATTERN = re.compile(r"(?<![A-Za-z])cell", re.IGNORECASE)

#: Exact tokens that are a DIFFERENT WORD, not a respelling of this one. Kept
#: deliberately tiny: a growing list here means the pattern is wrong.
ALLOWED_TOKENS = (
    "cell_contents",  # the Python closure-cell API
)

MARKER = "cell-spelling:"

SUCCESSOR = (
    "graph / compiled graph / graph specialization "
    "(the artifact is a COMPILED GRAPH keyed by a ck1- cg-key)"
)

EXCLUDED_PARTS = ("_vendor", "__pycache__", ".git")
EXCLUDED_SUFFIXES = ("lint_retired_cell_spellings.py",)
EXCLUDED_RELATIVE = (
    "src/gen_worker/pb",
    "proto",
    "src/gen_worker/contracts",
    "tests/testdata",
    "CHANGELOG.md",
    "changelog.d",
    "scripts/svdq_bench",
)


def _excluded(path: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return True
    if str(path).endswith(EXCLUDED_SUFFIXES):
        return True
    try:
        rel = path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return False
    return any(rel == e or rel.startswith(e + "/") for e in EXCLUDED_RELATIVE)


def scan_text(text: str, name: str = "") -> list[tuple[int, str]]:
    """Every retired spelling in `text`, minus lines carrying a proof."""
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if MARKER in line and line.split(MARKER, 1)[1].strip():
            continue
        probe = line
        for token in ALLOWED_TOKENS:
            probe = probe.replace(token, "")
        m = PATTERN.search(probe)
        if m:
            start = max(0, m.start() - 12)
            hits.append((lineno, probe[start:m.end() + 20].strip()))
    return hits


def _walk(roots: list[Path]) -> Iterator[Path]:
    for root in roots:
        if root.is_file():
            if not _excluded(root):
                yield root
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {
                    ".py", ".md", ".toml", ".yaml", ".yml", ".json"}:
                if not _excluded(path):
                    yield path


def selftest() -> int:
    """A sweep that can only ever print 'clean' guards nothing."""
    ok = True

    def check(label: str, text: str, want: bool) -> None:
        nonlocal ok
        hits = scan_text(text)
        if bool(hits) != want:
            print(f"SELFTEST FAIL [{label}]: want hits={want}, got {hits}")
            ok = False

    check("retired prose", '"""The cell is armed at boot."""\n', True)
    check("retired plural", "# two cells serve this family\n", True)
    check("retired identifier", "_PROVEN_CELLS = set()\n", True)
    check("retired wire ident", "mode = 'jit_cell'\n", True)
    check("retired dirname", 'DIRNAME = "aot-cells"\n', True)
    check("retired artifact", 'ARTIFACT = "cell.tar.gz"\n', True)
    check("retired env", 'ENV = "GEN_WORKER_LOCAL_CELLS_DIR"\n', True)
    check("retired CamelCase", "class CellLookup: ...\n", True)

    check("current vocabulary",
          '"""The compiled graph is armed at boot; one graph, many graph\n'
          'specializations, keyed by a ck1- cg-key."""\n', False)

    # The `cancel` family is 84 live occurrences in src/gen_worker. If this
    # arm ever goes red the pattern has over-reached and the fix is the
    # PATTERN, never the source.
    check("cancel family",
          "import asyncio\n"
          "from asyncio import CancelledError\n"
          "raise_if_cancelled()\n"
          "task.cancel()\n"
          "self._cancelling = True\n"
          "# cancellation is cooperative here\n"
          "except CancelledError:\n"
          "    was_cancelled = True\n", False)

    check("python closure api",
          "for c in fn.__closure__ or ():\n"
          "    candidate = c.cell_contents\n", False)

    check("proof at the line",
          'ENV = "GEN_WORKER_LOCAL_CELLS_DIR"  '
          "# cell-spelling: cozy-local paths.go pins it\n", False)
    check("bare marker is not a proof",
          'ENV = "GEN_WORKER_LOCAL_CELLS_DIR"  # cell-spelling:\n', True)

    with tempfile.TemporaryDirectory() as tmp:
        excluded = Path(tmp) / "CHANGELOG.md"
        excluded.write_text("the cell was renamed\n")
        if not scan_text(excluded.read_text()):
            print("SELFTEST FAIL: scan_text must not itself apply path exclusions")
            ok = False

    print("selftest: OK" if ok else "selftest: FAILED")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    roots = [Path(a) for a in argv] or [
        REPO / "src" / "gen_worker",
        REPO / "tests",
        REPO / "scripts",
        REPO / "docs",
    ]
    problems: list[str] = []
    findings = 0
    for path in _walk(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, excerpt in scan_text(text, str(path)):
            rel = path.resolve()
            try:
                rel = rel.relative_to(REPO)
            except ValueError:
                pass
            problems.append(f"{rel}:{lineno}: retired spelling in {excerpt!r}")
            findings += 1
    if findings:
        _lint_side.report(problems, "pgw#1547 retired `cell` spellings",
                          stream=sys.stdout)
        print(
            f"\nlint_retired_cell_spellings: {findings} finding(s). The word `cell` is "
            f"retired for compiled graphs — write {SUCCESSOR}.\n"
            f"If a line must keep it (a hub-owned wire spelling, an on-disk name under a "
            f"cross-repo parity contract, or a DIFFERENT sense of the word), prove it at "
            f"the line with `# {MARKER} <reason>`."
        )
        return 1
    print("pgw#1547: clean — no retired `cell` spelling survives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
