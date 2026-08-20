#!/usr/bin/env python3
"""pgw#1297: the retired distribution names do not survive in this tree.

`hashrepo` was renamed to `tensorfs` and `torch-compiled-graphs` (module
`torch_compiled_graphs`) to `torchcg`. Both old PyPI projects are DELETED
permanently, so an old name is never a working reference to anything — it is
either a stale import that will not resolve once `_vendor/` goes away
(pgw#1295), or documentation that sends the next reader to a repo path that no
longer exists.

WHY A FENCE AND NOT A ONE-TIME SWEEP. This rename has now been attempted twice
and half-landed both times: pgw#1310 repointed every IMPORT but left the
vendored package directory spelled `torch_compiled_graphs`, and the prose in a
dozen docstrings kept naming a module nobody can import. A grep that is run
once is a claim about one afternoon.

WHAT IT LOOKS FOR is the literal spelling of a retired name, anywhere in the
swept files. Deliberately a TEXT sweep and not an AST one, unlike
`lint_tcg_vocabulary.py`: here prose IS the finding. A docstring that says
"see torch_compiled_graphs.identity" is exactly the defect — the reader follows
it and lands nowhere.

THREE EXCLUSIONS, all structural rather than an allowlist:

* this file, which DEFINES the retired spellings and so must spell them;
* `src/gen_worker/_vendor/` — byte-identical upstream snapshots held to a digest
  fence (`scripts/_lint_scope.py`, pgw#1310). The vendored `torchcg` really does
  carry `torch-compiled-graphs` strings, including the on-disk ref namespace
  `torch-compiled-graphs/v1`, which must NEVER be respelled: it would orphan
  every stored compiled graph. Fixed upstream and re-vendored, never here.
* released changelog text is not swept, because it is a record of what was true
  when it was written: `CHANGELOG.md` is out of scope entirely, and a
  `changelog.d/` fragment is skipped once `changelog.d/consumed.tsv` says a cut
  assembled it — editing it then would desynchronise it from the published
  section it produced. A fragment still PENDING is swept: it describes the tree
  as it is about to ship.

Everything else is recognised by a PROOF AT THE LINE, per
`lint_repo_ref_pins.py`'s design: a `retired-name:` marker with a reason after
the colon. A filename would let a real stale import hide behind it; a reason
cannot, because deleting the reason turns the line red again.

Usage:

    python scripts/lint_retired_package_names.py [PATH ...]
    python scripts/lint_retired_package_names.py --selftest

Defaults to `src/`, `tests/`, `tests_v2/`, `scripts/`, `examples/`, `docs/`,
`benchmarks/`, `changelog.d/` and `.github/`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_scope import is_unowned  # noqa: E402
import _lint_side  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

DEFAULT_ROOTS = (
    REPO / "src", REPO / "tests", REPO / "tests_v2", REPO / "scripts",
    REPO / "examples", REPO / "docs", REPO / "benchmarks",
    REPO / "changelog.d", REPO / ".github", REPO / "pyproject.toml",
)

SUFFIXES = {".py", ".pyi", ".json", ".toml", ".yaml", ".yml", ".md", ".txt", ".cfg"}

#: retired spelling -> what it is now.
RETIRED: dict[str, str] = {
    "torch_compiled_graphs": "torchcg",
    "torch-compiled-graphs": "torchcg",
    "hashrepo": "tensorfs",
}

#: The marker that turns a finding into a classified exemption. It must carry a
#: reason after the colon — a bare marker is not a proof.
MARKER = "retired-name:"

CONSUMED_TSV = REPO / "changelog.d" / "consumed.tsv"


def _consumed_fragments() -> frozenset[Path]:
    """Changelog fragments a release cut has already assembled — released text."""
    try:
        rows = CONSUMED_TSV.read_text().splitlines()
    except OSError:
        return frozenset()
    stems = (
        row.split("\t")[1].strip()
        for row in rows
        if row.strip() and not row.startswith("#") and "\t" in row
    )
    return frozenset(CONSUMED_TSV.parent / f"{stem}.md" for stem in stems)


def _files(roots: Tuple[Path, ...]) -> Iterator[Path]:
    consumed = _consumed_fragments()
    for root in roots:
        if root.is_file():
            yield root
            continue
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            if "__pycache__" in path.parts or ".venv" in path.parts:
                continue
            if path.resolve() == SELF or path in consumed:
                continue
            if path.is_relative_to(REPO) and is_unowned(path, REPO):
                continue
            yield path


def _classified(lines: List[str], index: int) -> bool:
    """A `retired-name:` marker with a non-empty reason after the colon.

    Accepted on the finding's own line or in the comment block IMMEDIATELY
    above it, because the legitimate cases are multi-sentence rationales that
    do not fit on the line they exempt. Still a proof at the site: delete the
    reason and the line goes red again.
    """
    for line in (lines[index], lines[index - 1] if index else ""):
        _, sep, reason = line.partition(MARKER)
        if sep and reason.strip():
            return True
    return False


def scan(roots: Tuple[Path, ...]) -> List[Tuple[Path, int, str, str]]:
    findings: List[Tuple[Path, int, str, str]] = []
    for path in _files(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines):
            # Case-insensitive: the prose spelling `HashRepo` is the same stale
            # pointer as the import spelling, and it is the form a docstring
            # actually reaches for. A case-sensitive sweep of this tree found
            # 5 survivors; this one found 48.
            lowered = line.lower()
            for retired, replacement in RETIRED.items():
                if retired not in lowered:
                    continue
                if _classified(lines, index):
                    continue
                findings.append((path, index + 1, retired, replacement))
                break
    return findings


def _selftest() -> int:
    """A sweep that only ever prints "clean" guards nothing."""
    cases = [
        ("from gen_worker._vendor.torch_compiled_graphs import ARTIFACT_KIND\n", True),
        ('"""See :mod:`torch_compiled_graphs.identity` for the axes."""\n', True),
        ('DEPS = ["hashrepo>=0.3,<0.4"]\n', True),
        ("# torch-compiled-graphs owns compiler policy\n", True),
        ("from gen_worker._vendor.torchcg import ARTIFACT_KIND\n", False),
        ("from gen_worker._vendor.tensorfs import LocalCAS\n", False),
        # classified: a reason at the line
        ('DEAD = {"hashrepo"}  # retired-name: this fence must name the dead '
         "projects to refuse them\n", False),
        # a BARE marker is not a proof
        ('DEAD = {"hashrepo"}  # retired-name:\n', True),
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        for i, (body, should_flag) in enumerate(cases):
            path = root / f"case{i}.py"
            path.write_text(body)
            flagged = bool(scan((root,)))
            if flagged != should_flag:
                failures += 1
                print(
                    f"SELFTEST FAIL case{i}: expected "
                    f"{'a finding' if should_flag else 'clean'}, got the other\n"
                    f"  {body.strip()}",
                    file=sys.stderr,
                )
            path.unlink()
    if failures:
        print(f"{failures} selftest case(s) failed", file=sys.stderr)
        return 1
    print(f"selftest OK ({len(cases)} cases)")
    return 0


def main(argv: List[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    roots = tuple(Path(a).resolve() for a in argv) or DEFAULT_ROOTS
    findings = scan(roots)
    if not findings:
        print(f"no retired package name survives ({len(RETIRED)} spellings swept)")
        return 0
    problems = []
    for path, lineno, retired, replacement in findings:
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        problems.append(
            f"{rel}:{lineno}: `{retired}` is retired — it is `{replacement}` now, "
            f"and the old PyPI project is permanently deleted. Respell it, or "
            f"classify the line with `{MARKER} <reason>` if it must name the "
            f"dead project (a fence's own denylist, a historical rationale)."
        )
    _lint_side.report(problems, "pgw#1297 retired package names")
    print(f"{len(findings)} retired package name(s) survive", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
