#!/usr/bin/env python3
"""pgw#1346: the retired `Family*` SDK spellings do not survive in this tree.

The typed SDK's types were renamed in one cut (Paul, 2026-08-17): the thing an
author declares is a **model**, the class an endpoint annotates is `Model`, and
the package is `gen_worker.model`. `Family`, `FamilyBinding`, `FamilyInstance`,
`GraphFamily`, `FamilyError`, `FamilyRefusal`, `FamilyExport` and the module
path `gen_worker.family` are gone — not aliased, not deprecated. There is no
compatibility shim, so an old spelling is never a working reference: it is a
stale import, or prose that sends the next reader to a module that does not
exist.

WHY A FENCE AND NOT A ONE-TIME SWEEP, per `lint_retired_package_names.py`'s
design: a rename this wide half-lands. The words `family`/`families` remain
CORRECT and load-bearing everywhere else in this tree — `recipe_v1`'s `family`
field, `family_export_v1`, the `gen_worker.families` inference-defaults
registry, every mint request's `"family": "sdxl"` — so a human sweep cannot use
the word as its signal and has to work symbol by symbol. This fence encodes
exactly which spellings died.

WHAT IT LOOKS FOR is a retired SDK SYMBOL, by regex, in the swept files. It is a
TEXT sweep, not an AST one, because prose is a real finding here: a docstring
saying "declare a `GraphFamily`" is as broken as an import of one.

WHAT IT DELIBERATELY DOES NOT LOOK FOR is the bare word `Family`. It survives in
this tree as ordinary prose about model families ("Family keying", "Family
ladders"), which is the wire vocabulary and is NOT renamed. `class Family` is
matched instead — the declaration form, which is unambiguous.

EXCLUSIONS, all structural rather than an allowlist:

* this file, which DEFINES the retired spellings and so must spell them;
* `src/gen_worker/_vendor/` — byte-identical upstream snapshots (`_lint_scope`);
* `CHANGELOG.md` and any `changelog.d/` fragment a cut has already consumed:
  released text is a record of what was true when it was written.

Everything else is recognised by a PROOF AT THE LINE: a `retired-spelling:`
marker with a reason after the colon. A bare marker is not a proof.

Usage:

    python scripts/lint_retired_sdk_spellings.py [PATH ...]
    python scripts/lint_retired_sdk_spellings.py --selftest
"""

from __future__ import annotations

import re
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
    REPO / "changelog.d", REPO / ".github", REPO / "README.md",
    REPO / "pyproject.toml",
)

SUFFIXES = {".py", ".pyi", ".json", ".toml", ".yaml", ".yml", ".md", ".txt", ".cfg"}

#: retired spelling (regex) -> what it is now.
RETIRED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bgen_worker\.family\b"), "gen_worker.model"),
    (re.compile(r"\bgen_worker/family\b"), "gen_worker/model"),
    (re.compile(r"\bclass Family\b"), "class ModelSpec"),
    (re.compile(r"\bGraphFamily\b"), "GraphModelSpec"),
    (re.compile(r"\bFamilyBinding\b"), "Model"),
    (re.compile(r"\bFamilyInstance\b"), "Model"),
    (re.compile(r"\bFamilyError\b"), "ModelError"),
    (re.compile(r"\bFamilyRefusal\b"), "ModelRefusal"),
    (re.compile(r"\bFamilyExport\b"), "ModelExport"),
    (re.compile(r"\bbind_families\b"), "bind_models"),
    (re.compile(r"\bfake_families\b"), "fake_models"),
    (re.compile(r"\bbound_families\b"), "bound_models"),
    (re.compile(r"\bexport_family\b"), "export_model"),
    (re.compile(r"\bmint_family\b"), "mint_model"),
    (re.compile(r"\bcheck_family_bindings\b"), "check_model_bindings"),
    (re.compile(r"gen-worker family\b"), "gen-worker model"),
]

#: NOT swept yet, deliberately: the decorator kwarg `families=`. It becomes
#: `models=` in the same cut that DELETES the Slot-typed `models=`, because two
#: meanings of one kwarg is the one thing Paul's hardcut ruling forbids
#: (pgw#1346). That cut is blocked on the worker-side injection path — nothing
#: in `src/` calls `bind_models`/`resolver_instances`/`set_instance_resolver`
#: today, so a declared model cannot yet reach a handler. Add
#: `(re.compile(r"\bfamilies\s*=\s*[{(]"), "models={...}")` to RETIRED in the
#: PR that lands it; the sweep is already correct, it just has nothing to
#: refuse until then.

#: The marker that turns a finding into a classified exemption. It must carry a
#: reason after the colon — a bare marker is not a proof.
MARKER = "retired-spelling:"

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
    """A `retired-spelling:` marker with a non-empty reason after the colon."""
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
            for pattern, replacement in RETIRED:
                match = pattern.search(line)
                if match is None:
                    continue
                if _classified(lines, index):
                    continue
                findings.append((path, index + 1, match.group(0), replacement))
                break
    return findings


def _selftest() -> int:
    """A sweep that only ever prints "clean" guards nothing."""
    cases = [
        ("from gen_worker.family.runtime import FamilyBinding\n", True),
        ("from gen_worker.model.runtime import Model\n", False),
        ("class Family:\n", True),
        ("class ModelSpec:\n", False),
        ('"""Declare a GraphFamily when the composition has graph specializations."""\n', True),
        ('"""Declare a GraphModelSpec when the composition has graph specializations."""\n', False),
        ("@endpoint(models={'flux': Flux1Dev})\n", False),
        ("raise FamilyError(FamilyRefusal.TUNED_INVALID, msg)\n", True),
        ("raise ModelError(ModelRefusal.TUNED_INVALID, msg)\n", False),
        # the wire vocabulary is NOT retired and must stay clean
        ('meta = {"family": "sdxl"}\n', False),
        ("from gen_worker.families import register_family\n", False),
        ('EXPORT_VERSION = "family_export_v1"\n', False),
        ("gen-worker families export-schemas out/\n", False),
        ("Family keying: caches key on the graph digest.\n", False),
        # classified: a reason at the line
        ('DEAD = ("FamilyBinding",)  # retired-spelling: this fence names the '
         "dead symbols to refuse them\n", False),
        # a BARE marker is not a proof
        ('DEAD = ("FamilyBinding",)  # retired-spelling:\n', True),
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
        print(f"no retired SDK spelling survives ({len(RETIRED)} spellings swept)")
        return 0
    problems = []
    for path, lineno, spelling, replacement in findings:
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        problems.append(
            f"{rel}:{lineno}: `{spelling}` is retired (pgw#1346) — it is "
            f"`{replacement}` now, and there is no alias. Respell it, or "
            f"classify the line with `{MARKER} <reason>` if it must name the "
            f"dead spelling (a fence's own denylist, a historical rationale)."
        )
    _lint_side.report(problems, "pgw#1346 retired SDK spellings")
    print(f"{len(findings)} retired SDK spelling(s) survive", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
