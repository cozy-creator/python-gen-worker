#!/usr/bin/env python
"""pgw#966 guard: every skip the suite produces is CLASSIFIED, and the rows CI
is supposed to run are not among them.

**Why a guard and not a one-time cleanup.** A skipped row and a passing row look
identical in a green run, so a row that stops running is invisible for as long
as nobody happens to read the reason lines. pgw#858 went that way: its container
row skips on `ubuntu-latest` and has therefore only ever executed on developer
boxes, which is how the pgw#956 flake reached `dev` behind a green CI. A
one-time census would have found it once; this makes the NEXT one fail a build.

Input is whatever `--skip-census-out=` produced (see the repo-root conftest.py):
the skips a session ACTUALLY took, on the machine that took them — not a static
read of the marks, which cannot tell a `skipif` that fires from one that does
not.

Two rules, both cheap:

  1. Every observed key must appear in `scripts/skip_census.txt`. A new skip is
     a decision; making it costs one line naming which of the four dispositions
     it is.
  2. In `--ci` mode a key classified `CI-MUST-RUN` must NOT be observed. That is
     the pgw#858 shape stated positively: this row is expected to execute on the
     runner, so seeing it skip there is a failure and not a note.

Dispositions:

  CI-MUST-RUN   Runs on `ubuntu-latest`. Seeing it skipped in CI FAILS (rule 2).
  CI-SKIPPED    Legitimately un-runnable on the runner (no CUDA device, no
                multi-GPU fabric, missing optional host capability). It must say
                where it IS run, or that it is unverified.
  LOCAL-ONLY    Opt-in by design everywhere, CI included: benchmarks, rows keyed
                to a banked local artifact, rows behind an explicit env switch.
  STAGED        Blocked on work that has not landed, naming the issue that
                unblocks it.

Usage:
    uv run python scripts/lint_skip_census.py [--ci] <census.json> [<census.json> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
CENSUS = REPO / "scripts" / "skip_census.txt"

CLASSES = ("CI-MUST-RUN", "CI-SKIPPED", "LOCAL-ONLY", "STAGED")


def load_census() -> Tuple[Dict[str, str], List[str]]:
    """``{key: disposition}`` plus any malformed lines."""
    entries: Dict[str, str] = {}
    bad: List[str] = []
    for raw in CENSUS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, _, key = line.partition(" ")
        key = key.strip()
        if head not in CLASSES or not key:
            bad.append(raw)
            continue
        entries[key] = head
    return entries, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true",
                    help="enforce rule 2 (CI-MUST-RUN rows may not be skipped)")
    ap.add_argument("census", nargs="+", type=Path,
                    help="JSON written by pytest --skip-census-out")
    args = ap.parse_args()

    classified, malformed = load_census()

    observed: Dict[str, dict] = {}
    for path in args.census:
        if not path.exists():
            print(f"MISSING census input: {path} — the pytest step that writes "
                  "it did not run, so this guard has nothing to check", file=sys.stderr)
            return 2
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            prev = observed.get(row["key"])
            if prev is None:
                observed[row["key"]] = row
            else:
                prev["count"] += row["count"]

    unclassified = sorted(k for k in observed if k not in classified)
    must_run_but_skipped = sorted(
        k for k in observed if classified.get(k) == "CI-MUST-RUN")

    # Always print the census, green or red: the whole point is that a reader of
    # a passing run can see what it did not run.
    by_class: Dict[str, List[str]] = {c: [] for c in CLASSES}
    for key in sorted(observed):
        by_class.setdefault(classified.get(key, "UNCLASSIFIED"), []).append(key)
    total = sum(r["count"] for r in observed.values())
    print(f"pgw#966 skip census — {len(observed)} distinct skip reasons, "
          f"{total} skipped rows")
    for cls in list(CLASSES) + ["UNCLASSIFIED"]:
        keys = by_class.get(cls, [])
        if not keys:
            continue
        print(f"  {cls}: {len(keys)}")
        for key in keys:
            print(f"    - {key}  (x{observed[key]['count']})")
    never_seen = sorted(k for k in classified if k not in observed)
    if never_seen:
        print(f"  not skipped in this run: {len(never_seen)} classified key(s) "
              "— expected, CI and a developer box skip different things")

    failed = False
    if malformed:
        failed = True
        print(f"\nFAIL: {len(malformed)} malformed line(s) in {CENSUS.name} "
              f"(expected `<{'|'.join(CLASSES)}> <key>`):", file=sys.stderr)
        for line in malformed:
            print(f"  {line}", file=sys.stderr)
    if unclassified:
        failed = True
        print(f"\nFAIL: {len(unclassified)} skip(s) with no disposition in "
              f"scripts/{CENSUS.name}.", file=sys.stderr)
        print("A skip is a decision about where a property is measured. Add one "
              "line per key naming which disposition it is:", file=sys.stderr)
        for key in unclassified:
            print(f"  CI-SKIPPED {key}", file=sys.stderr)
            print(f"      reason: {observed[key]['reason'][:200]}", file=sys.stderr)
            print(f"      e.g.:   {observed[key]['examples'][0]}", file=sys.stderr)
    if args.ci and must_run_but_skipped:
        failed = True
        print(f"\nFAIL: {len(must_run_but_skipped)} row(s) classified CI-MUST-RUN "
              "did NOT run on the runner:", file=sys.stderr)
        for key in must_run_but_skipped:
            print(f"  {key}", file=sys.stderr)
            print(f"      reason: {observed[key]['reason'][:300]}", file=sys.stderr)
        print("These are the pgw#858 shape: coverage that a green run reports as "
              "present and did not execute. Fix the environment mismatch, or "
              "re-classify with the evidence that it cannot be fixed.",
              file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
