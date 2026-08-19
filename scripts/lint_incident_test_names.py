#!/usr/bin/env python3
"""pgw#1362 / DESIGN-RULINGS 4.34b: a NEW test module may not be named for an issue.

WHY THIS IS A GATE AND NOT A CONVENTION — the number that bought it. Across five
hours on 2026-08-17/18 the pgw#1362 consolidation landed three waves, removing 41
incident-named modules and ~1,900 lines. In the same window `origin/master`'s test
tree went from **441 files / 134,979 lines to 411 / 135,818** — thirty files fewer
and **839 lines MORE**, because other lanes added ~2,700 lines of new issue-named
tests. All of it correct, ruled work. **Nobody did anything wrong and the corpus
still grew.** That is Paul's asymmetry (*"agents tend to keep adding more
complexity, and never remove any of it"*), and it means a cleanup epic cannot
outrun its own intake. Without this file pgw#1362 has to be re-run every quarter.

WHAT IT REFUSES: a top-level `tests/test_*.py` or `tests_v2/test_*.py` whose
FILENAME carries an issue id (`pgw1234`, `gw640`, `th1130`, `cl72`, `ie655`,
`te148`) and is not in the grandfather baseline. A filename says WHEN a test was
written; the tree needs it to say WHAT the test exercises.

THE INCIDENT IS NOT LOST — it moves. The lineage lives as a one-line comment on
the test (`# pgw#1234: <one clause>`) and the full story lives in the tracker
issue. That is the trade DESIGN-RULINGS 4.34b makes: the ledger keeps the
narrative, the file name keeps the subject.

THE BASELINE ONLY SHRINKS, and the mechanism is structural rather than a promise
(the `lint_mypy_ratchet.py` shape). Two rails:

  1. a baseline entry that no longer exists on disk is a HARD FAILURE — so a lane
     that folds a module MUST delete its line, and the list cannot rot into a
     collection of names nobody can check;
  2. `BASELINE_HIGH_WATER` below caps the length. Lowering it is the burn-down
     and lands with the names you deleted; raising it is the thing this gate
     exists to refuse.

Together those make padding impossible: you cannot add a line without raising the
cap, and you cannot raise the cap without saying so in the diff.

SCOPE, and it is deliberate: TOP-LEVEL `tests/` and `tests_v2/` only. `tests/convert/`
is already a domain subdirectory — the shape the flat tree is being moved toward —
so an issue-named file inside it is a smaller problem, and folding those is a later
wave. `tests/harness/` holds rigs, not tests. Widen this when the flat tree is done,
not before: a gate that fires on things nobody is fixing yet trains lanes to ignore it.

NO ESCAPE-HATCH MARKER, deliberately. Every other fence in this tree admits a
`# <marker>: <reason>` proof at the line, because those fences sweep CONTENT and
content sometimes legitimately names the forbidden thing. This one sweeps a
FILENAME, and the fix is `git mv`. A naming rule with an opt-out is a naming
suggestion, which is what the tree already had.

Usage:

    python scripts/lint_incident_test_names.py
    python scripts/lint_incident_test_names.py --selftest
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO / "scripts" / "incident_named_tests.txt"
ROOTS: Tuple[str, ...] = ("tests", "tests_v2")

#: Length of the grandfather list. **Lower it with the names you deleted; never
#: raise it.** 375 at adoption (pgw#1362, after waves 1-3a folded 41 modules out
#: of the 416 that were issue-named when the epic's census was taken).
BASELINE_HIGH_WATER = 126  # 370 -> 128: pgw#1373 deleted the v1 SDK's test corpus;
                           # 128 -> 126: pgw#1438 deleted two more whose subjects went with it

#: The issue-id spellings this workspace uses, as they appear inside a filename
#: (`_pgw1234`, `_th1130`, `_gw640`, `_cl72`, `_ie655`, `_te148`). Matched
#: anywhere in the stem because the corpus puts them in both positions
#: (`test_th1111_stage_timing`, `test_receipts_pgw709`).
ISSUE_RE = re.compile(r"(?:pgw|gw|th|cl|ie|te)\d+")

#: Domain modules that already exist, named in the refusal so the author has
#: somewhere to put the test instead of a rule to argue with.
_EXAMPLES = (
    "test_wire_protocol.py, test_hub_transport.py, test_receipts_trust.py, "
    "test_telemetry.py, test_media_transport.py, test_child_faults.py, "
    "test_config_authority.py, test_config_boundary.py, test_sdk_authoring.py"
)


def load_baseline(path: Path = BASELINE_PATH) -> List[str]:
    rows: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(line)
    return rows


def issue_named(stem: str) -> bool:
    return bool(ISSUE_RE.search(stem))


def scan(repo: Path, baseline: Sequence[str]) -> Tuple[List[str], List[str]]:
    """``(new_issue_named, stale_baseline_entries)``."""
    known = set(baseline)
    new: List[str] = []
    for root in ROOTS:
        directory = repo / root
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if not (entry.is_file() and entry.name.startswith("test_")
                    and entry.suffix == ".py"):
                continue
            rel = f"{root}/{entry.name}"
            if issue_named(entry.stem) and rel not in known:
                new.append(rel)
    stale = [row for row in baseline if not (repo / row).exists()]
    return new, stale


def check(repo: Path = REPO, baseline_path: Path = BASELINE_PATH,
          high_water: int = BASELINE_HIGH_WATER) -> List[str]:
    baseline = load_baseline(baseline_path)
    new, stale = scan(repo, baseline)
    problems: List[str] = []

    for rel in new:
        problems.append(
            f"{rel}: a NEW test module named for an ISSUE. The file name says "
            f"WHEN this was written, not WHAT it exercises (DESIGN-RULINGS "
            f"4.34b). Put these tests in the domain module for their subject "
            f"— e.g. {_EXAMPLES} — or add a new domain module named for its "
            f"subject, and keep the lineage as a one-line comment on the test: "
            f"`# pgw#1234: <one clause>`. The full story belongs in the tracker "
            f"issue, which is where anyone looking for it will actually go."
        )

    for rel in stale:
        problems.append(
            f"{rel}: grandfathered in {baseline_path.name} but no longer on "
            f"disk. Folding or deleting a module means deleting its line here "
            f"and lowering BASELINE_HIGH_WATER to match — that edit is what "
            f"makes the burn-down a number in the diff."
        )

    actual = len(baseline)
    if actual > high_water:
        problems.append(
            f"{baseline_path.name} lists {actual} modules; the high-water mark "
            f"is {high_water}. **This list only shrinks.** Do not add a name — "
            f"rename the file to its domain instead."
        )
    elif actual < high_water:
        problems.append(
            f"{baseline_path.name} is down to {actual} from the recorded "
            f"{high_water}. Good — lower BASELINE_HIGH_WATER to {actual} in "
            f"{Path(__file__).name} so the ratchet holds the ground you took."
        )
    return problems


# ---------------------------------------------------------------------------
# selftest — a gate that has never been seen red is not known to be a gate
# ---------------------------------------------------------------------------

def _selftest() -> int:
    cases: List[Tuple[str, List[str], List[str], int, bool]] = [
        # (label, files on disk, baseline rows, high_water, expect_problem)
        ("a NEW issue-named module is REFUSED",
         ["tests/test_widget_pgw9999.py"], [], 0, True),
        ("...in tests_v2 too",
         ["tests_v2/test_widget_th4242.py"], [], 0, True),
        ("a grandfathered issue-named module passes",
         ["tests/test_widget_pgw9999.py"], ["tests/test_widget_pgw9999.py"], 1, False),
        ("a DOMAIN-named module passes",
         ["tests/test_hub_transport.py"], [], 0, False),
        ("every issue-id spelling is caught: gw",
         ["tests/test_a_gw640.py"], [], 0, True),
        ("every issue-id spelling is caught: th",
         ["tests/test_th1130_tail.py"], [], 0, True),
        ("every issue-id spelling is caught: cl",
         ["tests/test_a_cl72.py"], [], 0, True),
        ("every issue-id spelling is caught: ie",
         ["tests/test_a_ie655.py"], [], 0, True),
        ("every issue-id spelling is caught: te",
         ["tests/test_a_te148.py"], [], 0, True),
        ("a version suffix is NOT an issue id",
         ["tests/test_residency_v2.py", "tests/test_input_manifest_v4.py"], [], 0, False),
        # the ratchet itself
        ("a STALE baseline entry is refused (the list cannot rot)",
         [], ["tests/test_gone_pgw1.py"], 1, True),
        ("a PADDED baseline is refused (over the high-water mark)",
         ["tests/test_a_pgw1.py", "tests/test_b_pgw2.py"],
         ["tests/test_a_pgw1.py", "tests/test_b_pgw2.py"], 1, True),
        ("a SHRUNK baseline demands the mark be lowered",
         ["tests/test_a_pgw1.py"], ["tests/test_a_pgw1.py"], 2, True),
    ]
    failures = 0
    for label, files, rows, high_water, expect in cases:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            for root in ROOTS:
                (repo / root).mkdir(parents=True, exist_ok=True)
            for rel in files:
                (repo / rel).write_text("# fixture\n")
            baseline = repo / "baseline.txt"
            baseline.write_text("# fixture\n" + "".join(f"{r}\n" for r in rows))
            got = bool(check(repo, baseline, high_water))
        if got != expect:
            failures += 1
            print(
                f"SELFTEST FAIL — {label}: expected "
                f"{'a finding' if expect else 'clean'}, got the other",
                file=sys.stderr,
            )
    if failures:
        print(f"{failures} selftest case(s) failed", file=sys.stderr)
        return 1
    print(f"selftest OK ({len(cases)} cases)")
    return 0


def main(argv: List[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    problems = check()
    if not problems:
        print(
            f"no NEW issue-named test module "
            f"({len(load_baseline())} grandfathered, and that list only shrinks)"
        )
        return 0
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"{len(problems)} incident-naming problem(s)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
