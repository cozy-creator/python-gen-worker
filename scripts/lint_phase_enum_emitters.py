#!/usr/bin/env python3
"""A PHASE VOCABULARY MEMBER THAT NOTHING REFERENCES CANNOT FIRE (pgw#1480).

`phase` is a typed wire column on `worker_activity_events`, so a phase enum is
an INSTRUMENT: an operator counts by it, and a proof condition is written
against it. `se#780` wrote exactly that — *"`boot_ended_uncompiled` must be
ABSENT"*, and called it "the one that makes this a proof rather than a story".

**It was absent unconditionally.** `EagerPhase.BOOT_ENDED_UNCOMPILED` was
defined and referenced by nothing, so the pass condition could not fail. An
80 GB rental was about to be judged on it.

🔻 **AND IT IS NOT ONE MEMBER.** Measured at `2af52988`: **18 of `EagerPhase`'s
23 members are referenced NOWHERE outside their own definition**, and of the 183
`phase=` emit sites in `src/gen_worker`, **not one** feeds from a phase enum
symbolically. The enum was created (pgw#824, pgw#1035) precisely to close a
"two lists of literals" drift channel; the v2 hardcut then deleted its emitters
and left the vocabulary standing. A whole instrument panel wired to nothing.

WHAT THIS CHECKS, STATED HONESTLY: **referenced-in-code**, not "emitted". An
emit site can be reached through a variable, a dict lookup or a helper, so an
AST-precise "is this emitted" check would report false violations and get
switched off. The weaker property is still decisive for the defect class,
because **a member that nothing anywhere references cannot be emitted by
anything** — which is exactly the shape that made an unfalsifiable pass
condition.

WHY A CENSUS AND NOT A HARD BAN: 18 members are dead TODAY. A gate that is red
on all of them from day one gets bypassed, and deleting them is a WIRE
decision — the hub groups historical `worker_activity_events` by these values —
which a lint script does not get to make. So the dead set is recorded in
`scripts/phase_enum_census.txt` and the gate enforces the three properties that
actually kill the class going forward:

  1. a NEW member with no reference is RED (the class cannot grow);
  2. a censused member that GAINED a reference is RED (the census must shrink,
     so it can never quietly become a graveyard);
  3. a censused member that no longer EXISTS is RED (no stale rows).

Run::

    python scripts/lint_phase_enum_emitters.py
    python scripts/lint_phase_enum_emitters.py --write   # regenerate the census
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "gen_worker"
VENDOR = SRC / "_vendor"
CENSUS = REPO / "scripts" / "phase_enum_census.txt"

#: The phase vocabularies this gate polices, as `module_path::EnumName`.
#: Declared by NAME rather than discovered, for the same reason
#: `lint_serving_process_compiles` declares its module list: "which enums are
#: wire vocabularies" is a fact about the design, not something to infer from a
#: base class that any local enum might share.
POLICED = (("compiled_graph_adopt.py", "EagerPhase"),)


def _members(path: Path, enum_name: str) -> dict[str, str]:
    """`{MEMBER: value}` read out of the source, without importing torch."""

    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != enum_name:
            continue
        found: dict[str, str] = {}
        for row in node.body:
            if (
                isinstance(row, ast.Assign)
                and len(row.targets) == 1
                and isinstance(row.targets[0], ast.Name)
                and isinstance(row.value, ast.Constant)
                and isinstance(row.value.value, str)
            ):
                found[row.targets[0].id] = row.value.value
        return found
    return {}


def _code_lines(text: str) -> list[str]:
    """Lines with whole-line comments dropped — prose that NAMES a member is
    documentation, and a gate that counts it would call a dead member alive."""

    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


def _referenced(
    enum_name: str, member: str, value: str, blobs: dict[Path, str], home: Path
) -> bool:
    symbolic = re.compile(rf"\b{enum_name}\.{member}\b")
    for path, text in blobs.items():
        if path == home:
            continue
        # BOTH checks run over CODE lines only. The symbolic one used to scan
        # the raw text, and a comment EXPLAINING a dead member then made it
        # read as alive — which is the failure direction that matters here,
        # because it marks a dead instrument live. Caught by this gate's own
        # prose arm, not by review.
        for line in _code_lines(text):
            if symbolic.search(line):
                return True
            if f'"{value}"' in line or f"'{value}'" in line:
                return True
    return False


def _read_census() -> set[str]:
    if not CENSUS.is_file():
        return set()
    rows: set[str] = set()
    for line in CENSUS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            rows.add(line)
    return rows


def main(argv: list[str]) -> int:
    blobs = {
        p: p.read_text()
        for p in sorted(SRC.rglob("*.py"))
        if VENDOR not in p.parents
    }
    unreferenced: set[str] = set()
    live: set[str] = set()
    known: set[str] = set()
    for filename, enum_name in POLICED:
        home = SRC / filename
        members = _members(home, enum_name)
        if not members:
            print(
                f"{filename}: no enum {enum_name!r} with string members — this "
                f"gate is asserting over a vocabulary that moved or was "
                f"deleted. Fix POLICED, do not delete the row.",
                file=sys.stderr,
            )
            return 1
        for member, value in members.items():
            key = f"{enum_name}.{member}"
            known.add(key)
            (live if _referenced(enum_name, member, value, blobs, home)
             else unreferenced).add(key)

    if "--write" in argv:
        CENSUS.write_text(
            "# pgw#1480 — phase-vocabulary members that NOTHING in src/gen_worker\n"
            "# references. Generated by `scripts/lint_phase_enum_emitters.py "
            "--write`.\n"
            "#\n"
            "# Each line is a DEFECT, not an exemption: an instrument nothing can\n"
            "# fire is one a proof condition can be written against and never\n"
            "# fail (se#780 nearly spent an 80 GB rental on exactly that).\n"
            "# They are listed so the gate can be green while the set is burned\n"
            "# down, and the gate REFUSES to let this list grow or go stale.\n"
            "#\n"
            "# Wire-or-delete for each is a RULING, not a lint decision: the hub\n"
            "# groups historical `worker_activity_events` by these values.\n"
            "#\n"
            + "\n".join(sorted(unreferenced))
            + "\n"
        )
        print(f"wrote {CENSUS.relative_to(REPO)} ({len(unreferenced)} rows)")
        return 0

    census = _read_census()
    violations: list[str] = []
    for key in sorted(unreferenced - census):
        violations.append(
            f"{key} is referenced NOWHERE outside its own definition, and is "
            f"not in {CENSUS.name}. A phase member nothing references cannot "
            f"be emitted, so any proof condition written against it passes "
            f"unconditionally (pgw#1480). Emit it at the site its docstring "
            f"describes, or delete it.")
    for key in sorted(census & live):
        violations.append(
            f"{key} IS referenced now but is still listed in {CENSUS.name}. "
            f"Remove the line — a census that only grows becomes a graveyard, "
            f"and this is the check that stops it.")
    for key in sorted(census - known):
        violations.append(
            f"{key} is in {CENSUS.name} but no longer exists. Remove the line.")

    for line in violations:
        print(line, file=sys.stderr)
    if violations:
        print(f"\n{len(violations)} phase-vocabulary violation(s).", file=sys.stderr)
        return 1
    print(
        f"phase-vocabulary fence: clean "
        f"({len(live)} referenced, {len(unreferenced)} censused as unwired)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
