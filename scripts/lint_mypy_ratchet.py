#!/usr/bin/env python3
"""pgw#1202: the mypy strictness ratchet only turns one way.

`[tool.mypy] strict = true` makes every module strict by DEFAULT — a module
added tomorrow is born strict and nobody has to remember anything. The 101
modules that were not clean when the ratchet was adopted are named in
per-module override lists, each relaxing exactly one strict flag.

Those lists are the whole risk. `strict = true` cannot silently stop being
true — it is one line, present or absent — but a list can grow one name at a
time until the gate means nothing, and each individual addition looks
reasonable in review. This script is the mechanism that refuses that, because
the alternative is remembering, and remembering demonstrably fails: pgw#718
found a sixth instance of a defect class in a file whose author had already
fixed it once, in the same session.

So each list carries a HIGH-WATER MARK here. A list may shrink freely (that is
the burn-down, and the shrunk number is committed with it). A list that grows,
a new relaxing override that is not recorded, or `strict = true` going missing
is a hard failure.

Burning a module down means: fix it, delete its name, lower the number below.
The two edits land together, so the diff states the progress as a number.

Usage:

    python scripts/lint_mypy_ratchet.py [PYPROJECT]
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]

#: flag name -> (high-water module count, errors it was hiding at adoption).
#: Measured 2026-08-12 on 28018a0a: 413 --strict errors in 101 of 266 modules.
#: Lower a number only together with the names you deleted from pyproject.toml.
HIGH_WATER: Dict[str, Tuple[int, int]] = {
    "disallow_any_generics": (69, 285),
    "warn_return_any": (26, 48),
    "disallow_untyped_calls": (20, 37),
    "disallow_untyped_decorators": (3, 9),
    # implicit_reexport is split in two: our modules (a burn-down) and
    # third-party stub export gaps (not our debt — upstream's __all__).
    "implicit_reexport": (17, 34),
}

#: Overrides that relax something for a reason other than a burn-down. These
#: are exempt from the growth check but not from being listed: an override
#: appearing in neither table is a failure, so a new escape hatch cannot be
#: added without touching this file.
DECLARED_NON_BURNDOWN: Dict[str, str] = {
    "ignore_errors": "generated protobuf modules (gen_worker.pb.*)",
    "disallow_untyped_defs": "gen_worker.convert.* — inherited from cozy_convert",
}

#: Every strict-implied option that costs zero errors today and is therefore
#: simply ON. Naming them makes a future relaxation visible as a diff to this
#: list rather than as one word in a config block.
FREE_STRICT_OPTIONS: Tuple[str, ...] = (
    "check_untyped_defs",
    "disallow_incomplete_defs",
    "disallow_subclassing_any",
    "extra_checks",
    "no_implicit_optional",
    "strict_equality",
    "warn_redundant_casts",
    "warn_unused_ignores",
)


def check(pyproject: Path) -> List[str]:
    """Return every reason the ratchet is broken; empty means it holds."""
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)

    mypy = data.get("tool", {}).get("mypy", {})
    problems: List[str] = []

    if mypy.get("strict") is not True:
        problems.append(
            "[tool.mypy] strict is not true — the whole ratchet rests on it, "
            "and without it every module below is unchecked rather than "
            "partially checked."
        )

    for option in FREE_STRICT_OPTIONS:
        if mypy.get(option) is False:
            problems.append(
                f"[tool.mypy] {option} = false — this option cost zero errors "
                f"when the ratchet was adopted, so turning it off is paying "
                f"for nothing."
            )

    counts: Dict[str, int] = {}
    for index, override in enumerate(mypy.get("overrides", [])):
        modules = override.get("module", [])
        if isinstance(modules, str):
            modules = [modules]
        relaxers = [key for key in override if key != "module"]
        if not relaxers:
            problems.append(f"override #{index} relaxes nothing; delete it")
            continue
        for key in relaxers:
            if key in DECLARED_NON_BURNDOWN:
                continue
            if key not in HIGH_WATER:
                problems.append(
                    f"override #{index} sets `{key}`, which is recorded in "
                    f"neither HIGH_WATER nor DECLARED_NON_BURNDOWN in "
                    f"{Path(__file__).name}. A new way to opt out of strict "
                    f"needs a name and a number here first."
                )
                continue
            counts[key] = counts.get(key, 0) + len(modules)

    for flag, (limit, errors) in HIGH_WATER.items():
        actual = counts.get(flag, 0)
        if actual > limit:
            problems.append(
                f"`{flag}` is relaxed for {actual} modules; the high-water "
                f"mark is {limit}. This list shrinks and never grows — fix "
                f"the {actual - limit} new module(s) instead of exempting "
                f"them. (At adoption it hid {errors} errors.)"
            )
        elif actual < limit:
            problems.append(
                f"`{flag}` is now relaxed for only {actual} modules, below "
                f"the recorded {limit}. Good — lower HIGH_WATER to {actual} "
                f"in {Path(__file__).name} so the ratchet holds the ground "
                f"you just took."
            )

    return problems


def main(argv: List[str]) -> int:
    pyproject = Path(argv[1]) if len(argv) > 1 else REPO / "pyproject.toml"
    problems = check(pyproject)
    if problems:
        print("pgw#1202: the mypy strictness ratchet turned the wrong way:\n",
              file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}\n", file=sys.stderr)
        return 1
    total = sum(limit for limit, _ in HIGH_WATER.values())
    print(f"pgw#1202: strict = true; {total} module-exemptions remaining "
          f"across {len(HIGH_WATER)} flags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
