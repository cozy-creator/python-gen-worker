#!/usr/bin/env python3
"""The mypy strictness ratchet only turns one way.

`[tool.mypy] strict = true` makes every module strict by DEFAULT — a module
added tomorrow is born strict and nobody has to remember anything. The 101
modules that were not clean when the ratchet was adopted are named in
per-module override lists, each relaxing exactly one strict flag.

Those lists are the whole risk. `strict = true` cannot silently stop being
true — it is one line, present or absent — but a list can grow one name at a
time until the gate means nothing, and each individual addition looks
reasonable in review. This script is the mechanism that refuses that, because
the alternative is remembering, and remembering demonstrably fails.

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

#: flag name -> (high-water EXACT-module count, errors it was hiding at adoption).
#: Measured 2026-08-12 on 28018a0a: 413 --strict errors in 101 of 266 src modules,
#: and 2,016 errors in 316 of 486 test modules once the test tree could be checked
#: at all. Lower a number only together with the names you deleted from
#: pyproject.toml — that is what makes the burn-down a number in the diff.
HIGH_WATER: Dict[str, Tuple[int, int]] = {
    # 69 -> 62: pgw#1202 PR 3 cleared the endpoint-AUTHORING surface
    # (gen_worker.api.* + gen_worker.testing), 30 bare generics.
    # 62 -> 60: PR 6 cleared request_context (+ ._stream) — the `ctx` object
    # every handler is handed.
    # 60 -> 56: PR 7 cleared the BUILD-TIME surface (registry,
    # discovery.discover, discovery.execution_lanes, entrypoint) — what an
    # endpoint image build exercises.
    # 56 -> 55: pgw#1232 moved the generic transfer journal to tensorfs.
    # 55 -> 54: pgw#1270 deleted the duplicate worker package/runner surface.
    # pgw#1277: 54 -> 53. compiled_graph_key.py was relaxed here; its successor
    # gen_worker/graph_facts.py needs no relaxation, so the ground is kept.
    # 53 -> 52 / 25 -> 24 / 227 -> 226 (pgw#1466): tcg#56's rename retired a
    # relaxed module across all three flags at once. Recording the ground the
    # rename took, which is what this guard exists to make someone do.
    "disallow_any_generics": (52, 285),
    # 26 -> 25: PR 6, request_context.
    "warn_return_any": (24, 48),
    # 20 -> 19: pgw#1270 deleted the duplicate worker package/runner surface.
    "disallow_untyped_calls": (19, 37),
    "disallow_untyped_decorators": (3, 9),
    # implicit_reexport is split in two: our modules (a burn-down) and
    # third-party stub export gaps (not our debt — upstream's __all__).
    # 17 -> 16: pgw#1202 PR 4 closed 1 of the 8 OURS (the other 9 are
    # third-party stub gaps and are not debt). The remaining 5 are blocked on
    # open lanes owning their importers, not on difficulty.
    # 16 -> 15: pgw#1232 deleted the worker-owned chunk uploader.
    "implicit_reexport": (15, 34),
    # test modules still dirty at the relaxed test posture.
    # 170 of 486 were already clean and are checked from that commit on.
    # 314 -> 312: PR 8 took the two HARNESS doubles — `tests.harness.compiled_graph_hub`
    # and `tests.harness.adopt_rig`. These are the modules the "Protocols for
    # test doubles" item exists for, and they were wholly unchecked: a bogus
    # attribute plus a `reveal_type` injected into compiled_graph_hub.py still produced
    # `Success: no issues found`.
    # 312 -> 309: pgw#1215 deleted `tests.test_meta_roundtrip_pgw1111`,
    # `tests.test_shape_hints_pgw998` and
    # `tests.test_overlap_export_compile_pgw1052` with the save/load round trip
    # and the overlapped export/compile shape they pinned.
    # 309 -> 302: pgw#1235 deleted seven duplicate generic CAS test modules.
    # 302 -> 295: pgw#1246 deleted the seven component-substitution test
    # modules with the apparatus they pinned.
    # 295 -> 269: pgw#1272 deleted 40 mock-heavy unit modules, 5 deletion
    # fences for completed hard cuts, and one orphaned fixture.
    # test_group_processes_pgw783 was NOT deleted despite scoring mock-heavy:
    # it holds the only red-proof of `host_move_guard._refuse_if_over_budget`,
    # a live on-by-default guard, and the only coverage of procsplit.group/merge.
    # 269 -> 255: pgw#1270 deleted the implementation-only tests of the worker
    # package/runner/resume authorities TCG now owns.
    # 255 -> 242: pgw#1362 wave 1 folded 27 incident-named hub-wire test
    # modules into 5 domain modules and typed them clean, so the 13 of them
    # that were unchecked came off the list and nothing went back on.
    # 242 -> 238: wave 2 did the same to the cli/config/sdk cluster (9 modules
    # -> 3, plus one rename), taking four more off.
    # 238 -> 235: wave 3a folded the child-fault modules (5 -> 1), taking three.
    # 235 -> 231: pgw#1362 wave 3b folded the procsplit group (5 -> 2) and
    # hoisted its shared rig into tests/harness/split.py; all three are typed
    # clean, so four more unchecked modules came off with none added.
    # 231 -> 227: pgw#1373 deleted the v1 SDK and its test corpus.
    "ignore_errors": (226, 2016),
}

#: WILDCARD patterns are structural policy, not debt, so they are not counted
#: against a high-water mark — but they must be declared here, so a new blanket
#: exemption cannot be introduced by adding one line to pyproject.toml.
DECLARED_WILDCARDS: Dict[str, str] = {
    "gen_worker.pb.*": "generated protobuf; no source of ours to fix",
    "gen_worker._vendor.*": "vendored upstream snapshot; a digest fence refuses "
                            "an in-place fix (pgw#1310)",
    "gen_worker.convert.*": "inherited from cozy_convert; bodies ARE checked",
    "tests.*": "test fns may be `def test_x():`; contract checks stay on",
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

        wildcards = [m for m in modules if "*" in m]
        for pattern in wildcards:
            if pattern not in DECLARED_WILDCARDS:
                problems.append(
                    f"override #{index} exempts the WILDCARD `{pattern}`, "
                    f"which is not declared in {Path(__file__).name}. A "
                    f"blanket exemption covers modules that do not exist yet, "
                    f"so it needs a written reason before it can be used."
                )
        exact = [m for m in modules if "*" not in m]
        if not exact:
            # A declared-wildcard block. The declaration above is its
            # justification, so the flags it relaxes need no separate number.
            continue

        for key in relaxers:
            if key not in HIGH_WATER:
                problems.append(
                    f"override #{index} sets `{key}` on named modules, and "
                    f"`{key}` has no high-water mark in "
                    f"{Path(__file__).name}. A new way to opt out of strict "
                    f"needs a name and a number here first."
                )
                continue
            counts[key] = counts.get(key, 0) + len(exact)

    # The config above is worthless if CI stops handing mypy the paths. Dropping
    # `tests tests_v2` from the invocation would return the suite to zero static
    # coverage and every check would stay green — the exact shape of failure
    # this repo keeps paying for, so it is asserted rather than trusted.
    workflow = pyproject.parent / ".github" / "workflows" / "ci.yaml"
    if workflow.exists():
        invocations = [
            line.strip() for line in workflow.read_text().splitlines()
            if "run:" in line and " mypy " in line
        ]
        if not invocations:
            problems.append(f"no mypy invocation found in {workflow.name}")
        # The UNION of the invocations, not each one. The property being
        # guarded is "every path the config covers is handed to mypy
        # somewhere", and per-invocation matching is a stricter reading that
        # forbids a legitimate shape: a SECOND, deliberately-scoped call for a
        # tree that needs a different `MYPYPATH` (pgw#1347's `scripts/mint_rig`,
        # which must not join the shared path — doing so makes mypy resolve
        # every other scripts/*.py a test imports). Union is not weaker: the
        # failure this exists for is `tests tests_v2` disappearing, and dropping
        # them from ALL invocations still goes red.
        covered = " ".join(invocations)
        # pgw#1373 deleted `tests_v2/` — it was the v1 executor/transport suite
        # (`@endpoint`, `Slot`, `registry`), not the pgw#1382 v2 SDK its name reads as.
        for required in ("src/gen_worker", "tests"):
            if required not in covered:
                problems.append(
                    f"no mypy step in {workflow.name} checks `{required}`; the "
                    f"invocations found were {invocations!r}. Every path the "
                    f"config covers has to actually be handed to mypy, or the "
                    f"exemption lists below describe a check nobody runs."
                )

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
