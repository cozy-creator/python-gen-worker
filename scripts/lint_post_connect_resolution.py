#!/usr/bin/env python3
"""pgw#891 / pgw#904: the post-connect resolution surface may not GROW.

The threat, named as DESIGN-RULINGS §4.24 requires
-------------------------------------------------
A connected worker that can reconstruct or substitute a Hub decision is a
SECOND RESOLVER. The Hub issues one execution decision; a worker that re-derives
any part of it can serve a different checkpoint, lane, arm or artifact than the
one that was billed, ranked and recorded — silently, because both answers look
plausible from the outside. §1.10 and §1.31 put lane authority in the endpoint's
declaration and the owner's ladder, never in worker code; the wire contract says
outright that *"the worker never calls tensorhub for ref resolution; the
orchestrator is the only resolver"* (`proto/worker_scheduler.proto`, `Snapshot`).

Why a gate now, when the fix is pgw#891
---------------------------------------
The real repair is pgw#891's exact `ExecutionSpec`, and its schema half is
th#1457's — so the deletion cannot land yet. What CAN be protected today is the
SIZE of the job: this gate freezes the census at its current, enumerated set so
pgw#904's deletion stays a bounded list rather than an open-ended hunt. It is
deliberately not a behaviour change, and it fails on ADDITION only.

That is the whole claim. This gate does not prevent the second resolver from
running — the accepted `CONNECTED` sites below run on every boot today. It
prevents a ninth one appearing while the replacement is built.

What is watched, and why each one
---------------------------------
    discover / _discover_inner / _candidates   catalog listing + sibling ranking
                                               after the Hub already chose
                                               (pgw#904's `rows[0]` resolver)
    resolve_repo                               worker-side ref resolution
    parse_execution_lane_spec                  the DUAL-form parse; its FAMILY
                                               branch is the coarse-family
                                               expansion §1.31 forbids

Keyed on `<path>::<callee>`, never on a line number. That is not a style
preference — `scripts/lint_config_reads.py` learned it the hard way: its first
cut keyed `path:line` and went red within the hour when two sibling PRs shifted
lines in files nobody in that change had touched. A line number is a fact other
people change independently.

Every accepted site must NAME ITS CLASSIFICATION, which is what stops the
allowlist decaying into prose:

    CONNECTED    reachable from the connected dispatcher — a real second
                 resolver. Replacement-gated on pgw#891/pgw#904; every line
                 must say what blocks its deletion.
    STANDALONE   CLI / hub-less only, unreachable from the connected
                 dispatcher. This is pgw#904's box 5 and it is LEGITIMATE.
    PUBLISH      publish/declaration-time validation vocabulary, which pgw#891
                 explicitly retains ("declarations stay as capability
                 vocabulary for publish/mint validation only").
    VOCABULARY   parsing/definition with no resolution effect.

Only CONNECTED is a defect. Baselined green on arrival, following th#1383's
precedent — a gate that fails on day one gets switched off.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO / "src" / "gen_worker"
ALLOWLIST = REPO / "scripts" / "post_connect_resolution_allowlist.txt"

CLASSIFICATIONS = {"CONNECTED", "STANDALONE", "PUBLISH", "VOCABULARY"}

#: Callees whose invocation is a post-connect resolution act. Name-keyed: a
#: single-file AST walk cannot resolve an import chain, and the identifier is
#: the stable key regardless. A same-named callee elsewhere simply needs its own
#: classified line — which makes the census MORE honest, not less.
WATCHED = frozenset({
    "discover",
    "_discover_inner",
    "_candidates",
    "resolve_repo",
    "parse_execution_lane_spec",
})

#: Modules that DEFINE the watched surface. A call inside its own definition
#: module is an implementation detail, not a consumer reaching for a resolver.
DEFINING_MODULES = frozenset({
    "aot_cells.py",
    "hub_client.py",
    "ladder.py",
    "execution_lanes.py",
})


def _callee(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def census() -> set[str]:
    """Every `<relpath>::<callee>` reaching the watched surface."""
    found: set[str] = set()
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.name in DEFINING_MODULES:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # a parse failure must not read as "clean"
            print(f"error: cannot parse {path}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _callee(node) in WATCHED:
                found.add(f"{path.relative_to(REPO).as_posix()}::{_callee(node)}")
    return found


def allowed() -> dict[str, str]:
    """`<path>::<callee>` -> classification, from the allowlist file."""
    out: dict[str, str] = {}
    if not ALLOWLIST.exists():
        return out
    for raw in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 2:
            print(f"error: malformed allowlist line: {raw}", file=sys.stderr)
            raise SystemExit(2)
        key, classification = parts[0], parts[1]
        if classification not in CLASSIFICATIONS:
            print(
                f"error: unknown classification {classification!r} for {key} "
                f"(want one of {', '.join(sorted(CLASSIFICATIONS))})",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if len(parts) < 3 or not parts[2].strip():
            print(f"error: {key} has no reason — every line must name one",
                  file=sys.stderr)
            raise SystemExit(2)
        out[key] = classification
    return out


def main() -> int:
    found, accepted = census(), allowed()

    new = sorted(found - set(accepted))
    stale = sorted(set(accepted) - found)

    if new:
        print(
            "post-connect resolution surface GREW — pgw#891/pgw#904.\n"
            "\n"
            "A connected worker may not reconstruct or substitute a Hub\n"
            "decision. If this site is CLI/hub-less only, classify it\n"
            "STANDALONE with the reason it is unreachable from the connected\n"
            "dispatcher. If it is genuinely on the connected path, it is a new\n"
            "second resolver and needs a ruling, not an allowlist line.\n",
            file=sys.stderr,
        )
        for key in new:
            print(f"  + {key}", file=sys.stderr)
        return 1

    if stale:
        # A site that went away is good news, but the allowlist must not keep
        # carrying it: a stale exemption silently re-permits the site if it
        # ever comes back. Same defect class the gate exists to police.
        print("allowlist has entries with no matching call site — delete them:",
              file=sys.stderr)
        for key in stale:
            print(f"  - {key}", file=sys.stderr)
        return 1

    connected = sum(1 for c in accepted.values() if c == "CONNECTED")
    print(
        f"post-connect resolution surface: {len(found)} sites, "
        f"{connected} CONNECTED (pgw#904's replacement-gated deletion set)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
