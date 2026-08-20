#!/usr/bin/env python3
"""The post-connect resolution surface may not GROW.

The threat: a connected worker that can reconstruct or substitute a Hub decision
is a SECOND RESOLVER. The Hub issues one execution decision; a worker that
re-derives any part of it can serve a different checkpoint, lane, arm or
artifact than the one that was billed, ranked and recorded — silently, because
both answers look plausible from the outside. Lane authority lives in the
endpoint's declaration and the owner's ladder, never in worker code; the wire
contract says outright that *"the worker never calls tensorhub for ref
resolution; the orchestrator is the only resolver"*
(`proto/worker_scheduler.proto`, `Snapshot`).

A gate rather than the repair: the real repair is an exact `ExecutionSpec` whose
schema half is hub-side, so the deletion cannot land yet. This gate freezes the
census at its current enumerated set so the deletion stays a bounded list rather
than an open-ended hunt. It is not a behaviour change and it fails on ADDITION
only — the accepted `CONNECTED` sites still run on every boot.

What is watched:

    discover / _discover_inner / _candidates   catalog listing + sibling ranking
                                               after the Hub already chose
                                               (the `rows[0]` resolver)
    resolve_repo                               worker-side ref resolution
    parse_execution_lane_spec                  the DUAL-form parse; its FAMILY
                                               branch is the coarse-family
                                               expansion §1.31 forbids

Keyed on `<path>::<callee>`, never on a line number: a `path:line` key goes red
whenever a sibling PR shifts lines in a file nobody in that change touched.

Every accepted site must NAME ITS CLASSIFICATION:

    CONNECTED    reachable from the connected dispatcher — a real second
                 resolver. Replacement-gated; every line must say what blocks
                 its deletion.
    STANDALONE   CLI / hub-less only, unreachable from the connected
                 dispatcher. LEGITIMATE.
    PUBLISH      publish/declaration-time validation vocabulary, which is
                 retained ("declarations stay as capability vocabulary for
                 publish/mint validation only").
    VOCABULARY   parsing/definition with no resolution effect.

Only CONNECTED is a defect. Baselined green on arrival — a gate that fails on
day one gets switched off.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lint_side  # noqa: E402
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
    "aot_compiled_graphs.py",
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
        _lint_side.report([f"+ {key}" for key in new],
                          "pgw#891 post-connect resolution surface")
        return 1

    if stale:
        # A site that went away is good news, but the allowlist must not keep
        # carrying it: a stale exemption silently re-permits the site if it
        # ever comes back. Same defect class the gate exists to police.
        print("allowlist has entries with no matching call site — delete them:",
              file=sys.stderr)
        _lint_side.report([f"- {key}" for key in stale],
                          "pgw#891 post-connect resolution surface")
        return 1

    connected = sum(1 for c in accepted.values() if c == "CONNECTED")
    print(
        f"post-connect resolution surface: {len(found)} sites, "
        f"{connected} CONNECTED (pgw#904's replacement-gated deletion set)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
