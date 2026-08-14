#!/usr/bin/env python3
"""NO EXPORT AND NO AOTI COMPILE IN THE SERVING PROCESS.

    th#1299 was never "a mint needs its own process". Its defect was
    inductor's GIL-holding orchestration starving the ONE asyncio task that
    carries the 10 s beat and eager serving. Spawn / poll / collect / digest /
    CAS-write / publish are I/O and supervision and do not starve that loop —
    so the greenfield compile loop (th#1834 §PHASE 3, pgw#1215) deletes the
    middle mint-child tier and lets the serving parent supervise compile
    children directly.

    What used to be bought with a whole extra pipeline load is bought here
    instead: **``torch.export`` and the AOTI compile entry points may only be
    NAMED by a module that runs inside a compile/trace child.** Every other
    module in ``src/gen_worker`` is serving-process code by default and a call
    there is red.

Why the fence lands WITH the supervisor rewrite and not after it: a fence that
arrives after the change it guards never had a chance to go red.

**``torch.compile`` is deliberately NOT banned.** th#1834's coordinator ruling
keeps the dynamo arm as JIT intake, and pgw#680's guard-miss heal doctrine
lives on it — so dynamo compiles legitimately happen in the serving process.
The other half of th#1299 covers them and is structural rather than lexical:
every dynamo compile runs on the ONE background warm thread inside the
executor's exclusive turn gate, and since pgw#1215 step 4 an ungated router
refuses typed (``hot_swap.RouterNotGated``) instead of degrading to an inline
compile on the serving thread. Banning ``torch.compile`` here would fence the
arm that ruling keeps; leaving the turn gate optional would leave the hole
this fence exists to close. Both halves, or neither.

**What "runs inside a child" means, and why the list is short.** A child-only
module is one whose export/compile calls are reached only from a
``python -m`` child entry point. That is a fact about the tree, so it is
declared here by NAME with a reason each, not inferred from an import closure
— ``aot_mint`` is imported by ``executor`` (for phase-event re-emission) and
an import-based fence would therefore be red today over no violation at all.

**Known limit, stated rather than papered over:** this reads CALL NODES in the
AST, not strings and not bare attribute access. A deferred
``getattr(torch.export, "save")`` in a serving module would pass. That is a
deliberate trade — the banned names appear in dozens of comments and
docstrings that describe the boundary correctly, and a string check would
make the gate pure noise. The cell-key layout fence counts strings because
its banned vocabulary never appears in prose; this one's does.

Run::

    python scripts/lint_serving_process_compiles.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "gen_worker"

#: Dotted attribute paths whose CALL is a graph export or an AOTI compile.
#: Matched on the trailing segments, so ``torch.export.save(...)`` and a
#: ``from torch.export import save`` -> ``save(...)`` both hit.
BANNED_CALLS: Tuple[Tuple[str, ...], ...] = (
    ("torch", "export", "export"),
    ("torch", "export", "export_for_training"),
    ("torch", "export", "save"),
    ("torch", "export", "load"),
    ("torch", "_dynamo", "export"),
    ("aoti_compile_and_package",),
    ("aot_compile",),
)

#: Modules whose export/AOTI calls run ONLY inside a compile/trace child
#: process. Every entry states the child that reaches it. Adding a row is a
#: claim that the module never executes on the serving parent's loop — make it
#: deliberately.
CHILD_ONLY: Dict[str, str] = {
    "aot_mint": (
        "the export + AOTI recipe itself; reached from the compile child "
        "(`aot_compile_child`) and the trace child (`boot_trace_child`). The "
        "serving parent imports it for `emit_phase_events` / "
        "`pack_graph_classes` only — supervision and packaging, never a "
        "compile."
    ),
    "aot_compile_pool": (
        "the K-wide compile driver: it SPAWNS children and stages their "
        "programs. `torch.export.save` here is the parent half of the pool's "
        "process boundary (pgw#784) and pgw#1215's keystone is deleting it — "
        "the child traces its own share."
    ),
    "aot_compile_child": (
        "`python -m gen_worker.aot_compile_child` — the pool's child half: it "
        "`torch.export.load`s the staged program and AOTI-compiles it."
    ),
}


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)


def _dotted(node: ast.AST) -> Tuple[str, ...]:
    out: List[str] = []
    while isinstance(node, ast.Attribute):
        out.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        out.append(node.id)
    return tuple(reversed(out))


def _is_banned(path: Tuple[str, ...]) -> bool:
    for banned in BANNED_CALLS:
        if len(path) >= len(banned) and path[-len(banned):] == banned:
            return True
    return False


def main() -> int:
    violations: List[str] = []
    seen_allowed: set[str] = set()
    for file in sorted(SRC.rglob("*.py")):
        module = _module_name(file)
        tree = ast.parse(file.read_text(), filename=str(file))
        hits: List[Tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted(node.func)
            if dotted and _is_banned(dotted):
                hits.append((node.lineno, ".".join(dotted)))
        if not hits:
            continue
        if module in CHILD_ONLY:
            seen_allowed.add(module)
            continue
        for lineno, name in hits:
            violations.append(
                f"{file.relative_to(REPO)}:{lineno}: {name}() runs in the "
                f"SERVING PROCESS. th#1299: inductor's GIL-holding "
                f"orchestration starves the beat and eager serving. Move it "
                f"into a compile/trace child, or — if this module really is "
                f"child-only — declare it in CHILD_ONLY with the child that "
                f"reaches it.")

    # A fence that names a DELETED module guards nothing and passes vacuously
    # forever (th#1834's census records the ratchet trap of exactly this
    # shape: a stale name prints `unused section(s)` and EXITS 0).
    stale = sorted(set(CHILD_ONLY) - seen_allowed)
    for module in stale:
        violations.append(
            f"CHILD_ONLY names `{module}`, which no longer calls any banned "
            f"symbol. Drop the row — a fence exemption for a module that "
            f"stopped violating is a hole waiting for the next edit.")

    for line in violations:
        print(line, file=sys.stderr)
    if violations:
        print(
            f"\n{len(violations)} serving-process compile fence violation(s).",
            file=sys.stderr)
        return 1
    print(
        f"serving-process compile fence: clean "
        f"({len(seen_allowed)} declared child-only module(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
