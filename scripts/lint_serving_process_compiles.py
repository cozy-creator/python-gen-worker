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
        "(`aot_compile_child`) and the trace child (`boot_trace_child`). "
        "pgw#1215 step 4 made the serving parent CALL into this module — "
        "`mint_supervisor` drives `mint_graph_classes`, `declared_class_rows` "
        "and `fold_held_graph_classes` on the serving process — so the claim "
        "this row makes is narrower than it looks and is worth restating: "
        "every one of those is supervision, enumeration or folding. The "
        "banned calls all sit under `_export_entry`, which only "
        "`trace_for_key` reaches and only a child calls."
    ),
    "model.export": (
        "pgw#1332's DECLARATION-TIME export. It is not reached from a serving "
        "process at all, and the claim is structural rather than a promise: "
        "`gen_worker.model`'s facade does not import it, the generated "
        "bindings do not import it, and `cli/model.py` imports it inside the "
        "`export` handler — so the only way to run it is to type "
        "`gen-worker model export`, which is a catalog BUILD step whose output "
        "is committed. `tests/test_model_sdk.py::"
        "test_the_family_facade_does_not_import_the_exporter` executes that "
        "claim instead of restating it. The trace is fake-tensor only: no "
        "weights, no GPU, no compile, so even run by hand it is shape "
        "arithmetic rather than the GIL-holding inductor orchestration this "
        "fence exists to keep off the beat."
    ),
}

#: Modules a SUPERVISING parent calls into, asserted to hold no banned call of
#: their own. Not exemptions — the opposite: naming them here is a claim that
#: the default rule (serving-process code may not compile) applies to them,
#: and it is checked, so a later edit that puts an export in the supervisor's
#: call path reds even though the module was never suspected.
#:
#: pgw#1215 step 4 is why the list exists. `aot_compile_pool` and
#: `aot_compile_child` WERE declared child-only until `601af398` shrank the
#: set — they were exempt to describe a `torch.export.save`/`load` round trip
#: the keystone deleted. The supervisor now calls the pool directly from the
#: serving loop, so their status changed from "exempt" to "must be clean", and
#: a stale exemption would have been indistinguishable from a real one.
#: Modules that are a standalone ENTRY POINT: a `python -m` probe whose whole
#: process exists to compile and measure, and which no serving-package module
#: imports. That last half is the difference between this and an exemption —
#: it is a CHECKED claim, and a serving module that starts importing one of
#: these goes red here even though the probe itself never changed.
STANDALONE_ENTRY: Dict[str, str] = {
    "benchmarks.adopt_only_serve": (
        "pgw#1328's GPU bar. Its FIRST phase mints one small graph so its "
        "SECOND — a child process that declared the adopt-only role and "
        "installed the import blocker — has something to adopt; the compile "
        "IS the fixture, and the process serves no tenant. The claim this row "
        "makes is narrow and checked: the export lives in the eager-capable "
        "parent, and no serving-package module imports this probe."
    ),
    "benchmarks.store_arm_parity": (
        "pgw#1329's GPU parity probe. It mints one small graph and arms it "
        "twice to compare bytes; the compile IS the measurement, and the "
        "process serves nothing. Not CHILD_ONLY: no serving parent spawns it."
    ),
}

SUPERVISED: Dict[str, str] = {
    "aot_compile_pool": (
        "the K-wide pool. `mint_supervisor` constructs and drives it on the "
        "serving process (in a worker thread); it spawns, reaps and collects, "
        "and it must never itself export or compile."
    ),
    "mint_supervisor": (
        "the serving parent's own mint driver. Supervision only: enumerate, "
        "spawn, collect, adopt, publish."
    ),
    "mint_adapter": (
        "pgw#1328's eager-capable side of the serve/mint seam. It is the one "
        "module whose job is to NAME the mint lane, so it is the obvious place "
        "for an export to grow — and it must never hold one: every method is "
        "supervision or declaration reading, delegated to `mint_supervisor` "
        "and `aot_mint`, which carry their own rows here."
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
    seen_modules: set[str] = set()
    for file in sorted(SRC.rglob("*.py")):
        module = _module_name(file)
        seen_modules.add(module)
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
        if module in CHILD_ONLY or module in STANDALONE_ENTRY:
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
    stale = sorted((set(CHILD_ONLY) | set(STANDALONE_ENTRY)) - seen_allowed)
    for module in stale:
        violations.append(
            f"CHILD_ONLY names `{module}`, which no longer calls any banned "
            f"symbol. Drop the row — a fence exemption for a module that "
            f"stopped violating is a hole waiting for the next edit.")

    # The same staleness rule on the other list: a SUPERVISED row that names a
    # module the tree no longer has is a claim about nothing. It cannot go red
    # for a violation (a supervised module is checked by the default rule
    # above, like every other module), so the only way it can go red at all is
    # this — and a row that can never go red is not a fence.
    for module in sorted(set(SUPERVISED) - seen_modules):
        violations.append(
            f"SUPERVISED names `{module}`, which is not a module in "
            f"src/gen_worker. Drop the row or fix the name — it is asserting "
            f"the serving-process rule over a file that does not exist.")
    # The standalone claim, CHECKED: nothing in the serving package may import
    # a module that is allowed to compile because it is nobody's dependency.
    for file in sorted(SRC.rglob("*.py")):
        module = _module_name(file)
        if module in STANDALONE_ENTRY:
            continue
        imported: set[str] = set()
        for node in ast.walk(ast.parse(file.read_text(), filename=str(file))):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                imported.add(base)
                imported.update(f"{base}.{alias.name}" for alias in node.names)
        for entry in STANDALONE_ENTRY:
            # `benchmarks.store_arm_parity` reached as itself, as
            # `gen_worker.benchmarks.store_arm_parity`, or as a relative
            # `from .benchmarks import store_arm_parity`.
            if not any(
                name == entry or name.endswith(f".{entry}")
                for name in imported
            ):
                continue
            violations.append(
                f"{file.relative_to(REPO)}: imports `{entry}`, which "
                f"STANDALONE_ENTRY exempts from the serving-process compile "
                f"fence precisely because nothing imports it. Either drop the "
                f"import or drop the exemption.")

    overlap = sorted(set(SUPERVISED) & (set(CHILD_ONLY) | set(STANDALONE_ENTRY)))
    for module in overlap:
        violations.append(
            f"`{module}` is in BOTH CHILD_ONLY and SUPERVISED — it cannot be "
            f"exempt from the serving-process rule and asserted to obey it. "
            f"Decide which the module is.")

    for line in violations:
        print(line, file=sys.stderr)
    if violations:
        print(
            f"\n{len(violations)} serving-process compile fence violation(s).",
            file=sys.stderr)
        return 1
    print(
        f"serving-process compile fence: clean "
        f"({len(seen_allowed)} declared child-only/standalone module(s), "
        f"{len(SUPERVISED)} supervised module(s) proven clean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
