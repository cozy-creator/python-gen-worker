#!/usr/bin/env python3
"""NO EXPORT AND NO AOTI COMPILE IN THE SERVING PROCESS.

    th#1299 was never "a mint needs its own process". Its defect was
    inductor's GIL-holding orchestration starving the ONE asyncio task that
    carries the 10 s beat and eager serving. Spawn / poll / collect / digest /
    CAS-write / publish are I/O and supervision and do not starve that loop —
    so the greenfield compile loop (th#1834 §PHASE 3, pgw#1215) deletes the
    middle mint-child tier and lets the serving parent supervise compile
    children directly.

    **``torch.export`` and the AOTI compile entry points may only be NAMED by a
    module that runs inside a compile/trace child.** Every other module in
    ``src/gen_worker`` is serving-process code by default and a call there is red —
    bought structurally rather than with a whole extra pipeline load.

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
make the gate pure noise. The compiled graph-key layout fence counts strings because
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
    "_vendor.torchcg.discovery": (
        "the trace-once-at-publish instrumented derive (tcg#41/pgw#1370): "
        "torch.export runs per OBSERVED call to hash graph specializations into the "
        "release document. It executes at PUBLISH time — the derive driver's "
        "CPU pass inside the release image (pgw#1370/th#2134), never on a "
        "serving pod (the ruling: serving pods are ADOPT-ONLY, no trace, no "
        "derivation). The serving path consumes `_vendor.torchcg.adopt`, "
        "which holds no export; nothing in the boot or dispatch flow imports "
        "this module. The export here is fake-tensor shape arithmetic (the "
        "model.export precedent), not inductor orchestration."
    ),
    "release.derive": (
        "the publish-time instrumented derive (pgw#1370). `torch.export.save` "
        "here SERIALIZES an already-traced ExportedProgram into the tensorfs "
        "CAS (Paul, 2026-08-20: the derive stores the whole traced graph, and "
        "the runtime miner downloads it and runs inductor rather than "
        "re-tracing). It runs at PUBLISH, inside the release image, on CPU, "
        "under fake tensors — the same pass that already owns "
        "`_vendor.torchcg.discovery`'s export. A serving pod never reaches it: "
        "the only entry is `gen-worker release derive` (cli/release.py), and "
        "no boot or dispatch path imports `gen_worker.release`. Serialization "
        "is not inductor orchestration — it holds no GIL-bound compile."
    ),
    "serving.mint_child": (
        "pgw#1371's runtime-mint COMPILE CHILD. Its `torch.export.load` + "
        "`Engine.compile` are the whole point of the module, and it is a "
        "process entry (`python -m gen_worker.serving.mint_child`) spawned "
        "one-per-graph by `serving.mint._child_compiler` — the serving parent "
        "supervises it and never imports it. Structural, not a promise: "
        "`serving/__init__.py` does not import it, `serving/mint.py` names it "
        "only as an argv string to a subprocess, and the parent's whole "
        "interface to it is an exit status plus an artifact path on disk. "
        "The split is why a wedged compile is KILLABLE at all, which is what "
        "makes the progress guard's condemnation actionable."
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
}

SUPERVISED: Dict[str, str] = {
    "aot_compile_pool": (
        "the K-wide pool. `mint_supervisor` constructs and drives it on the "
        "serving process (in a worker thread); it spawns, reaps and collects, "
        "and it must never itself export or compile."
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
