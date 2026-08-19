#!/usr/bin/env python3
"""CONVERSION IS UPSTREAM OF COMPUTE — no compiled graph re-keys.

    Every byte the endpoint's loader/`setup()` observes is in one of the slot's
    DECLARED accepted layouts. Conversion completes strictly before
    materialization into the worker's snapshot tree, and the endpoint has no code
    path that can observe the source layout, the conversion, or its provenance.
    Therefore the traced graph, `Compile.contract_axes()`, the ck1 compiled graph key and
    the declared envelope are unchanged by any conversion, and no compiled graph re-keys —
    ever.

Two fences, both structural:

**FENCE 1 — no compiled graph-key axis reads the layout vocabulary.** The fenced module
set is DERIVED, not hand-listed: any module that calls a compiled graph-key axis producer
(`compiled_graph_key.from_axes`, `envelope_digest`, `toolchain_axis_digest`,
`facts_digest`, `from_entry_metadata`, or constructs `CompiledGraphKey`) is
in it, plus `compiled_graph_key` itself. A new module that starts computing a key joins
the fence automatically — the failure mode of a hand-maintained list is that the
one file that violates the rule is the one nobody added.

Inside the fence, referencing the DEMAND / CONVERSION half of the layout
vocabulary is red: the `layout_converters` module, `Slot.layouts`, `LayoutId`,
the planner, the verdict, the provenance. Whether by import, by attribute, or by
a string naming the module — a deferred `importlib.import_module` is the obvious
way around an import check, so strings count.

Deliberately NOT banned: `@implements_contract` and `ContractDecoder`. Those are
the DECODER CENSUS half of the same module — "which decoders does this image
contain" — which is a property of the wheel that legitimately reaches the
toolchain closure. The banned set is the demand/conversion half.

**FENCE 2 — the compatibility RELATION holds no format knowledge.** The
extensibility invariant: contract CONTENTS evolve; the compatibility RELATION
never does. So the functions that compute the rung must contain zero literal
contract handles. A handle literal appearing there is the first line of a
per-format special case; it is also the shape a similarity heuristic arrives in.

Run::

    python scripts/lint_compiled_graph_key_layout_fence.py
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lint_scope import is_unowned  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "gen_worker"

#: Calling one of these MAKES a module a compiled graph-key axis producer. Narrow on
#: purpose: `facts_digest` and `subject_digest` are generic canonical-digest
#: helpers whose callers say so (`contract_facts`: "NOT a key-axis input";
#: `subject_digest` names a MINT OBLIGATION, never a compiled graph key), so probing on
#: them would fence modules that compute no key and make the gate noise.
# A fence that names a DELETED symbol guards nothing and passes vacuously
# forever. After ANY rename, re-read every fence and ask what symbol it names
# NOW.
#
# `envelope_digest` is deliberately absent: the envelope is a MANIFEST fact,
# not a key axis, so fencing on it would fence modules that compute no key —
# exactly the noise the note above says to avoid.
AXIS_PRODUCERS: Tuple[str, ...] = (
    # pgw#1277 moved this derivation out to torchcg.identity,
    # so no module in src/gen_worker calls it TODAY. The entry stays on
    # purpose: a worker calling tcg_identity.from_axes() would be computing
    # a key axis, which is exactly what this fence covers.
    # fence-symbol-exempt: lives in TCG now; kept to fence the next call site
    "from_axes",
    "toolchain_axis_digest",
    "from_artifact_metadata",
    "CompiledGraphKey",
)

#: Always fenced, whatever they call: they DEFINE the key or one of its THREE
#: axis inputs (graph / sm / toolchain). The envelope is not one of them.
#:
#: pgw#1277: the KEY's own definition left this tree for
#: ``torchcg.identity``, so it has no seed entry here any more.
#: ``graph_facts.py`` deliberately does NOT replace it — it defines the
#: declared envelope and the coverage label, and neither is a key axis.
FENCE_SEED: Tuple[str, ...] = (
    "aot_mint.py",
    "env_seal.py",
    "host_isa.py",
    "guard_closure.py",
    "compile_cache.py",
)

#: The DEMAND / CONVERSION half of the layout vocabulary. Names.
BANNED_NAMES: Tuple[str, ...] = (
    "LayoutId",
    "LayoutRung",
    "LayoutVerdict",
    "LayoutProduction",
    "ConversionPlan",
    "ConversionHop",
    "ConversionResult",
    "TopologyConversion",
    "QuantRepack",
    "classify_layout",
    "plan_layout_conversions",
    "run_layout_conversion",
    "conversion_provenance",
    "derived_artifact_identity",
    "registered_layout_conversions",
    "registered_layout_productions",
    "normalize_layout_demand",
    "validate_layout_handle",
    "parse_layout_id",
    "known_contracts",
    "layouts",
    "CONVERSION_PROVENANCE_KEY",
)

#: Module paths that must not be reachable from a fenced module, including via
#: a string handed to importlib.
BANNED_MODULES: Tuple[str, ...] = (
    "gen_worker.convert.layout_converters",
    "convert.layout_converters",
    "layout_converters",
)

#: The functions that COMPUTE the rung. Fence 2 applies to their bodies.
RELATION_MODULE = SRC / "convert" / "layout_converters.py"
RELATION_FUNCTIONS: Tuple[str, ...] = (
    "classify_layout",
    "plan_layout_conversions",
    "_shortest_chain",
    "_axis_satisfied",
    "_reachable_productions",
    "_unevaluated",
)

_HANDLE_LITERAL = re.compile(r"^[a-z0-9]+\.[a-z0-9][a-z0-9._-]*@[1-9][0-9]*$")


def _iter_modules(root: Path) -> List[Path]:
    # pgw#1332: skip the subtrees that are not ours to change. `_vendor` is the
    # one that matters here and it is a REAL finding shape, not a nuisance:
    # torchcg's `recipe.py` computes a compiled graph-key axis (`GraphSpecializationVariant.key`)
    # AND reads `.layouts` — but that `layouts` is the TRACED-LAYOUT axis of a
    # graph-specialization row (torchcg G15), a different concept that happens to share
    # the word with §1.33's demand/conversion vocabulary. torchcg says so
    # explicitly: it RECORDS the token and never enumerates, interprets or forks
    # §1.32/§1.33. Either way the snapshot is byte-identical to a pinned
    # upstream rev under a digest fence, so a finding in it is unfixable here by
    # construction — which is exactly what `_lint_scope` exists to say once.
    return sorted(
        path for path in root.rglob("*.py") if not is_unowned(path, root)
    )


def _called_names(tree: ast.AST) -> Set[str]:
    """Every callee's last name component in one module."""
    out: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            out.add(func.attr)
        elif isinstance(func, ast.Name):
            out.add(func.id)
    return out


def fenced_modules(root: Path = SRC) -> Dict[Path, str]:
    """Modules the fence covers, mapped to WHY they are covered."""
    out: Dict[Path, str] = {}
    for path in _iter_modules(root):
        if path.name in FENCE_SEED and path.parent == root:
            out[path] = "defines the compiled graph key or one of its axis inputs"
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        hits = sorted(_called_names(tree) & set(AXIS_PRODUCERS))
        if hits:
            out[path] = f"computes a compiled graph-key axis ({', '.join(hits)})"
    return out


def _docstring_nodes(tree: ast.AST) -> Set[int]:
    """Every docstring Constant, by id.

    PROSE IS NOT A REFERENCE. A gate that reds on the word appearing in a
    comment teaches lanes to avoid explaining themselves.
    """
    out: Set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            out.add(id(body[0].value))
    return out


def _violations(path: Path) -> List[Tuple[int, str]]:
    """Layout-vocabulary references in one fenced module."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    banned = set(BANNED_NAMES)
    word = re.compile(r"\b(" + "|".join(re.escape(n) for n in BANNED_NAMES) + r")\b")
    docstrings = _docstring_nodes(tree)
    hits: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in BANNED_MODULES:
                    hits.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in BANNED_MODULES or module.endswith(".layout_converters"):
                hits.append((node.lineno, f"from {module} import ..."))
            for alias in node.names:
                if alias.name in banned:
                    hits.append((node.lineno, f"from {module} import {alias.name}"))
        elif isinstance(node, ast.Attribute) and node.attr in banned:
            hits.append((node.lineno, f".{node.attr}"))
        elif isinstance(node, ast.Name) and node.id in banned:
            hits.append((node.lineno, node.id))
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            # Non-docstring strings count: `getattr(slot, "layouts")`, a
            # deferred `import_module("...layout_converters")` and a stringized
            # annotation are the three ways around a name/import check.
            found = word.search(node.value)
            if node.value in BANNED_MODULES:
                hits.append((node.lineno, f"string {node.value!r}"))
            elif found and any(m in node.value for m in BANNED_MODULES):
                hits.append((node.lineno, f"string {node.value!r}"))
            elif found:
                hits.append((node.lineno, f"string names {found.group(1)!r}"))
    return sorted(set(hits))


def _relation_handle_literals() -> List[Tuple[int, str]]:
    """Contract-handle literals inside the rung computation (fence 2)."""
    tree = ast.parse(RELATION_MODULE.read_text(encoding="utf-8"))
    hits: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in RELATION_FUNCTIONS:
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Constant)
                    and isinstance(inner.value, str)
                    and _HANDLE_LITERAL.match(inner.value)):
                hits.append((inner.lineno, f"{node.name}: {inner.value!r}"))
    return sorted(set(hits))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path, default=SRC,
        help="package root to sweep (default: src/gen_worker). Exists so the "
             "gate's own RED can be executed against a tree that violates it — "
             "a fence nobody has watched fail is a fence nobody knows works.")
    args = parser.parse_args(argv)
    failures: List[str] = []

    fenced = fenced_modules(args.src)
    if not fenced:
        print("FAIL: the fence covers NO module — the axis-producer probe is "
              "stale, which makes this gate green for the wrong reason")
        return 1
    print(f"fence 1: {len(fenced)} compiled graph-key module(s)")
    for path, why in sorted(fenced.items()):
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        hits = _violations(path)
        if hits:
            for line, what in hits:
                failures.append(f"{rel}:{line}: reads the layout vocabulary ({what})")
            print(f"  [FAIL] {rel} — {why}")
        else:
            print(f"  [ok]   {rel} — {why}")

    literals = _relation_handle_literals()
    print(f"fence 2: {len(RELATION_FUNCTIONS)} relation function(s)")
    if literals:
        for line, what in literals:
            failures.append(
                f"{RELATION_MODULE.relative_to(REPO)}:{line}: the compatibility "
                f"RELATION names a format ({what})")
        print("  [FAIL] a handle literal reached the relation")
    else:
        print("  [ok]   no contract-handle literal in the relation")

    if failures:
        print(f"\n§1.33 point 5 fence BROKEN ({len(failures)} violation(s)):")
        for line in failures:
            print(f"  - {line}")
        print(
            "\nConversion is UPSTREAM of compute. If a compiled graph-key axis needs to "
            "know the layout, the conversion is happening too late — move it "
            "ahead of materialization instead of widening the key.")
        return 1
    print("\n§1.33 point 5 fence holds: no compiled graph re-keys on a conversion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
