#!/usr/bin/env python3
"""CONVERSION IS UPSTREAM OF COMPUTE — compiled graphs never key on layout.

Every byte the endpoint loader observes is already in one of the slot's
declared layouts.  Conversion completes before materialization into the
worker's snapshot tree, so neither TCG's graph-class declaration nor its
runtime identity may observe layout demand, conversion, or provenance.

Two structural fences enforce that boundary:

1. Modules that call TCG's canonical declaration/runtime/identity producers,
   or the worker's sole ``tcg_graph_class_spec`` bridge, are discovered from
   their imports and calls.  In those modules the demand/conversion vocabulary
   is forbidden.  TCG remains the only identity implementation; this script
   contains no key arithmetic.
2. The layout compatibility relation contains no concrete contract handle.
   Contract contents may evolve, but the relation never gains per-format
   branches.

Run::

    python scripts/lint_cell_key_layout_fence.py
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "gen_worker"

# These are TCG 0.4's public inputs to graph declaration and compiled-graph
# identity.  The symbol guard resolves them against the installed TCG package,
# so an API rename cannot leave this fence green and vacuous.
TCG_PRODUCERS: Tuple[str, ...] = (
    "CompiledGraphKey",
    "GraphClassDeclaration",
    "GraphClassSpec",
    "RuntimeCompatibility",
    "from_artifact_metadata",
    "from_axes",
    "toolchain_axis_digest",
)
TCG_MODULE_PREFIX = "torch_compiled_graphs"

# The worker has one translation from an exported row into GraphClassSpec.
# Both mint and boot must call it rather than restating declaration inputs.
WORKER_DECLARATION_BRIDGES: Tuple[str, ...] = ("tcg_graph_class_spec",)

# The demand/conversion half of the layout vocabulary.  Decoder census symbols
# are deliberately absent: which decoders a wheel contains is a legitimate
# toolchain fact, while which layout a slot requests is not.
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

BANNED_MODULES: Tuple[str, ...] = (
    "gen_worker.convert.layout_converters",
    "convert.layout_converters",
    "layout_converters",
)

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
    return sorted(root.rglob("*.py"))


def _root_name(node: ast.AST) -> str:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _producer_calls(tree: ast.AST) -> Set[str]:
    """Canonical producer symbols called by one module.

    Imports are resolved first so aliases remain visible.  Merely importing a
    producer does not fence a consumer that only handles a typed value; the
    boundary is where declaration or identity inputs are actually assembled.
    """

    producers = set(TCG_PRODUCERS)
    imported: Dict[str, str] = {}
    modules: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == TCG_MODULE_PREFIX or module.startswith(TCG_MODULE_PREFIX + "."):
                for alias in node.names:
                    local = alias.asname or alias.name
                    if alias.name in producers:
                        imported[local] = alias.name
                    else:
                        modules[local] = f"{module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == TCG_MODULE_PREFIX or alias.name.startswith(
                    TCG_MODULE_PREFIX + "."
                ):
                    modules[alias.asname or alias.name.split(".", 1)[0]] = alias.name

    hits: Set[str] = set()
    bridges = set(WORKER_DECLARATION_BRIDGES)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            if function.id in imported:
                hits.add(imported[function.id])
            elif function.id in bridges:
                hits.add(function.id)
        elif isinstance(function, ast.Attribute):
            if function.attr in bridges:
                hits.add(function.attr)
            elif function.attr in producers and _root_name(function) in modules:
                hits.add(function.attr)
    return hits


def fenced_modules(root: Path = SRC) -> Dict[Path, str]:
    """Modules that assemble TCG declaration or identity inputs."""

    fenced: Dict[Path, str] = {}
    for path in _iter_modules(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        hits = sorted(_producer_calls(tree))
        if hits:
            fenced[path] = f"calls canonical compiled-graph producer(s): {', '.join(hits)}"
    return fenced


def _docstring_nodes(tree: ast.AST) -> Set[int]:
    """Every docstring constant by identity; prose is not a reference."""

    out: Set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            out.add(id(body[0].value))
    return out


def _violations(path: Path) -> List[Tuple[int, str]]:
    """Layout-vocabulary references in one compiled-graph producer module."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    banned = set(BANNED_NAMES)
    word = re.compile(r"\b(" + "|".join(re.escape(name) for name in BANNED_NAMES) + r")\b")
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
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            found = word.search(node.value)
            if node.value in BANNED_MODULES:
                hits.append((node.lineno, f"string {node.value!r}"))
            elif found and any(module in node.value for module in BANNED_MODULES):
                hits.append((node.lineno, f"string {node.value!r}"))
            elif found:
                hits.append((node.lineno, f"string names {found.group(1)!r}"))
    return sorted(set(hits))


def _relation_handle_literals(
    path: Path = RELATION_MODULE,
) -> Tuple[List[Tuple[int, str]], Set[str]]:
    """Concrete handles and missing named functions in the layout relation."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: Set[str] = set()
    hits: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in RELATION_FUNCTIONS:
            continue
        found.add(node.name)
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Constant)
                and isinstance(inner.value, str)
                and _HANDLE_LITERAL.fullmatch(inner.value)
            ):
                hits.append((inner.lineno, f"{node.name}: {inner.value!r}"))
    return sorted(set(hits)), set(RELATION_FUNCTIONS) - found


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=SRC,
        help=(
            "package root to sweep; retained so CI's exact entry point can be "
            "red-proved against a synthetic violating tree"
        ),
    )
    args = parser.parse_args(argv)
    failures: List[str] = []

    fenced = fenced_modules(args.src)
    if not fenced:
        print(
            "FAIL: no TCG declaration or identity producer is fenced; "
            "the producer discovery is stale"
        )
        return 1
    print(f"fence 1: {len(fenced)} compiled-graph producer module(s)")
    for path, why in sorted(fenced.items()):
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        hits = _violations(path)
        if hits:
            for line, what in hits:
                failures.append(f"{rel}:{line}: reads the layout vocabulary ({what})")
            print(f"  [FAIL] {rel} — {why}")
        else:
            print(f"  [ok]   {rel} — {why}")

    literals, missing = _relation_handle_literals()
    print(f"fence 2: {len(RELATION_FUNCTIONS)} relation function(s)")
    if missing:
        failures.append(f"layout relation fence names missing functions: {sorted(missing)!r}")
        print(f"  [FAIL] missing relation functions: {', '.join(sorted(missing))}")
    elif literals:
        for line, what in literals:
            failures.append(
                f"{RELATION_MODULE.relative_to(REPO)}:{line}: "
                f"the compatibility relation names a format ({what})"
            )
        print("  [FAIL] a handle literal reached the relation")
    else:
        print("  [ok]   no contract-handle literal in the relation")

    if failures:
        print(f"\nconversion-before-compute fence BROKEN ({len(failures)} violation(s)):")
        for failure in failures:
            print(f"  - {failure}")
        print(
            "\nConversion is upstream of compute. Move layout handling before "
            "materialization; never widen or locally restate compiled-graph identity."
        )
        return 1
    print("\nconversion-before-compute fence holds: compiled graphs never key on layout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
