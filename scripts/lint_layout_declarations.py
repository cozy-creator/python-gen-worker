#!/usr/bin/env python3
"""§1.33: `Slot(layouts=...)` is READABLE WITHOUT IMPORTING.

The per-slot DEMAND is the one fact the hub's layout gate reads out of the
manifest, so it has to be legible to BOTH paths that ever look at it:

* the IMPORT path — `gen_worker.discovery` imports endpoint modules with heavy
  deps stubbed (`discovery/heavy_deps.py`: "that metadata is torch-free by
  design"), and
* the AST path — a source sweep, and this script.

A computed declaration (a comprehension, an f-string, a dict built by a
helper, a value read from config) is invisible to the second. It would make
the published manifest the only place the demand can be read — a
dual-declaration hazard — and it is unreviewable in a diff, which is where a
layout demand is actually judged.

So: `layouts=` must be a DICT LITERAL whose keys are string literals and whose
values are tuple/list literals of string literals or of names imported from
`gen_worker.models.tensor_layout_contract`. Nothing else.

This is a fence, not a taste rule. It carries no allowlist by design: there is
no declaration this refuses that could not be written literally instead.

Usage:

    python scripts/lint_layout_declarations.py [PATH ...]

Defaults to this repository's `src/`. An ENDPOINT repo runs the same script
against its own tree — the constraint is on the declaration, not on where it
lives.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterator, List, Set, Tuple

REPO = Path(__file__).resolve().parents[1]

#: `src/` only, deliberately. The subject is a declaration that gets
#: PUBLISHED — an endpoint's `Slot(layouts=...)` reaching a release manifest.
#: `tests/` holds the opposite by construction: a test proving the constructor
#: refuses a computed declaration has to WRITE one, and sweeping it would make
#: this fence and its own negative test mutually exclusive. An endpoint repo
#: runs this against its own tree, where every declaration is a real one.
DEFAULT_ROOTS = (REPO / "src",)

#: The module whose module-level constants may stand in for a handle literal.
#: It is the SDK's transcription of tensorhub's registry, so a name imported
#: from it resolves to a handle string the lint can still classify by NAME.
VOCABULARY_MODULE = "gen_worker.models.tensor_layout_contract"


def _iter_python_files(roots: Tuple[Path, ...]) -> Iterator[Path]:
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            yield path


def _vocabulary_names(tree: ast.Module) -> Set[str]:
    """Local names bound to constants of the vocabulary module."""
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == VOCABULARY_MODULE:
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == VOCABULARY_MODULE:
                    names.add(alias.asname or alias.name.split(".")[0])
    return names


def _handle_is_readable(node: ast.expr, vocabulary: Set[str]) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.Name):
        return node.id in vocabulary
    if isinstance(node, ast.Attribute):
        # `tensor_layout_contract.CONTRACT_PLAIN_BF16` — the module itself was
        # imported under a name the sweep can resolve.
        root: ast.expr = node
        while isinstance(root, ast.Attribute):
            root = root.value
        return isinstance(root, ast.Name) and root.id in vocabulary
    return False


def _check_layouts(
    path: Path, call: ast.Call, value: ast.expr, vocabulary: Set[str],
) -> List[str]:
    where = f"{path}:{value.lineno}"
    if not isinstance(value, ast.Dict):
        return [
            f"{where}: layouts= is a {type(value).__name__}, not a dict "
            "literal — the AST sweep cannot read it, so the hub's copy would "
            "be the only place this demand can be read"
        ]
    problems: List[str] = []
    for key, item in zip(value.keys, value.values):
        if key is None:
            problems.append(
                f"{where}: layouts= splats another mapping (**); a splat hides "
                "the component paths from the sweep")
            continue
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            problems.append(
                f"{path}:{key.lineno}: layouts= key is not a string literal")
            continue
        if not isinstance(item, (ast.Tuple, ast.List)):
            problems.append(
                f"{path}:{item.lineno}: layouts[{key.value!r}] is a "
                f"{type(item).__name__}, not a tuple literal — order is "
                "preference and a computed sequence has no reviewable order")
            continue
        for element in item.elts:
            if not _handle_is_readable(element, vocabulary):
                problems.append(
                    f"{path}:{element.lineno}: layouts[{key.value!r}] holds a "
                    f"{type(element).__name__}; every handle must be a string "
                    f"literal or a constant imported from {VOCABULARY_MODULE}")
    return problems


def _check_undeclarable(path: Path, value: ast.expr) -> List[str]:
    """The escape is reviewed in the diff like any other declaration, so its
    REASON has to be a literal too — a computed one is unreadable exactly
    where it matters."""
    if isinstance(value, ast.Constant) and isinstance(value.value, str) \
            and value.value.strip():
        return []
    return [
        f"{path}:{value.lineno}: layouts_undeclarable= must be a non-empty "
        "string literal saying which bytes this slot holds and why no "
        "registered handle names them"
    ]


def _check_requirements(
    path: Path, value: ast.expr, vocabulary: Set[str],
) -> List[str]:
    """The requirements axis is reviewed in the diff like the demand it
    guards, so it is a dict literal of handle -> compact string literal."""
    where = f"{path}:{value.lineno}"
    if not isinstance(value, ast.Dict):
        return [
            f"{where}: layout_requirements= is a {type(value).__name__}, not "
            "a dict literal — the AST sweep cannot read it"
        ]
    problems: List[str] = []
    for key, item in zip(value.keys, value.values):
        if key is None or not _handle_is_readable(key, vocabulary):
            problems.append(
                f"{where}: layout_requirements= key must be a handle literal "
                f"or a constant imported from {VOCABULARY_MODULE}")
            continue
        if not (isinstance(item, ast.Constant) and isinstance(item.value, str)):
            problems.append(
                f"{path}:{item.lineno}: layout_requirements= value must be "
                "the compact string form (e.g. \'sm100+\'); a computed "
                "requirement is unreviewable exactly where it matters")
    return problems


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def scan(path: Path) -> List[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"{path}: could not parse ({exc})"]
    vocabulary = _vocabulary_names(tree)
    problems: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "Slot":
            continue
        declared = False
        for kw in node.keywords:
            if kw.arg == "layouts":
                declared = True
                problems.extend(_check_layouts(path, node, kw.value, vocabulary))
            elif kw.arg == "layouts_undeclarable":
                declared = True
                problems.extend(_check_undeclarable(path, kw.value))
            elif kw.arg == "layout_requirements":
                problems.extend(
                    _check_requirements(path, kw.value, vocabulary))
            elif kw.arg is None:
                # `Slot(cls, **kwargs)` — a layouts= could be hiding in there.
                problems.append(
                    f"{path}:{node.lineno}: Slot(**kwargs) may carry a "
                    "layouts= the sweep cannot see; pass the declaration "
                    "explicitly")
                declared = True   # unknowable; the splat refusal above stands
        if not declared:
            problems.append(
                f"{path}:{node.lineno}: this model slot declares no consumed "
                "tensor-layout contract. A19 is a hard cut — ABSENT is a "
                "refusal, never the UNDECLARED tri-state. Write "
                'layouts={"*": ("plain.bf16@1",)}, or, if no registered '
                "handle names this slot's bytes, "
                'layouts_undeclarable="<why>".')
    return problems


def main(argv: List[str]) -> int:
    roots = tuple(Path(a).resolve() for a in argv[1:]) or DEFAULT_ROOTS
    problems: List[str] = []
    for path in _iter_python_files(roots):
        problems.extend(scan(path))
    if problems:
        print("pgw#1143 / A19: Slot layout declarations that are missing or "
              "that the AST sweep cannot read:\n", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "\nWrite the declaration literally — a dict literal of string "
            "literals mapping to tuple literals of handles. The demand is "
            "reviewed in the diff and read by the hub; both need it visible "
            "without running the module.",
            file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
