#!/usr/bin/env python3
"""pgw#1328: THE ADOPT-ONLY SERVE ROLE MAY NOT REACH THE MINT LANE.

``executor.py`` is both hosts in one class — it serves tenants AND drives
``mint_supervisor`` — and nothing enforced the separation, so "this pod cannot
compile" was a claim about intent. This is the claim as a build gate.

The rule: starting from every module in
:data:`gen_worker.serve.role.SERVE_ROLE_MODULES`, the transitive STATIC import
closure inside ``gen_worker`` must not contain any module in
:data:`gen_worker.serve.role.MINT_MACHINERY`. **Function-local imports count**
— a lazy ``from . import aot_mint`` inside a method is exactly the shape a
re-coupling takes, and an import-time-only fence would miss it. That is not
hypothetical: it is how ``fleet_cells`` reached the mint lane before this
landed.

THE TWO LISTS ARE READ OUT OF THE SOURCE, NOT RETYPED HERE
-----------------------------------------------------------
pgw#1176's measured finding: this repo shipped a fence naming a deleted symbol
in its own string literals, green the whole time, guarding nothing. So this
file parses ``src/gen_worker/serve/role.py`` for the two tuples rather than
holding a second copy. Rename a module and the role declaration is the only
place that has to change; delete one and this goes red at the staleness check
below instead of passing vacuously.

WHAT THIS SUPERSEDES
--------------------
``scripts/lint_serve_keyset_closure.py`` (pgw#1327), whose own docstring
nominated this file as its replacement. That fence rooted at
``boot_adopt``/``keyset`` and banned three tracer modules; this one roots at
those AND the arm, the dispatch and the role itself, and bans nine. It is a
strict superset in both directions, which is the only honest reason to delete a
guard.

RUNTIME IS THE OTHER HALF, AND IT IS NOT THIS
----------------------------------------------
This reads the tree, so it cannot see ``importlib.import_module`` on a computed
name, and it cannot see the tenant's own endpoint modules at all — they are not
in this repo. :mod:`gen_worker.serve.guard` covers those at run time. Neither
half subsumes the other.

Run::

    python scripts/lint_serve_role_closure.py
    python scripts/lint_serve_role_closure.py --selftest
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "gen_worker"
ROLE_FILE = SRC / "serve" / "role.py"


def _declared_tuple(name: str, *, source: str = "", where: Path = ROLE_FILE) -> Tuple[str, ...]:
    """Read a module-level tuple-of-string-literals out of the role module.

    Literals only. A computed membership list would put the role's own
    definition somewhere this fence cannot read, which is the dual-declaration
    hazard one level up (pgw#1143's `Slot(layouts=)` rule, same argument).
    """
    tree = ast.parse(source or where.read_text(), filename=str(where))
    for node in tree.body:
        targets: Sequence[ast.expr]
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        if not isinstance(value, ast.Tuple):
            raise SystemExit(
                f"{where}: {name} is not a tuple literal — this fence reads it "
                f"statically so the role and the guard cannot disagree.")
        out: List[str] = []
        for element in value.elts:
            if not isinstance(element, ast.Constant) or not isinstance(
                    element.value, str):
                raise SystemExit(
                    f"{where}: {name} holds a non-literal entry; every module "
                    f"name must be readable without importing the tree.")
            out.append(element.value)
        return tuple(out)
    raise SystemExit(
        f"{where}: no module-level {name} — the fence's roots come from the "
        f"role's own declaration and it is gone.")


def _rel(name: str) -> str:
    return name[len("gen_worker"):].lstrip(".").replace(".", "/")


def _is_package(name: str) -> bool:
    return (SRC / _rel(name) / "__init__.py").is_file()


def _module_path(name: str) -> Path | None:
    if not name.startswith("gen_worker"):
        return None
    rel = _rel(name)
    for candidate in (SRC / rel / "__init__.py", SRC / f"{rel}.py"):
        if candidate.is_file():
            return candidate
    return None


def _package_of(module: str) -> str:
    if _is_package(module):
        return module
    return module.rsplit(".", 1)[0]


def _imports(path: Path, module: str) -> Set[str]:
    """Every ``gen_worker`` module this file names, relative imports resolved."""
    package = _package_of(module)
    out: Set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("gen_worker"):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                if node.level > 1:
                    parts = parts[: -(node.level - 1)]
                prefix = ".".join(parts)
                if node.module:
                    prefix = f"{prefix}.{node.module}"
            else:
                prefix = node.module or ""
            if not prefix.startswith("gen_worker"):
                continue
            out.add(prefix)
            for alias in node.names:
                out.add(f"{prefix}.{alias.name}")
    return out


def closure(roots: Sequence[str]) -> Tuple[Set[str], Dict[str, str]]:
    """Every gen_worker module the roots reach, and who first reached it."""
    seen: Set[str] = set()
    via: Dict[str, str] = {root: "<root>" for root in roots}
    queue: List[str] = list(roots)
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        path = _module_path(name)
        if path is None:
            continue
        seen.add(name)
        for target in sorted(_imports(path, name)):
            if target in seen:
                continue
            via.setdefault(target, name)
            queue.append(target)
    return seen, via


def _chain(name: str, via: Dict[str, str]) -> str:
    hops = [name]
    cursor = name
    while via.get(cursor) not in (None, "<root>"):
        cursor = via[cursor]
        hops.append(cursor)
    return " <- ".join(hops)


def check(roots: Sequence[str], banned: Sequence[str]) -> List[str]:
    problems: List[str] = []
    for name in (*roots, *banned):
        if _module_path(name) is None:
            problems.append(
                f"`{name}` is declared in serve/role.py but is not a module "
                f"under src/gen_worker. A fence naming a module that does not "
                f"exist passes vacuously forever (pgw#1176).")
    if problems:
        return problems
    seen, via = closure(roots)
    if not seen:
        return ["the serve-role roots resolved to NOTHING — this guard is "
                "guarding nothing"]
    for name in banned:
        if name in seen:
            problems.append(
                f"pgw#1328: the ADOPT-ONLY serve role reaches {name} via "
                f"{_chain(name, via)}. That role adopts by key and refuses or "
                f"routes on a miss (§4.28/§4.29) — it must not be able to "
                f"compile. Put the call behind gen_worker.serve.mint_seam, or "
                f"take the module out of SERVE_ROLE_MODULES and mean it.")
    return problems


_SELFTEST_ROLE = '''
SERVE_ROLE_MODULES = ("gen_worker.serve.role",)
MINT_MACHINERY = ("gen_worker.aot_mint",)
'''


def selftest() -> int:
    """Prove the fence goes RED, on a root that really does reach the lane.

    A guard whose green has never been contrasted with a red is a guard nobody
    has tested. The violating root is ``gen_worker.mint_adapter`` — the
    EAGER-CAPABLE side of the seam, whose whole job is to name the mint lane,
    and which names it exclusively through FUNCTION-LOCAL imports. So this one
    root proves both halves at once: that the walk fires, and that it still
    follows lazy imports, which is the shape the coupling actually took (it is
    how ``fleet_cells`` reached ``mint_supervisor`` before this landed).
    """
    banned = _declared_tuple("MINT_MACHINERY")
    problems = check(("gen_worker.mint_adapter",), banned)
    if not problems:
        print(
            "SELFTEST FAILED: rooting the walk at gen_worker.mint_adapter "
            "produced no violation, but that module exists to import "
            "mint_supervisor and aot_mint inside its methods. Either the walk "
            "stopped following function-local imports — and this fence is now "
            "blind — or the adapter no longer bridges to the lane at all.",
            file=sys.stderr)
        return 1
    # …and the literal reader must refuse a role file it cannot read
    # statically, or the roots could silently become computed.
    try:
        _declared_tuple("SERVE_ROLE_MODULES", source="SERVE_ROLE_MODULES = list(x)")
    except SystemExit:
        pass
    else:
        print(
            "SELFTEST FAILED: a non-literal SERVE_ROLE_MODULES was accepted. "
            "The fence would then be reading something other than the role.",
            file=sys.stderr)
        return 1
    print(
        f"pgw#1328 selftest: the fence goes red as designed "
        f"({len(problems)} violation(s) from a mint-reaching root, and a "
        f"computed declaration is refused)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    roots = _declared_tuple("SERVE_ROLE_MODULES")
    banned = _declared_tuple("MINT_MACHINERY")
    problems = check(roots, banned)
    for line in problems:
        print(line, file=sys.stderr)
    if problems:
        print(
            f"\n{len(problems)} adopt-only serve-role violation(s).",
            file=sys.stderr)
        return 1
    seen, _ = closure(roots)
    print(
        f"pgw#1328: the adopt-only serve role is mint-free "
        f"({len(roots)} declared root(s), {len(seen)} gen_worker modules in "
        f"the closure, {len(banned)} mint modules absent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
