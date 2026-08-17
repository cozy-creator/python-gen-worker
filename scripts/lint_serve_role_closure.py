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
            # `*OTHER_TUPLE` splices another module-level declaration, which is
            # how one set states that it CONTAINS another without retyping its
            # rows (pgw#824's drift rule). Still literal-only: the splice
            # resolves by reading that tuple out of this same file.
            if isinstance(element, ast.Starred):
                if not isinstance(element.value, ast.Name):
                    raise SystemExit(
                        f"{where}: {name} splices a non-name; a spliced set "
                        f"must be another module-level tuple in this file.")
                out.extend(_declared_tuple(element.value.id, source=source, where=where))
                continue
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


def _type_checking_only(tree: ast.AST) -> Set[int]:
    """Every import node inside an ``if TYPE_CHECKING:`` body.

    Those never execute. Every module in this repo carries
    ``from __future__ import annotations``, so an annotation is a string and the
    block exists for the type checker alone — which is exactly why a generated
    family binding puts ``from torch import Tensor`` there. Following such an
    edge would make this fence report imports a serve process cannot perform,
    and reporting a library that is never loaded is how a guard loses its
    readers.

    Only the ``if`` body is skipped, never the ``else``: an ``else`` branch of a
    ``TYPE_CHECKING`` test is the RUNTIME branch.
    """
    skipped: Set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        name = (
            test.id
            if isinstance(test, ast.Name)
            else test.attr if isinstance(test, ast.Attribute) else ""
        )
        if name != "TYPE_CHECKING":
            continue
        for statement in node.body:
            for inner in ast.walk(statement):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    skipped.add(id(inner))
    return skipped


def _guarded(tree: ast.AST) -> Set[int]:
    """Every import node lexically inside a ``try`` that catches ImportError.

    An ``except ImportError`` around an import is a module SAYING it works
    without the thing — pgw#1339's degrade-loudly-and-serve, which is exactly
    how a generated family binding exposes an optional ``SPEC``. Following such
    an edge would drag every declaration into the serve closure and make this
    fence guard nothing, so the walk stops there and
    :data:`role.OPTIONAL_SERVE_IMPORTS` enumerates where it is allowed to stop.

    Only ``ImportError``/``ModuleNotFoundError`` count. A bare ``except`` or an
    ``except Exception`` is NOT a declaration that the import is optional; it is
    a module swallowing everything, and treating it as a hatch would let any
    ``try:`` walk through this fence.
    """
    optional: Set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        names: Set[str] = set()
        for handler in node.handlers:
            for caught in (
                handler.type.elts
                if isinstance(handler.type, ast.Tuple)
                else [handler.type]
            ):
                if isinstance(caught, ast.Name):
                    names.add(caught.id)
        if not names & {"ImportError", "ModuleNotFoundError"}:
            continue
        for statement in node.body:
            for inner in ast.walk(statement):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    optional.add(id(inner))
    return optional


#: Parsed-import memo, keyed by (path, module). The model-bearing ledger walks
#: one closure per declared row, and every walk re-parses the same ~150 files;
#: memoizing takes the whole fence from 21s to under 3s, which is the
#: difference between it fitting the required-gate budget and not. Results are
#: read-only everywhere they are used.
_IMPORT_MEMO: Dict[Tuple[str, str], Tuple[Set[str], Set[str], Set[str]]] = {}


def _imports(path: Path, module: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """What this file imports: (required gen_worker, guarded gen_worker, libraries).

    ``libraries`` is every non-``gen_worker`` top-level package the file names,
    guarded or not — a forbidden library behind a ``try`` is still a library
    this process would acquire whenever it is installed, which on an
    eager-capable pod is always.
    """
    memo = _IMPORT_MEMO.get((str(path), module))
    if memo is not None:
        return memo
    package = _package_of(module)
    tree = ast.parse(path.read_text(), filename=str(path))
    optional_nodes = _guarded(tree)
    unexecuted = _type_checking_only(tree)
    required: Set[str] = set()
    guarded: Set[str] = set()
    libraries: Set[str] = set()
    for node in ast.walk(tree):
        if id(node) in unexecuted:
            continue
        if isinstance(node, ast.Import):
            found = {alias.name for alias in node.names}
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
            found = {prefix} | {f"{prefix}.{alias.name}" for alias in node.names}
        else:
            continue
        bucket = guarded if id(node) in optional_nodes else required
        for name in found:
            if name.startswith("gen_worker"):
                bucket.add(name)
            elif name:
                libraries.add(name.split(".", 1)[0])
    _IMPORT_MEMO[(str(path), module)] = (required, guarded, libraries)
    return required, guarded, libraries


def closure(
    roots: Sequence[str],
) -> Tuple[Set[str], Dict[str, str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Walk the REQUIRED closure, and record what it stopped at.

    Returns ``(modules, who-reached-each, libraries-per-module,
    guarded-edges-per-module)``. Guarded edges are recorded but not followed —
    see :func:`_guarded`.
    """
    seen: Set[str] = set()
    via: Dict[str, str] = {root: "<root>" for root in roots}
    libraries: Dict[str, Set[str]] = {}
    guarded: Dict[str, Set[str]] = {}
    queue: List[str] = list(roots)
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        path = _module_path(name)
        if path is None:
            continue
        seen.add(name)
        required, optional, used = _imports(path, name)
        libraries[name] = used
        if optional:
            guarded[name] = optional
        for target in sorted(required):
            if target in seen:
                continue
            via.setdefault(target, name)
            queue.append(target)
    return seen, via, libraries, guarded


def _chain(name: str, via: Dict[str, str]) -> str:
    hops = [name]
    cursor = name
    while via.get(cursor) not in (None, "<root>"):
        cursor = via[cursor]
        hops.append(cursor)
    return " <- ".join(hops)


def check(roots: Sequence[str], banned: Sequence[str]) -> List[str]:
    """pgw#1328: the adopt-only serve role cannot reach the mint lane."""
    problems: List[str] = []
    for name in (*roots, *banned):
        if _module_path(name) is None:
            problems.append(
                f"`{name}` is declared in serve/role.py but is not a module "
                f"under src/gen_worker. A fence naming a module that does not "
                f"exist passes vacuously forever (pgw#1176).")
    if problems:
        return problems
    seen, via, _, _ = closure(roots)
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


def check_model_free(
    roots: Sequence[str],
    libraries: Sequence[str],
    optional: Sequence[str],
    within: Sequence[str] = (),
) -> List[str]:
    """pgw#1331: this surface reaches no model library, and stops nowhere else.

    A SEPARATE walk from :func:`check`, rooted at the subset the claim is about.
    Rooting it at the whole serve role would report the eager-capable worker's
    own guts — which import diffusers inside functions and are entitled to —
    and a fence that reports things nobody can fix is a fence people learn to
    ignore. The scope is declared in ``role.MODEL_FREE_MODULES``, with the
    reason on the tuple.
    """
    problems: List[str] = []
    for name in (*roots, *optional):
        if _module_path(name) is None:
            problems.append(
                f"`{name}` is declared in serve/role.py but is not a module "
                f"under src/gen_worker. A fence naming a module that does not "
                f"exist passes vacuously forever (pgw#1176).")
    outside = sorted(set(roots) - set(within)) if within else []
    if outside:
        problems.append(
            f"role.MODEL_FREE_MODULES names {outside[0]}, which is not in "
            f"SERVE_ROLE_MODULES. A module asserted model-free but not "
            f"asserted MINT-free is checked for the smaller of the two "
            f"properties — splice the sets rather than keeping two lists.")
    if problems:
        return problems
    seen, via, used, guarded = closure(roots)
    if not seen:
        return ["the model-free roots resolved to NOTHING — this guard is "
                "guarding nothing"]
    forbidden = set(libraries)
    for module in sorted(seen):
        for library in sorted(used.get(module, set()) & forbidden):
            problems.append(
                f"pgw#1331: the MODEL-FREE serve surface imports {library!r} in "
                f"{module} (reached via {_chain(module, via)}). The serve path "
                f"calls graph classes and bare math; a model library on it is "
                f"seconds of process start and gigabytes of host RAM bought to "
                f"run reshapes. Move the model code to the family's DECLARATION "
                f"half, or take the module out of SERVE_ROLE_MODULES and mean it.")

    # …and the ONLY places the walk is allowed to stop are the enumerated ones.
    allowed = set(optional)
    stopped: Set[str] = set()
    for module in sorted(guarded):
        if module not in seen:
            continue
        for target in sorted(guarded[module]):
            if _module_path(target) is None:
                continue  # a `from pkg import name` leaf, not a module
            if target in seen:
                continue  # already required by another edge; nothing is hidden
            stopped.add(target)
            if target not in allowed:
                problems.append(
                    f"pgw#1331: {module} reaches {target} through an "
                    f"ImportError-guarded import, which this fence does not "
                    f"follow. That hatch is a CLOSED list: add {target!r} to "
                    f"role.OPTIONAL_SERVE_IMPORTS with a reason, or make the "
                    f"import unguarded so the closure walks it.")
    for target in sorted(allowed - stopped):
        problems.append(
            f"pgw#1331: role.OPTIONAL_SERVE_IMPORTS names {target}, but no "
            f"serve-role module reaches it through a guarded import. An "
            f"enumerated hatch nobody uses is a hatch nobody is checking "
            f"(pgw#1176) — delete the row.")
    return problems


def check_bearing_ledger(
    bearing: Sequence[str],
    free: Sequence[str],
    libraries: Sequence[str],
) -> List[str]:
    """pgw#1331: every module EXCUSED from the model-free claim still needs it.

    ``MODEL_BEARING_SERVE_MODULES`` is the residue of the model-free cut, and
    an unchecked residue is the exact shape pgw#1176 measured rotting: prose
    naming an owed cut, green forever, describing a tree that has moved. So the
    list is asserted TRUE in both directions.

    * A row that no longer reaches a forbidden library goes RED here. The only
      way to make it green is to move it into ``MODEL_FREE_MODULES``, where the
      real walk then holds it. **The list can only shrink.**
    * A row that is also in ``MODEL_FREE_MODULES`` goes red: a module cannot
      both be asserted model-free and be excused from the assertion.
    """
    problems: List[str] = []
    overlap = sorted(set(bearing) & set(free))
    if overlap:
        problems.append(
            f"pgw#1331: {overlap[0]} is in BOTH role.MODEL_FREE_MODULES and "
            f"role.MODEL_BEARING_SERVE_MODULES. One module, one claim — the "
            f"model-free walk would assert it while the ledger excuses it.")
    forbidden = set(libraries)
    for name in bearing:
        if _module_path(name) is None:
            problems.append(
                f"`{name}` is declared in serve/role.py but is not a module "
                f"under src/gen_worker. A fence naming a module that does not "
                f"exist passes vacuously forever (pgw#1176).")
            continue
        seen, _, used, _ = closure((name,))
        if not any(used.get(module, set()) & forbidden for module in seen):
            problems.append(
                f"pgw#1331: role.MODEL_BEARING_SERVE_MODULES names {name}, but "
                f"its closure no longer reaches any of {sorted(forbidden)}. "
                f"That cut has LANDED — move {name!r} into "
                f"role.MODEL_FREE_MODULES so the real walk holds it. This "
                f"ledger only ever shrinks; it is not an allowlist.")
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
    libraries = _declared_tuple("FORBIDDEN_LIBRARIES")
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
    # pgw#1331's half: a family DECLARATION is a model-library module, so
    # rooting the walk there must produce a forbidden-library violation. The
    # declaration reaches diffusers exclusively through FUNCTION-LOCAL imports,
    # so this proves the library check follows lazy imports too — the same
    # shape the mint-lane half is proven against, one layer up.
    declaration = "gen_worker.model.catalog.flux1_dev"
    library_problems = [
        line
        for line in check_model_free((declaration,), libraries, ())
        if "imports" in line
    ]
    if not library_problems:
        print(
            f"SELFTEST FAILED: rooting the walk at {declaration} produced no "
            "forbidden-library violation, but that module's build callables "
            "exist to construct diffusers and transformers modules. Either the "
            "walk stopped following function-local imports — and this fence is "
            "now blind — or the declaration no longer names a model library, "
            "which would mean the catalog stopped declaring anything.",
            file=sys.stderr)
        return 1
    # …and a guarded edge to an unlisted module must be refused, or the hatch
    # is not a list, it is a door.
    binding = "gen_worker.model.catalog._generated.flux1_dev"
    if not [line for line in check_model_free((binding,), libraries, ()) if "guarded" in line]:
        print(
            f"SELFTEST FAILED: {binding} reaches its declaration through an "
            "ImportError-guarded import, and an EMPTY OPTIONAL_SERVE_IMPORTS "
            "accepted it. The hatch would then be open to any `try: import`.",
            file=sys.stderr)
        return 1
    # …and the model-bearing LEDGER must go red in both of its directions, or
    # the residue of the model-free cut is an allowlist that rots (pgw#1176).
    # `serve.role` itself is model-FREE, so naming it as bearing must be
    # refused — that is the arm which forces a landed cut to be recorded.
    if not check_bearing_ledger(("gen_worker.serve.role",), (), libraries):
        print(
            "SELFTEST FAILED: a model-FREE module was accepted into "
            "MODEL_BEARING_SERVE_MODULES. The ledger would then be an "
            "allowlist that never shrinks, which is the prose it replaced.",
            file=sys.stderr)
        return 1
    if not check_bearing_ledger((declaration,), (declaration,), libraries):
        print(
            "SELFTEST FAILED: a module claimed BOTH model-free and "
            "model-bearing was accepted. One module, one claim.",
            file=sys.stderr)
        return 1
    print(
        f"pgw#1328/#1331 selftest: the fence goes red as designed "
        f"({len(problems)} violation(s) from a mint-reaching root, "
        f"{len(library_problems)} from a model-library root, an unlisted "
        f"guarded edge is refused, a landed cut left in the bearing ledger is "
        f"refused, a double-claimed module is refused, and a computed "
        f"declaration is refused)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    roots = _declared_tuple("SERVE_ROLE_MODULES")
    banned = _declared_tuple("MINT_MACHINERY")
    model_free = _declared_tuple("MODEL_FREE_MODULES")
    bearing = _declared_tuple("MODEL_BEARING_SERVE_MODULES")
    libraries = _declared_tuple("FORBIDDEN_LIBRARIES")
    optional = _declared_tuple("OPTIONAL_SERVE_IMPORTS")
    problems = check(roots, banned)
    problems.extend(check_model_free(model_free, libraries, optional, within=roots))
    problems.extend(check_bearing_ledger(bearing, model_free, libraries))
    for line in problems:
        print(line, file=sys.stderr)
    if problems:
        print(
            f"\n{len(problems)} adopt-only serve-role violation(s).",
            file=sys.stderr)
        return 1
    seen, _, _, _ = closure(roots)
    free, _, _, _ = closure(model_free)
    print(
        f"pgw#1328: the adopt-only serve role is mint-free ({len(roots)} "
        f"declared root(s), {len(seen)} gen_worker modules in the closure, "
        f"{len(banned)} mint modules absent)")
    print(
        f"pgw#1331: the model-free serve surface holds no model library "
        f"({len(model_free)} of {len(roots)} declared root(s), {len(free)} "
        f"gen_worker modules in the closure, {len(libraries)} libraries "
        f"absent, {len(optional)} enumerated guarded edge(s))")
    print(
        f"pgw#1331: the model-bearing ledger is exact ({len(bearing)} root(s) "
        f"named as still reaching a model library, every one of them verified "
        f"to still reach one — the list only shrinks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
