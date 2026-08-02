#!/usr/bin/env python3
"""pgw#849 guard 2 — public surface in the AOT/mint/serve packages that NOTHING
in production reaches.

This program's dominant defect class is *wiring*, not logic: correct code, green
unit tests, no production caller. ``entry_workers(peak_rss_bytes=…)`` handled a
measurement no caller ever passed, so RAM was divided by a 3 GiB constant on
every mint ever run. ``aot_serve.set_guard_failure_callback`` was never called,
so every AOT arm ever built was unadvertisable, for weeks. A unit test is
structurally blind to both, because the unit test IS the caller the production
path is not.

So: parse the tree, take every public callable and public keyword parameter
defined in the guarded modules, and subtract everything a NON-TEST file
reaches. What is left is surface only tests touch.

RESOLUTION, not name matching. A module-level function in another module can
only be reached through an import, so the scan resolves ``from .x import y``
and ``import x as m`` bindings and asks whether the DEFINITION was reached —
not whether some unrelated symbol shares its name. That distinction is
load-bearing: ``compile_cache.build`` looks reached to a name matcher because
``warmup``, ``mint_child`` and ``export_contract`` each define their own local
``build``. Methods keep the coarse attribute-name test (a receiver's type is
not knowable from the AST), which makes the method half conservative — it
under-reports rather than crying wolf.

WHAT THIS GUARD DOES NOT CATCH, stated plainly: ``podguard.attend()`` returned
``lease.attend(api)``, which CONSTRUCTS a Keeper and never starts it — only
``__enter__`` calls ``start()``. The call site exists and reads correctly; what
is missing is that the returned object is never entered. No reachability
analysis sees that, and this guard does not pretend to. It catches the
"nothing calls it" half of the class, not the "called but never driven" half.

Scope (``GUARDED``) is deliberately the aot / mint / serve / arm modules where
the eleven instances landed. Whole-package scope was measured and is noisier for
no extra yield (pgw#849). This is not a general dead-code linter and must not
grow into one.

Usage::

    python scripts/lint_unreached_surface.py             # gate: guarded scope
    python scripts/lint_unreached_surface.py --params    # + keyword params
    python scripts/lint_unreached_surface.py --all       # whole package, report
    python scripts/lint_unreached_surface.py --explain   # show why exempted
"""

from __future__ import annotations

import argparse
import ast
import builtins
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
PKG = SRC / "gen_worker"

# The POD's production path — the only reach that clears a finding. `tests/`
# and `tests_v2/` are absent on purpose; that absence is the entire point.
POD_ROOTS = (SRC,)

# Operator harnesses. A symbol only these reach is still not on the pod's
# production path, so it is reported — annotated, not cleared. Calling a
# benchmark script "production" is how `entry_workers(peak_rss_bytes=…)` would
# have been argued away.
TOOL_ROOTS = (REPO / "scripts", REPO / "examples", REPO / "agents",
              REPO / "benchmarks")

# This file names hundreds of identifiers in string literals; counting itself
# as reach would exempt half the tree.
SELF = Path(__file__).resolve()

# The RATCHET. Every entry is a live hit measured on 2026-08-01 (pgw#849) and
# recorded rather than silently tolerated: the guard fails on anything NEW, and
# equally on an entry that has since been wired up and not removed. A baseline
# nobody has to shrink is a baseline nobody reads.
BASELINE = REPO / "scripts" / "unreached_surface_baseline.txt"

# The guarded scope: the modules carrying the mint / arm / serve path.
GUARDED = re.compile(
    r"^gen_worker\.(aot_.*|mint_.*|forge|executor|lifecycle|worker|hot_swap"
    r"|boot_phases|guard_closure|fleet_cells|local_cells|cell_key"
    r"|compile_cache|preload|numerics_ladder)$")

# The authored-worker API. Its callers are endpoint repos that vendor the
# wheel; this tree is not their call site and never will be.
EXEMPT_PACKAGES = ("gen_worker.api",)

# Decorators meaning "something else calls this, by table not by name".
DYNAMIC_DECORATORS = {
    "endpoint", "worker_function", "property", "cached_property", "setter",
    "deleter", "overload", "abstractmethod", "singledispatch", "register",
    "contextmanager", "asynccontextmanager", "staticmethod", "classmethod",
    "fixture", "hookimpl", "atexit", "lru_cache", "cache",
}

# Method names the stdlib / runtime protocols call for you.
PROTOCOL_METHODS = {
    "run", "close", "start", "stop", "write", "read", "flush", "emit",
    "keys", "items", "values", "get", "next", "send", "throw",
}

SKIP_PARAMS = {"self", "cls"}


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

@dataclass
class Definition:
    module: str
    qual: str                     # "Class.method" or "func"
    name: str
    path: Path
    line: int
    is_method: bool
    decorators: Tuple[str, ...]
    params: Tuple[str, ...]       # keyword-passable, positional order
    kwonly: Tuple[str, ...]
    alias_of: Tuple[str, ...]     # private globals a one-line accessor returns
    alias_calls: Tuple[str, ...]  # everything a one-line accessor delegates to
    doc: str = ""

    @property
    def target(self) -> str:
        return f"{self.module}.{self.qual}"


def module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _decorators(node: ast.AST) -> Tuple[str, ...]:
    out: List[str] = []
    for d in getattr(node, "decorator_list", []):
        cur = d.func if isinstance(d, ast.Call) else d
        if isinstance(cur, ast.Name):
            out.append(cur.id)
        elif isinstance(cur, ast.Attribute):
            out.append(cur.attr)
    return tuple(out)


def _alias_of(fn: ast.AST) -> Tuple[str, ...]:
    """If the whole body is ``return <expr over private state>``, the private
    names that expression reads. Such a function is a READ ALIAS: if the state
    it exposes is read directly elsewhere in production, a missing caller means
    duplicated surface, not unwired behavior. Measured: this is the only
    false-positive class the guard produced on the real tree."""
    body = [s for s in fn.body if not (
        isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return ()
    privates = {
        n.id for n in ast.walk(body[0].value) if isinstance(n, ast.Name)
        and n.id.startswith("_")
    } | {
        n.attr for n in ast.walk(body[0].value) if isinstance(n, ast.Attribute)
        and n.attr.startswith("_")
    }
    return tuple(sorted(privates))


def _alias_calls(fn: ast.AST) -> Tuple[str, ...]:
    """If the whole body is a single ``return``, every callable it delegates
    to. A one-line wrapper over machinery production already calls is
    duplicated surface, not unwired behavior — measured, and the second of the
    two false-positive classes on the real tree."""
    body = [s for s in fn.body if not (
        isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return ()
    out: Set[str] = set()
    for n in ast.walk(body[0].value):
        if isinstance(n, ast.Call):
            fnode = n.func
            if isinstance(fnode, ast.Name):
                out.add(fnode.id)
            elif isinstance(fnode, ast.Attribute):
                out.add(fnode.attr)
    # Builtins are not "machinery production already calls" — `return
    # int(budget_bytes)` is a brand-new entrypoint, not a wrapper. Measured:
    # without this filter the guard exempted its own red arm.
    return tuple(sorted(out - set(dir(builtins))))


def collect_definitions(paths: List[Path]) -> List[Definition]:
    out: List[Definition] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        mod = module_name(path)

        def visit(node: ast.AST, prefix: str, is_method: bool) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    visit(child, f"{prefix}{child.name}.", True)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("_"):
                        continue
                    a = child.args
                    pos = [p.arg for p in (*a.posonlyargs, *a.args)
                           if p.arg not in SKIP_PARAMS]
                    kwonly = [p.arg for p in a.kwonlyargs]
                    out.append(Definition(
                        module=mod, qual=f"{prefix}{child.name}", name=child.name,
                        path=path, line=child.lineno, is_method=is_method,
                        decorators=_decorators(child),
                        params=tuple(p for p in pos if not p.startswith("_")),
                        kwonly=tuple(p for p in kwonly if not p.startswith("_")),
                        alias_of=_alias_of(child),
                        alias_calls=_alias_calls(child),
                        doc=ast.get_docstring(child) or ""))
                elif isinstance(child, (ast.If, ast.Try)):
                    visit(child, prefix, is_method)

        visit(tree, "", False)
    return out


# ---------------------------------------------------------------------------
# reach
# ---------------------------------------------------------------------------

@dataclass
class Reach:
    qualified: Set[str] = field(default_factory=set)        # "mod.func"
    local: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    attrs: Set[str] = field(default_factory=set)            # any `.name`
    strings: Set[str] = field(default_factory=set)          # getattr/dispatch
    private_reads: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set))           # mod -> {_name}
    kwargs: Set[Tuple[str, str]] = field(default_factory=set)
    arity: Dict[str, int] = field(default_factory=dict)
    splat: Set[str] = field(default_factory=set)


class _FileScan(ast.NodeVisitor):
    """Resolve import bindings, then record what this file reaches."""

    def __init__(self, mod: str, reach: Reach) -> None:
        self.mod = mod
        self.pkg = mod.rsplit(".", 1)[0] if "." in mod else mod
        self.reach = reach
        self.direct: Dict[str, str] = {}     # local name -> "mod.attr"
        self.modules: Dict[str, str] = {}    # local name -> "mod"
        self.all_strings: Set[str] = set()

    # -- imports ----------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self.modules[a.asname or a.name.split(".")[0]] = a.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = node.module or ""
        if node.level:
            parts = self.mod.split(".")
            root = ".".join(parts[: max(len(parts) - node.level + 1, 1)]) \
                if node.level == 1 else ".".join(parts[: max(len(parts) - node.level, 1)])
            # level 1 = current package (drop the module component)
            root = ".".join(self.mod.split(".")[:-node.level]) or self.mod.split(".")[0]
            base = f"{root}.{base}" if base else root
        for a in node.names:
            local = a.asname or a.name
            self.direct[local] = f"{base}.{a.name}"
            self.modules.setdefault(local, f"{base}.{a.name}")
        self.generic_visit(node)

    # -- references -------------------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.direct:
            self.reach.qualified.add(self.direct[node.id])
        else:
            self.reach.local[self.mod].add(node.id)
        if node.id.startswith("_"):
            self.reach.private_reads[self.mod].add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.reach.attrs.add(node.attr)
        if node.attr.startswith("_"):
            self.reach.private_reads[self.mod].add(node.attr)
        owner = node.value
        if isinstance(owner, ast.Name) and owner.id in self.modules:
            self.reach.qualified.add(f"{self.modules[owner.id]}.{node.attr}")
        self.generic_visit(node)

    # NOTE: a bare identifier-shaped string literal is NOT reach. Half the
    # tree names some symbol in a log line or an activity kind; treating that
    # as a call site is how `numerics_ladder.gate` looked wired while nothing
    # in `src/` imports the module at all. Only a genuine dynamic lookup
    # counts — see `visit_Call`.

    def visit_Assign(self, node: ast.Assign) -> None:
        # `__all__` entries are re-exports, not call sites.
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            return
        self.generic_visit(node)

    DYNAMIC_LOOKUPS = {"getattr", "setattr", "hasattr", "attrgetter",
                       "methodcaller", "import_module"}

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        fname = fn.id if isinstance(fn, ast.Name) else \
            fn.attr if isinstance(fn, ast.Attribute) else ""
        if fname in self.DYNAMIC_LOOKUPS:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value.isidentifier():
                    self.all_strings.add(arg.value)
        keys: List[str] = []
        if isinstance(fn, ast.Name):
            keys.append(self.direct.get(fn.id, fn.id))
            keys.append(fn.id)
        elif isinstance(fn, ast.Attribute):
            keys.append(fn.attr)
            owner = fn.value
            if isinstance(owner, ast.Name) and owner.id in self.modules:
                keys.append(f"{self.modules[owner.id]}.{fn.attr}")
        positional = sum(1 for a in node.args if not isinstance(a, ast.Starred))
        starred = any(isinstance(a, ast.Starred) for a in node.args)
        for key in keys:
            self.reach.arity[key] = max(self.reach.arity.get(key, 0), positional)
            if starred:
                self.reach.splat.add(key)
            for kw in node.keywords:
                if kw.arg is None:
                    self.reach.splat.add(key)
                else:
                    self.reach.kwargs.add((key, kw.arg))
        self.generic_visit(node)


def collect_reach(paths: List[Path]) -> Reach:
    reach = Reach()
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        mod = module_name(path) if _under(path, SRC) else f"_script.{path.stem}"
        scan = _FileScan(mod, reach)
        scan.visit(tree)
        reach.strings |= scan.all_strings
    return reach


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def py_files(roots) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        if root.exists():
            out.extend(p for p in root.rglob("*.py")
                       if "__pycache__" not in p.parts
                       and p.name != "conftest.py"
                       and p.resolve() != SELF)
    return sorted(out)


def published_names() -> Set[str]:
    names: Set[str] = set()
    tree = ast.parse((PKG / "__init__.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__" and \
                        isinstance(node.value, (ast.List, ast.Tuple)):
                    names.update(e.value for e in node.value.elts
                                 if isinstance(e, ast.Constant)
                                 and isinstance(e.value, str))
    return names


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    kind: str
    label: str
    path: Path
    line: int
    note: str = ""


def exempt(d: Definition, published: Set[str], reach: Reach) -> Optional[str]:
    if any(d.module == p or d.module.startswith(p + ".") for p in EXEMPT_PACKAGES):
        return "authored-worker API (consumers are endpoint repos)"
    if d.name in published:
        return "re-exported from gen_worker/__init__.py"
    if set(d.decorators) & DYNAMIC_DECORATORS:
        return f"dynamic dispatch via @{sorted(set(d.decorators) & DYNAMIC_DECORATORS)[0]}"
    if d.is_method and d.name in PROTOCOL_METHODS:
        return "runtime/stdlib protocol method"
    if d.name in reach.strings:
        return "named by a string (getattr / dispatch table)"
    if d.name.endswith(("_for_tests", "_for_test")):
        return "name declares it as test-only surface"
    low = d.doc.lower()
    if any(m in low for m in ("(tests", "(test ", "test teardown",
                             "tests /", "tests,", "for tests")):
        return "docstring declares it as test / diagnostic surface"
    if d.alias_of and all(
            any(p in reads for reads in reach.private_reads.values())
            for p in d.alias_of):
        return (f"read alias of {', '.join(d.alias_of)}, which production "
                f"reads directly")
    if d.alias_calls and all(
            c in reach.attrs or any(c in n for n in reach.local.values())
            for c in d.alias_calls):
        return (f"one-line wrapper over {', '.join(d.alias_calls)}, which "
                f"production already calls")
    return None


def reached(d: Definition, reach: Reach) -> bool:
    if d.is_method:
        # A receiver's type is not knowable from the AST: fall back to the
        # attribute name. Conservative — under-reports, never cries wolf.
        return d.name in reach.attrs
    return (d.target in reach.qualified
            or d.name in reach.local.get(d.module, set()))


def run(scope_all: bool, want_params: bool, explain: bool) -> List[Finding]:
    src_files = py_files([PKG])
    defs = [d for d in collect_definitions(src_files)
            if scope_all or GUARDED.match(d.module)]
    reach = collect_reach(py_files(POD_ROOTS))
    tools = collect_reach(py_files(TOOL_ROOTS))
    published = published_names()

    findings: List[Finding] = []
    for d in defs:
        why = exempt(d, published, reach)
        if why:
            if explain:
                print(f"  exempt {d.target}: {why}", file=sys.stderr)
            continue
        if not reached(d, reach):
            note = ("reached only by an operator script — not the pod's "
                    "production path") if reached(d, tools) else ""
            findings.append(
                Finding("callable", f"{d.target}()", d.path, d.line, note))
            continue
        if not want_params:
            continue
        # Parameters of a callable that IS reached.
        keys = [d.target, d.name] if not d.is_method else [d.name]
        if any(k in reach.splat for k in keys):
            continue
        max_pos = max((reach.arity.get(k, 0) for k in keys), default=0)
        if d.is_method:
            max_pos = max(max_pos - 1, 0)
        for idx, p in enumerate(d.params):
            if idx < max_pos or any((k, p) in reach.kwargs for k in keys):
                continue
            findings.append(Finding("param", f"{d.target}({p}=…)", d.path, d.line))
        for p in d.kwonly:
            if not any((k, p) in reach.kwargs for k in keys):
                findings.append(Finding("param", f"{d.target}({p}=…)", d.path, d.line))
    return findings




# ---------------------------------------------------------------------------
# The INERT-DECLARATION check (pgw#849, added after the twelfth instance).
#
# The twelfth instance is a DIFFERENT SHAPE and the reachability scan above is
# blind to it: z-image declares ``warm_changes_key=True``, and that field is
# validated, keyed and recorded — six production readers, a perfectly healthy
# call-site count — while nothing PERFORMS the pre-warm. Used everywhere except
# the one place it must act.
#
# So classify each read of a declared field by what the reading site DOES with
# it. A field whose every read is validate / key / record is INERT: the fleet
# has been told a fact, has hashed it into the cell identity, and has written it
# into metadata, and no behaviour anywhere differs because of it.
#
# The heuristic is stated so it can be argued with: a read is an ACT when it
# steers control flow (an if/while/ternary/comprehension test) or is handed to a
# call that is not a recorder. It is a RECORD when it lands in a metadata dict
# or a setdefault. It is VALIDATE when it is read inside the declaration API
# itself. Anything this misclassifies is a false negative, not a false alarm.
# ---------------------------------------------------------------------------

DECLARATION_MODULES = ("gen_worker.api.decorators", "gen_worker.api.export_contract")
RECORD_SINKS = {"specialization", "metadata", "meta", "axes", "record", "row",
                "fields", "payload", "details", "facts", "out", "info"}


def declaration_fields() -> Dict[str, Path]:
    """Declared field name -> the file that declares it."""
    out: Dict[str, Path] = {}
    for mod in DECLARATION_MODULES:
        path = SRC / (mod.replace(".", "/") + ".py")
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and \
                        isinstance(stmt.target, ast.Name) and \
                        not stmt.target.id.startswith("_"):
                    out.setdefault(stmt.target.id, path)
    return out


def _all_read_roles(fields: Set[str]) -> Dict[str, Dict[str, List[str]]]:
    """One pass over the package: field -> role -> read sites."""
    roles: Dict[str, Dict[str, List[str]]] = {
        f: defaultdict(list) for f in fields}
    for path in py_files([PKG]):
        mod = module_name(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        parents: Dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for node in ast.iter_child_nodes(parent):
                parents[id(node)] = parent
        for hit in ast.walk(tree):
            if not (isinstance(hit, ast.Attribute) and hit.attr in fields
                    and isinstance(hit.ctx, ast.Load)):
                continue
            field = hit.attr
            where = f"{path.relative_to(REPO)}:{hit.lineno}"
            if mod in DECLARATION_MODULES:
                roles[field]["validate"].append(where)
                continue
            role = "act"
            cur: Any = hit
            for _ in range(6):
                parent = parents.get(id(cur))
                if parent is None:
                    break
                if isinstance(parent, (ast.If, ast.While, ast.IfExp)) and \
                        _contains(parent.test, hit):
                    role = "act"
                    break
                if isinstance(parent, ast.comprehension):
                    role = "act"
                    break
                if isinstance(parent, ast.Call):
                    fn = parent.func
                    name = fn.attr if isinstance(fn, ast.Attribute) else \
                        fn.id if isinstance(fn, ast.Name) else ""
                    role = "record" if name in (
                        "setdefault", "update", "append", "add") else "act"
                    break
                if isinstance(parent, ast.Assign):
                    for tgt in parent.targets:
                        if isinstance(tgt, ast.Subscript) and \
                                isinstance(tgt.value, ast.Name) and \
                                tgt.value.id in RECORD_SINKS:
                            role = "record"
                    break
                if isinstance(parent, (ast.keyword, ast.Dict)):
                    role = "record"
                    break
                cur = parent
            roles[field][role].append(where)
    return roles


def _contains(node: ast.AST, needle: ast.AST) -> bool:
    return any(n is needle for n in ast.walk(node))


def inert_declarations() -> List[Tuple[str, Dict[str, List[str]]]]:
    fields = set(declaration_fields())
    every = _all_read_roles(fields)
    out = []
    for field in sorted(fields):
        roles = every[field]
        outside = sum(len(v) for k, v in roles.items() if k != "validate")
        if outside and not roles.get("act"):
            out.append((field, dict(roles)))
    return out



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="scan all of gen_worker (report only, never a gate)")
    ap.add_argument("--params", action="store_true",
                    help="also flag keyword parameters with no caller")
    ap.add_argument("--explain", action="store_true",
                    help="print every exemption and its reason")
    ap.add_argument("--inert-declarations", action="store_true",
                    help="declared fields that are keyed and recorded but "
                         "steer nothing (the pgw#849 twelfth-instance shape)")
    ap.add_argument("--write-baseline", action="store_true",
                    help="rewrite the ratchet file from the current tree")
    args = ap.parse_args()

    if args.inert_declarations:
        rows = inert_declarations()
        for field, roles in rows:
            reads = ", ".join(f"{k}={len(v)}" for k, v in sorted(roles.items()))
            print(f"INERT declaration: {field} — {reads}")
            for role in ("key", "record"):
                for where in roles.get(role, [])[:3]:
                    print(f"    {role}: {where}")
        print(f"\n{len(rows)} inert declared field(s)", file=sys.stderr)
        return 0

    findings = run(args.all, args.params, args.explain)
    for f in sorted(findings, key=lambda f: (str(f.path), f.line, f.label)):
        tail = f"  [{f.note}]" if f.note else ""
        print(f"{f.path.relative_to(REPO)}:{f.line}: [{f.kind}] "
              f"{f.label} — no non-test call site{tail}")

    scope = "gen_worker" if args.all else "aot/mint/serve"
    kinds = "callables+params" if args.params else "callables"
    print(f"\npgw#849 guard 2: {len(findings)} {kinds} unreached in {scope}",
          file=sys.stderr)
    if args.all or args.params:
        return 0                      # report modes, never a gate

    current = {f.label for f in findings}
    if args.write_baseline:
        BASELINE.write_text("\n".join(sorted(current)) + "\n", encoding="utf-8")
        print(f"wrote {len(current)} entries to {BASELINE}", file=sys.stderr)
        return 0

    known = {ln.strip() for ln in BASELINE.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")}
    new = sorted(current - known)
    stale = sorted(known - current)
    for label in new:
        print(f"NEW unreached public surface: {label}\n"
              f"  Nothing on the pod's production path calls this. This is the "
              f"program's dominant defect class (pgw#849) — wire it, delete it, "
              f"or, if its caller is genuinely outside this tree, add an "
              f"exemption WITH A REASON in {SELF.name}.", file=sys.stderr)
    for label in stale:
        print(f"STALE baseline entry: {label}\n"
              f"  It now has a production caller. Remove the line from "
              f"{BASELINE.name} in the same commit — the ratchet only turns "
              f"one way if somebody turns it.", file=sys.stderr)
    return 1 if (new or stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())
