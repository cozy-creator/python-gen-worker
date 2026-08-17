#!/usr/bin/env python3
"""Public surface in `gen_worker` that NOTHING in production reaches.
Whole-package scope.

This program's dominant defect class is *wiring*, not logic: correct code, green
unit tests, no production caller. A unit test is structurally blind to it,
because the unit test IS the caller the production path is not.

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

WHAT THIS GUARD DOES NOT CATCH: a call site that exists and reads correctly
while the object it returns is never driven (e.g. a context manager that is
constructed but never entered). No reachability analysis sees that. It catches
the "nothing calls it" half of the class, not the "called but never driven"
half.

**This is still not a general dead-code linter.** It reports PUBLIC callables
only, it clears anything an operator script or an endpoint repo reaches, and
every finding is a question — wire it, delete it, or prove the caller is
elsewhere — never an instruction.

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
from typing import Any, Dict, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
PKG = SRC / "gen_worker"

# The POD's production path — the only reach that clears a finding. `tests/`
# and `tests_v2/` are absent on purpose; that absence is the entire point.
POD_ROOTS = (SRC,)

# Operator harnesses. A symbol only these reach is still not on the pod's
# production path, so it is reported — annotated, not cleared.
TOOL_ROOTS = (REPO / "scripts", REPO / "examples", REPO / "agents",
              REPO / "benchmarks")

# This file names hundreds of identifiers in string literals; counting itself
# as reach would exempt half the tree.
SELF = Path(__file__).resolve()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_scope import is_unowned  # noqa: E402

# The RATCHET. Every entry is a live hit, recorded rather than silently
# tolerated: the guard fails on anything NEW, and equally on an entry that has
# since been wired up and not removed. A baseline nobody has to shrink is a
# baseline nobody reads.
BASELINE = REPO / "scripts" / "unreached_surface_baseline.txt"

# The guarded scope: THE WHOLE PACKAGE. `gen_worker.pb` is excluded — generated
# protobuf stubs, authored by protoc.
GUARDED = re.compile(r"^gen_worker(?!\.pb($|\.))")

# Packages whose whole reason to exist is a consumer OUTSIDE this repository,
# so `src/` is structurally not their call site and never will be. This is a
# package-shaped statement, not a per-symbol pardon: every symbol in one of
# these is consumer surface by construction, which is what makes the blanket
# honest here and dishonest anywhere else.
EXEMPT_PACKAGES: Dict[str, str] = {
    # Endpoint repos vendor the wheel and call this.
    "gen_worker.api": "authored-worker API (consumers are endpoint repos)",
    # th#1947 §4.2: the cross-repo corpora and their accessor, shipped as
    # package data so a PINNED consumer reads the exact bytes it pinned.
    # tensorhub's internal/wirecontract/peers.lock is the consumer of record.
    "gen_worker.contracts": "cross-repo contract corpora (consumers are peer repos)",
    # pgw#1332: the typed Family SDK. Same class as `gen_worker.api`, and for
    # the same structural reason rather than by analogy: a family is the ONE
    # authoring contract, so its declaration vocabulary, its generated bindings
    # and its instance surface exist to be called from an endpoint repo. `src/`
    # reaching them would mean the worker runtime imports the catalog, which is
    # the thing pgw#1328's adopt-only role is fenced against. The seam the
    # worker DOES own is `set_instance_resolver`, and it is inside this package
    # for cohesion, not because the pod is expected to be its only caller.
    "gen_worker.family": "typed Family SDK (consumers are endpoint repos)",
}

# The same fact, one symbol at a time, WITH PROOF. `gen_worker.api` is exempt by
# package because the whole package is authored-worker surface; these are
# symbols OUTSIDE it that an endpoint repo nonetheless calls. Each row names the
# call site.
AUTHOR_SURFACE_FILE = REPO / "scripts" / "author_surface_allowlist.txt"


def author_surface() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not AUTHOR_SURFACE_FILE.exists():
        return out
    for raw in AUTHOR_SURFACE_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.split("\t") if p]
        if len(parts) < 2:
            continue
        target, site = parts[0], parts[1]
        # Rows are written the way the report prints them (`io.read_image`);
        # a Definition's target carries the package (`gen_worker.io.…`).
        if not target.startswith("gen_worker."):
            target = f"gen_worker.{target}"
        kind = parts[2] if len(parts) > 2 else "PROD"
        out[target] = f"called from {site} ({kind}) — see author_surface_allowlist.txt"
    return out


AUTHOR_SURFACE: Dict[str, str] = author_surface()

#: Set by `run`. The author-surface staleness check only means something when
#: every module was collected; in guarded scope most targets are simply absent.
SCOPE_ALL = False

#: Every label this tree currently DEFINES, set by `run`. A baseline row can go
#: stale two ways that need opposite fixes, and telling a reader the wrong one
#: sends them hunting for a caller that does not exist.
DEFINED_LABELS: Set[str] = set()

# target -> why it is unreached AND who deletes it. Each row is a DELETION
# THAT IS OWED, never a permanent pardon: the only legitimate reason to be in
# here is that something outside this lane's scope owns the removal, so every
# row names that owner. A row whose target no longer exists is RED (see
# `stale_exemptions`) — the same rule the config-reads and settings-writers
# allowlists follow, and what stops this table becoming the place dead code
# goes to be forgotten.
EXEMPT_TARGETS: dict[str, str] = {}

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
    "exec_module",
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
    #: "callable" or "type". A symbol sweep that enumerates only functions
    #: misses a dead TYPE entirely — one survived a whole deletion pass on a
    #: sibling lane and only the compiler caught it.
    kind: str = "callable"

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
    duplicated surface, not unwired behavior."""
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
    duplicated surface, not unwired behavior."""
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
    # int(budget_bytes)` is a brand-new entrypoint, not a wrapper.
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
                    if not child.name.startswith("_"):
                        out.append(Definition(
                            module=mod, qual=f"{prefix}{child.name}",
                            name=child.name, path=path, line=child.lineno,
                            is_method=False, decorators=_decorators(child),
                            params=(), kwonly=(), alias_of=(), alias_calls=(),
                            doc=ast.get_docstring(child) or "", kind="type"))
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

    def __init__(self, mod: str, reach: Reach, *, is_package: bool = False) -> None:
        self.mod = mod
        self.is_package = is_package
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
            # A relative import resolves against the importer's PACKAGE. For a
            # module that is the parent (drop one component per level); for a
            # package's own `__init__.py` the package IS `self.mod`, so one
            # fewer. Getting this wrong makes every `from . import x` inside an
            # `__init__.py` resolve to the grandparent and read as unreached.
            drop = node.level - 1 if self.is_package else node.level
            parts = self.mod.split(".")
            root = ".".join(parts[: len(parts) - drop]) if drop else self.mod
            root = root or parts[0]
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

    # NOTE: a bare identifier-shaped string literal is NOT reach. Half the tree
    # names some symbol in a log line or an activity kind, and treating that as
    # a call site makes a module nothing imports look wired. Only a genuine
    # dynamic lookup counts — see `visit_Call`.

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
        scan = _FileScan(mod, reach, is_package=path.name == "__init__.py")
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
            # `.venv` is checkout-borne, never tree content: an installed env
            # under a TOOL_ROOT (~10k files) blows the walk into a
            # RecursionError that a clean checkout never reproduces.
            out.extend(p for p in root.rglob("*.py")
                       if "__pycache__" not in p.parts
                       and ".venv" not in p.parts
                       and p.name != "conftest.py"
                       and not is_unowned(p)
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


def stale_exemptions(defs: List[Definition]) -> List[str]:
    """Rows in :data:`EXEMPT_TARGETS` naming something that no longer exists.

    An exemption is a deletion somebody OWES. Once the owner lands it the row
    is a pardon for nothing, and a pardon for nothing is how the next dead
    symbol gets waved through under a comment about a different one.
    """
    have = {d.target for d in defs}
    return sorted(set(EXEMPT_TARGETS) - have) + sorted(
        f"{t} (author_surface_allowlist.txt)"
        for t in set(AUTHOR_SURFACE) - have)


def exempt(d: Definition, published: Set[str], reach: Reach) -> Optional[str]:
    if d.target in EXEMPT_TARGETS:
        return EXEMPT_TARGETS[d.target]
    if d.target in AUTHOR_SURFACE:
        return AUTHOR_SURFACE[d.target]
    for pkg, why in EXEMPT_PACKAGES.items():
        if d.module == pkg or d.module.startswith(pkg + "."):
            return why
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
    if d.kind == "type":
        # A type is reached by CONSTRUCTION, by annotation, by subclassing, by
        # isinstance, by `except`, or by a bare name after an import — all of
        # which land as a Name or an Attribute. Deliberately permissive: this
        # half under-reports rather than crying wolf, exactly like the method
        # half above.
        return (d.target in reach.qualified
                or d.name in reach.attrs
                or any(d.name in names for names in reach.local.values()))
    if d.is_method:
        # A receiver's type is not knowable from the AST: fall back to the
        # attribute name. Conservative — under-reports, never cries wolf.
        return d.name in reach.attrs
    return (d.target in reach.qualified
            or d.name in reach.local.get(d.module, set()))


# ---------------------------------------------------------------------------
# THE CROSS-REPO CHECK
#
# The finding this whole guard produces is "no non-test call site IN THIS REPO",
# and that is NOT the claim "no consumer": a symbol whose caller is a conversion
# endpoint in a sibling repo, or that backs a PRICED hub product
# (`e2e/manifests/pricing.yaml`), has a consumer even when no code in any repo
# imports it.
#
# CI has none of these checkouts, which is why the answer is a checked-in file
# (`author_surface_allowlist.txt`) and this mode is the thing that MAINTAINS it.
# Run it on a developer box before any deletion:
#
#     python scripts/lint_unreached_surface.py --siblings
#
# It is RED when a finding is reached from a sibling repo and is not in the
# allowlist — i.e. exactly when somebody is about to delete a cross-repo
# consumer's dependency.
# ---------------------------------------------------------------------------

# DIRECTORY names, and this list being CLOSED is the fence. A NAMED sibling that
# is not on disk is a REFUSAL that prints the path, never a skip — the loop below
# used to `continue` past it, which is how the 2026-08-16 rename made this scan
# read 5 corpora while printing "across 6 siblings, 0 unlisted consumers"
# (pgw#1323). Renaming the entry fixed that instance; refusing fixes the class.
#
# `inference-endpoints` became `serverless-endpoints` on 2026-08-16 and
# `training-endpoints` became `jobs` on 2026-08-17; both directory moves are done.
SIBLINGS = ("jobs", "serverless-endpoints",
            "private-inference-endpoints", "tensorhub", "e2e", "cozy-local")
def _workspace() -> Path:
    """The directory the sibling repos sit in. Walk up rather than assume
    `REPO.parent`: a lane works from `~/cozy/.worktrees/<repo>/<branch>`, where
    the parent holds worktrees and no siblings at all — and a check that
    silently finds nothing is worse than no check."""
    for cand in [REPO, *REPO.parents][:6]:
        # `.worktrees/` holds a directory per repo too, so "the name exists" is
        # not enough — require an actual checkout.
        if sum((cand / s / ".git").exists() for s in SIBLINGS) >= 2:
            return cand
    return REPO.parent


WORKSPACE = _workspace()
# fence-symbol-exempt: "manifests" here is a PATH COMPONENT, not a symbol —
# the documented false-positive shape of scripts/lint_fence_symbols.py.
PRICING = WORKSPACE / "e2e" / "manifests" / "pricing.yaml"


def _sibling_texts(
    with_branches: bool,
    extra: Tuple[str, ...] = (),
    siblings: Tuple[str, ...] = (),
    workspace: Optional[Path] = None,
) -> Tuple[List[Tuple[str, str, str]], List[str], List[str]]:
    """``(corpus, scanned, refusals)``.

    ``corpus`` is ``(repo, where, text)`` over every sibling checkout, and
    optionally over the content of every REMOTE BRANCH — a call site can live on
    an unmerged branch and is a consumer just the same. ``scanned`` is the repos
    actually READ, which is the number the report prints; ``len(SIBLINGS)`` is
    the number it used to print, and the gap between them is the whole defect.

    **A named sibling that is not on disk is a REFUSAL, not a skip.** So is a
    ``git grep`` that fails for any reason other than "no match" (rc 1). This
    scan's only job is to say "nobody outside this repo calls X"; a corpus it
    silently did not read makes that sentence false while it still prints green.

    ``git grep`` rather than a filesystem walk: it reads the index, skips
    ``.venv`` / ``node_modules`` / build output for free, and turns a
    ten-minute rglob over tensorhub into about a second. Only lines mentioning
    ``gen_worker`` or a ``ctx.`` call can be a call site, so that is all we
    carry.
    """
    import subprocess
    names = siblings or SIBLINGS
    root_dir = WORKSPACE if workspace is None else workspace
    # `gen_worker` and `ctx.` alone are NOT enough: a sibling writes
    # `io.read_image(path)` on a line that mentions neither. So every finding's
    # leaf name is a pattern too — that is the whole point of the check.
    pats: List[str] = ["-e", "gen_worker", "-e", r"ctx\.[a-z_]"]
    for name in extra:
        pats += ["-e", name]
    out: List[Tuple[str, str, str]] = []
    scanned: List[str] = []
    refusals: List[str] = []
    for name in names:
        root = root_dir / name
        if not (root / ".git").exists():
            refusals.append(
                f"{name}: NO CHECKOUT at {root} — this fence certifies that NOTHING outside "
                f"this repo calls a finding, and it cannot certify a corpus it did not read. "
                f"Clone it, or remove {name!r} from SIBLINGS deliberately (the 2026-08-16/17 "
                f"repo renames are what made this list stale while it still printed green)")
            continue
        try:
            r = subprocess.run(["git", "grep", "-h", "-I", *pats], cwd=root,
                               capture_output=True, text=True, timeout=180)
        except Exception as e:
            refusals.append(f"{name}: git grep failed at {root}: {e}")
            continue
        # rc 1 is "no match", which is a real answer. Anything above it is not.
        if r.returncode > 1:
            refusals.append(
                f"{name}: git grep exited {r.returncode} at {root} "
                f"({r.stderr.strip().splitlines()[:1]}) — unread, not clean")
            continue
        scanned.append(name)
        if r.stdout:
            out.append((name, name, r.stdout))
        if not with_branches:
            continue
        try:
            heads = subprocess.run(["git", "ls-remote", "--heads", "origin"],
                                   cwd=root, capture_output=True, text=True,
                                   timeout=180).stdout.split()
        except Exception as e:
            refusals.append(
                f"{name}: --branches asked for every remote branch and ls-remote failed "
                f"at {root}: {e} — the branch half of this repo went unread")
            continue
        for ref in [h for h in heads if h.startswith("refs/heads/")]:
            br = ref[len("refs/heads/"):]
            try:
                blob = subprocess.run(
                    ["git", "grep", "-h", "-I", *pats, f"origin/{br}"],
                    cwd=root, capture_output=True, text=True,
                    timeout=180).stdout
            except Exception as e:
                refusals.append(f"{name}@{br}: git grep failed: {e} — branch unread")
                continue
            if blob:
                out.append((name, f"{name}@{br}", blob))
    return out, scanned, refusals


def siblings_selftest() -> int:
    """A fence that cannot go red proves nothing. Both arms, forced.

    This runs anywhere — it builds throwaway git repos in a temp directory and
    never touches the real siblings — so CI can prove the refusal exists even
    though CI has no checkouts to scan.
    """
    import subprocess
    import tempfile

    bad = 0
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        present = ("alpha", "beta")
        for name in present:
            d = ws / name
            d.mkdir()
            (d / "use.py").write_text("from gen_worker.io import read_image\n")
            for cmd in (["git", "init", "-q"],
                        ["git", "add", "use.py"],
                        ["git", "-c", "commit.gpgsign=false", "-c", "user.name=t",
                         "-c", "user.email=t@t", "commit", "-qm", "x"]):
                subprocess.run(cmd, cwd=d, check=True, capture_output=True)

        # ARM 1 — every named sibling present: clean, and the count is the truth.
        corpus, scanned, refusals = _sibling_texts(False, (), present, ws)
        if refusals:
            print(f"FAIL selftest all-present: expected no refusals, got {refusals}")
            bad += 1
        if sorted(scanned) != sorted(present):
            print(f"FAIL selftest all-present: scanned {scanned}, want {list(present)}")
            bad += 1
        if len(corpus) != 2:
            print(f"FAIL selftest all-present: {len(corpus)} corpora, want 2")
            bad += 1

        # ARM 2 — one named sibling absent: a REFUSAL that names the path, and a
        # `scanned` count that is smaller than the declared list. This is the
        # exact shape the 2026-08-16/17 repo renames created, and before this
        # change it was a `continue` under a printed "across 6 siblings".
        gone = ws / "jobs"
        corpus, scanned, refusals = _sibling_texts(False, (), (*present, "jobs"), ws)
        if not refusals:
            print("FAIL selftest missing-sibling: a named sibling that is not on "
                  "disk produced NO refusal — that is the silent skip this exists for")
            bad += 1
        elif not any(str(gone) in r for r in refusals):
            print(f"FAIL selftest missing-sibling: refusal does not name {gone}: {refusals}")
            bad += 1
        if len(scanned) != 2:
            print(f"FAIL selftest missing-sibling: scanned {scanned}, want 2 — the count "
                  f"printed must be what was READ, never len(SIBLINGS)")
            bad += 1

    print(f"siblings selftest: {'FAILED' if bad else 'both arms hold'} ({bad} failure(s))",
          file=sys.stderr)
    return 1 if bad else 0


def cross_repo_report(findings: List["Finding"], with_branches: bool) -> int:
    leaves = tuple(sorted({f.label.rstrip("()").split(".")[-1]
                           for f in findings}))
    corpus, scanned, refusals = _sibling_texts(with_branches, leaves)
    # A NAMED sibling this scan could not read makes its one sentence — "nothing
    # outside this repo calls X" — false. It is a refusal that names the path,
    # and it is the answer whether or not the readable half found anything.
    if refusals:
        for r in refusals:
            print(f"CROSS-REPO SCAN REFUSED: {r}", file=sys.stderr)
        print(f"\ncross-repo check REFUSED: read {len(scanned)} of {len(SIBLINGS)} named "
              f"siblings ({', '.join(scanned) or 'none'}) in {WORKSPACE} — a verdict about "
              f"consumers cannot be issued over a corpus that was not read",
              file=sys.stderr)
        return 1
    priced = PRICING.read_text(encoding="utf-8") if PRICING.exists() else ""
    missing: List[str] = []
    for f in findings:
        target = f.label.rstrip("()")
        parts = target.split(".")
        leaf, mod = parts[-1], ".".join(parts[1:-1])
        modtail = mod.split(".")[-1] if mod else ""
        # ANY gen_worker module, not just the defining one: packages re-export
        # (a sibling imports `from gen_worker.convert import Dataset` while the
        # class is defined in `gen_worker.convert.dataset`). Requiring the
        # defining path would make this check blind to every re-export, so it
        # over-matches on purpose — over-matching says "do not delete".
        pats = [rf"from gen_worker[.\w]* import [^\n]*\b{re.escape(leaf)}\b",
                rf"\bctx\.{re.escape(leaf)}\s*\("]
        if modtail:
            pats.append(rf"\b{re.escape(modtail)}\.{re.escape(leaf)}\s*\(")
        for repo, where, text in corpus:
            if any(re.search(pat, text) for pat in pats):
                if target not in AUTHOR_SURFACE:
                    missing.append(f"{target} — reached from {where}")
                break
        else:
            # A priced product function names a consumer no import shows.
            hyphen = leaf.replace("_", "-")
            if priced and re.search(rf"^\s*{re.escape(hyphen)}\s*:", priced, re.M):
                missing.append(
                    f"{target} — backs the PRICED function '{hyphen}' "
                    f"({PRICING.relative_to(WORKSPACE)})")
    for m in sorted(set(missing)):
        print(f"CROSS-REPO CONSUMER, not in author_surface_allowlist.txt: {m}",
              file=sys.stderr)
    # THE COUNT, and it is `scanned`, never `SIBLINGS`. Printing the declared
    # length while the loop had skipped one is precisely how this scan read 5
    # corpora and announced 6 (pgw#1323). They can only differ now if `refusals`
    # was empty, which the branch above guarantees — the assert says so.
    assert len(scanned) == len(SIBLINGS), (scanned, SIBLINGS)
    print(f"\ncross-repo check: {len(corpus)} corpora (files/branches) scanned across "
          f"{len(scanned)}/{len(SIBLINGS)} siblings ({', '.join(scanned)}), "
          f"{len(set(missing))} unlisted consumers", file=sys.stderr)
    return 1 if missing else 0


def run(scope_all: bool, want_params: bool, explain: bool) -> List[Finding]:
    global SCOPE_ALL, DEFINED_LABELS
    SCOPE_ALL = scope_all
    src_files = py_files([PKG])
    defs = [d for d in collect_definitions(src_files)
            if scope_all or GUARDED.match(d.module)]
    reach = collect_reach(py_files(POD_ROOTS))
    tools = collect_reach(py_files(TOOL_ROOTS))
    published = published_names()
    DEFINED_LABELS = {f"{d.target}()" for d in defs}

    findings: List[Finding] = []
    for target in stale_exemptions(defs):
        findings.append(Finding(
            "stale-exemption", target, SELF, 0,
            "EXEMPT_TARGETS names a symbol this tree no longer defines — the "
            "deletion it was waiting for has landed; drop the row"))
    for d in defs:
        why = exempt(d, published, reach)
        if why:
            if explain:
                print(f"  exempt {d.target}: {why}", file=sys.stderr)
            continue
        if not reached(d, reach):
            note = ("reached only by an operator script — not the pod's "
                    "production path") if reached(d, tools) else ""
            label = d.target if d.kind == "type" else f"{d.target}()"
            findings.append(Finding(d.kind, label, d.path, d.line, note))
            continue
        if not want_params or d.kind == "type":
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
# The INERT-DECLARATION check — REPORT MODE, never a gate.
#
# "A declared fact with no consequence" is a DIFFERENT SHAPE and the
# reachability scan above is blind to it: a per-family flag that is validated,
# keyed into the cell identity and recorded into the spec has three or more
# production readers and a perfectly healthy call-site count, while nothing
# performs the EFFECT it exists to cause. Used everywhere except the one place
# it must act, and strictly harder to see than an unused function, because
# every grep for it finds hits.
#
# This check has never had a confirmed live instance, which is why it REPORTS
# and does not gate.
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
    for field_name in sorted(fields):
        roles = every[field_name]
        outside = sum(len(v) for k, v in roles.items() if k != "validate")
        if outside and not roles.get("act"):
            out.append((field_name, dict(roles)))
    return out



def _upstream_baseline() -> Optional[Set[str]]:
    """The ratchet file as `origin/master` has it, or None if unavailable.

    Why a gate reads git: a STALE row is usually not a finding at all. Master
    moves, somebody turns the ratchet there, and every branch that predates the
    turn goes red on a row that is already gone upstream. The guard can tell —
    the row is absent from `origin/master` — so it says REBASE instead of
    letting a reader investigate a deletion that has already happened.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["git", "show", f"origin/master:{BASELINE.relative_to(REPO)}"],
            cwd=REPO, capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return {ln.partition("#")[0].strip() for ln in out.stdout.splitlines()
            if ln.strip() and not ln.startswith("#")} - {""}


def rewrite_baseline(current: Set[str]) -> None:
    """Rewrite the ratchet, PRESERVING the documented sections verbatim.

    The naive version — dump `sorted(current)` over the whole file — destroys
    the prose, and the prose is load-bearing: `baseline_rows` reads the
    OWNER/EXPIRY notes out of it, and those notes are what tell a reader that a
    row is a deletion somebody else OWES.

    So the header is kept byte-for-byte, and only rows it does NOT already list
    are appended. That second half is not a nicety: the documented sections
    contain DATA rows under their notes, so a writer that appends everything
    duplicates every documented row.
    """
    lines = BASELINE.read_text(encoding="utf-8").splitlines()
    comment_idx = [i for i, ln in enumerate(lines) if ln.startswith("#")]
    head_end = max(comment_idx) + 1 if comment_idx else 0
    header = lines[:head_end]
    documented = {ln.partition("#")[0].strip() for ln in header
                  if ln.strip() and not ln.strip().startswith("#")} - {""}
    # Inline reasons are DATA, not decoration: `gate_rows_without_reasons`
    # fails the gate on a gate-shaped row that has lost one, so a rewriter
    # that dropped them would turn every annotated gate row red.
    _, _, reasons = baseline_rows()
    tail = [label + (f"  # {reasons[label]}" if label in reasons else "")
            for label in sorted(current - documented)]
    dropped = documented - current
    BASELINE.write_text("\n".join(header) + "\n\n" + "\n".join(tail) + "\n",
                        encoding="utf-8")
    print(f"wrote {len(tail)} entries + {len(documented & current)} already "
          f"listed in the documented sections = {len(current)} to "
          f"{BASELINE.name}", file=sys.stderr)
    for label in sorted(dropped):
        print(f"  NOTE: {label} is written into a documented section but is no "
              f"longer a finding — edit that section by hand; this writer will "
              f"not touch prose.", file=sys.stderr)


#: A symbol whose LEAF NAME starts with one of these is a GATE: its whole
#: purpose is to refuse. pgw#1271 — a gate on the ratchet is categorically
#: different from an unreported metric or a duplicated helper, and the file's
#: own header has always said so ("A gate that never runs" is its first
#: severity class). The ratchet absorbed them as unannotated lines anyway, so
#: `boot_key.assert_memo_honest` — the check that made the boot-key memo safe
#: to have at all — sat here with permanent tenure while the memo produced the
#: `graph` axis of every published key with nothing checking it.
GATE_PREFIXES = ("assert_", "verify_", "refuse_", "require_", "check_")


def is_gate_label(label: str) -> bool:
    """Whether a baseline label names a gate (by its leaf symbol, not its
    module: `models.checkpoint.load` is not a gate, `x.check_slots` is)."""
    leaf = label.rstrip("()").rsplit(".", 1)[-1]
    return leaf.startswith(GATE_PREFIXES)


def baseline_rows() -> Tuple[Set[str], Dict[str, str], Dict[str, str]]:
    """The ratchet's labels, the OWNER/EXPIRY note governing each, and each
    row's own INLINE reason.

    The file writes the notes as prose above the rows they cover. Parsing
    them turns a note nobody reads into the first line of the failure that
    needs it: a row with an owner and an expiry is a deletion somebody OWES,
    and the reader who trips the gate is usually not that somebody.

    An INLINE reason is written after the label on its own line::

        gen_worker.geometry.FamilyGeometry.assert_declared()  # a test-only …

    It is required on gate-shaped rows and optional everywhere else — see
    :func:`gate_rows_without_reasons`.
    """
    known: Set[str] = set()
    notes: Dict[str, str] = {}
    reasons: Dict[str, str] = {}
    current = ""
    buf: List[str] = []
    for raw in BASELINE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            if set(body) == {"-"} or not body:
                # a separator or a blank comment ends the block's scope
                if buf:
                    current = " ".join(buf)
                    buf = []
                elif set(body) == {"-"}:
                    current = ""
                continue
            if "OWNER:" in body or "EXPIRY:" in body or buf:
                buf.append(body)
            continue
        if not line:
            continue
        if buf:
            current = " ".join(buf)
            buf = []
        label, _, inline = line.partition("#")
        label = label.strip()
        if not label:
            continue
        known.add(label)
        if inline.strip():
            reasons[label] = inline.strip()
        if current:
            notes[label] = current
    return known, notes, reasons


def gate_rows_without_reasons(
    known: Set[str], reasons: Dict[str, str],
) -> List[str]:
    """Gate-shaped baseline rows carrying no inline reason (pgw#1271).

    THE STANDING RULE: a gate that is on this list is a gate that CANNOT GO
    RED, and the ratchet's own delisting precedent (`numerics_ladder.gate`,
    wired and removed 2026-08-02) proves the correct outcome is usually to
    wire it. Absorbing it as a bare line grants it permanent tenure instead,
    and — because the reasoning around an unwired gate reads exactly like
    enforcement — nothing else in this repo will ever notice.

    So a gate row must SAY, in one line, why it is not wired. "Call from
    tests" is an acceptable reason. Silence is not: it is indistinguishable
    from a gate everyone assumes is running.
    """
    return sorted(
        label for label in known
        if is_gate_label(label) and not reasons.get(label))


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
                         "steer nothing — report only; see the header note")
    ap.add_argument("--write-baseline", action="store_true",
                    help="rewrite the ratchet file from the current tree")
    ap.add_argument("--siblings", action="store_true",
                    help="check every finding against the sibling repos and "
                         "the priced-function manifest (developer box only)")
    ap.add_argument("--branches", action="store_true",
                    help="with --siblings, also scan every REMOTE BRANCH of "
                         "each sibling (slow; a call site on an unmerged "
                         "branch is a consumer too)")
    ap.add_argument("--siblings-selftest", action="store_true",
                    help="force both arms of the sibling scan (missing sibling "
                         "-> refusal, all present -> the true count) in a temp "
                         "workspace; runs anywhere, including CI")
    args = ap.parse_args()

    if args.siblings_selftest:
        return siblings_selftest()

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
    if args.siblings:
        return cross_repo_report(findings, args.branches)
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
        rewrite_baseline(current)
        return 0

    known, notes, reasons = baseline_rows()
    unexplained = gate_rows_without_reasons(known, reasons)
    for label in unexplained:
        print(f"GATE ROW WITH NO REASON: {label}\n"
              f"  This is a GATE — its leaf name is one of "
              f"{'/'.join(GATE_PREFIXES)}* — and it is on the ratchet, which "
              f"means NOTHING IN src/ CALLS IT. A gate that cannot go red is "
              f"the one class on this list that is worse than dead code: the "
              f"reasoning around it reads like enforcement, so nothing else "
              f"will ever notice.\n"
              f"    WIRE IT   the usual outcome — see numerics_ladder.gate, "
              f"delisted 2026-08-02 by acquiring a caller;\n"
              f"    DELETE IT with its tests, if the property it guards is no "
              f"longer load-bearing;\n"
              f"    SAY WHY   append an inline reason to its line in "
              f"{BASELINE.name}:\n"
              f"                {label}  # a test-only assertion; …\n"
              f"  'Call from tests' is an acceptable reason. Silence is not.",
              file=sys.stderr)
    new = sorted(current - known)
    stale = sorted(known - current)
    for label in new:
        print(f"NEW unreached public surface: {label}\n"
              f"  Nothing in src/ reaches this. That is the program's dominant "
              f"defect class (pgw#849) and one of three things:\n"
              f"    WIRE IT    the caller is missing and should exist;\n"
              f"    DELETE IT  with its tests, never porting them (§4.34);\n"
              f"    PROVE IT   the caller is in an ENDPOINT REPO — run\n"
              f"               `python {SELF.name} --siblings` and, if it "
              f"passes, add the row WITH its call site to\n"
              f"               scripts/author_surface_allowlist.txt.\n"
              f"  'No call site in this repo' is NOT 'no consumer' (§4.34, "
              f"eb245671): a symbol behind a priced hub function has a consumer "
              f"even when no code imports it.\n"
              f"  IT MAY NOT BE YOUR CHANGE. This gate reads the MERGED tree, "
              f"so a callable whose last caller was removed by somebody else's "
              f"PR goes red on yours — and on theirs it was green. That is "
              f"invisible to both authors and it has cost this program four "
              f"blocked lanes once already. Before you touch anything, find "
              f"the change that stranded it:\n"
              f"      git log --oneline -S {label.rstrip('()').rsplit('.', 1)[-1]!r} -- src/\n"
              f"  If the answer is not your commit, the fix is a baseline row "
              f"WITH AN OWNER AND AN EXPIRY, not a deletion.", file=sys.stderr)
    # A row goes stale two ways and they need opposite readings: telling a
    # reader a DELETED symbol "now has a production caller" sends them hunting
    # for something that does not exist. A gate's message is part of the gate.
    upstream = _upstream_baseline() if stale else None
    for label in stale:
        if upstream is not None and label not in upstream:
            print(f"BEHIND MASTER — rebase, this is not a finding: {label}\n"
                  f"  The row is already gone from {BASELINE.name} on "
                  f"origin/master: somebody turned this ratchet upstream and "
                  f"your branch predates it. Rebase and the red goes with it. "
                  f"Nothing here needs investigating.", file=sys.stderr)
            continue
        note = notes.get(label, "")
        owed = f"\n  THE ROW SAID: {note}" if note else ""
        if label in DEFINED_LABELS:
            print(f"WIRED UP — turn the ratchet: {label}{owed}\n"
                  f"  This is GOOD news, not a regression: the symbol still "
                  f"exists and now HAS a production caller. That is the "
                  f"outcome this guard exists to produce. Delete its line from "
                  f"{BASELINE.name} in the same commit as the wiring.",
                  file=sys.stderr)
        else:
            print(f"DELETED — turn the ratchet: {label}{owed}\n"
                  f"  The symbol is GONE from the tree, so nothing calls it and "
                  f"nothing can. Delete its line from {BASELINE.name} in the "
                  f"same commit as the deletion. (Do not go looking for its new "
                  f"caller — there isn't one.)", file=sys.stderr)
    return 1 if (new or stale or unexplained) else 0


if __name__ == "__main__":
    raise SystemExit(main())
