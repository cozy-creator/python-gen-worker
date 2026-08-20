#!/usr/bin/env python3
"""Keep compiled-graph arming behind the declared structural seams.

``aot_serve``'s arm functions are the only production code allowed to assemble
live TCG serve state. Each must resolve and load the exact key through TCG
(directly or through the one shared resolver), bind constants, create or reuse
an :class:`EntryDispatch` and register the entry. A second constructor,
registration, or wrapper call anywhere else is red: callers submit an admitted
key to a seam; they do not rebuild part of its state machine.

pgw#1329 made this TWO seams, differing in exactly one axis — the constant
SOURCE. ``arm_compiled_graph`` reads a resident eager module and therefore owns
the wrapper swap and the pipeline marker; ``arm_compiled_graph_from_store``
reads the store by manifest FQN and has no ``nn.Module`` at all, so the fence
requires it to install NEITHER. "Two seams" is not a relaxation: each is
audited for its own complete operation set, and the store arm additionally
carries a FORBIDDEN set the module arm does not.

The fence also retains two smaller invariants that are still process-global:
compiled proof/quarantine writes must be classified, and objects with one
canonical constructor may not acquire a second map. Deleted AOT key registries
and their test-only hand-feed exceptions are deliberately absent.

Every invocation first executes five synthetic red proofs: each seam missing a
required operation, the store seam claiming a module, a second dispatch writer,
and a direct marker writer. A green audit therefore proves the rejection paths
ran before the real tree was accepted.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lint_side  # noqa: E402

ARM_MODULE = "aot_serve.py"
ARM_SEAM = "arm_compiled_graph"

#: The scope both seams delegate their exact-key resolve to (pgw#1329). Its
#: calls count toward EVERY seam that calls it, because the fence's subject is
#: "did this arm resolve through TCG", not "which line did it happen on". A
#: shared resolve is the OPPOSITE of a weakening here: two seams that each
#: re-derived the artifact are the pgw#816/#822 class one level up.
ARM_RESOLVER = "_resolve_graph_specialization"

#: pgw#1329: there are TWO arm seams and they differ in exactly one axis, the
#: constant SOURCE. The fence must state both, and must state what each one
#: may NOT do — a store-sourced arm that installed a module wrapper or a
#: pipeline marker would be claiming a module it does not have.
ARM_SEAMS: tuple[tuple[str, frozenset[str], frozenset[str]], ...] = (
    (
        # module-sourced: the eager module is the constant source, so this
        # seam owns the wrapper swap and the pipeline marker.
        ARM_SEAM,
        frozenset({
            "open_worker_engine",
            "resolve",
            "runner",
            "bind",
            "_marker",
            "EntryDispatch",
            "dispatch.add",
            "wrap_module",
        }),
        frozenset(),
    ),
    (
        # store-sourced: no nn.Module anywhere on the path, so no wrapper and
        # no pipeline marker — asserted, not assumed.
        "arm_compiled_graph_from_store",
        frozenset({
            "open_worker_engine",
            "resolve",
            "runner",
            "bind",
            "EntryDispatch",
            "dispatch.add",
        }),
        frozenset({"_marker", "wrap_module"}),
    ),
)

#: Scopes allowed to touch the exclusive operations at all.
ARM_SEAM_SCOPES = frozenset({name for name, _required, _forbidden in ARM_SEAMS})

# These operations assemble the arm state and therefore have exactly one
# production caller EACH per seam. EntryDispatch.remove is a serve-time
# verdict, not admission, and remains owned by disarm_entry.
EXCLUSIVE_ARM_CALLS = frozenset({
    "_marker",
    "EntryDispatch",
    "dispatch.add",
    "wrap_module",
})

# Module marker installation and removal live in these low-level
# implementations. Their callers are still fenced above.
MARKER_WRITER_SCOPES = frozenset({
    (ARM_MODULE, "_marker"),
    (ARM_MODULE, "wrap_module"),
    (ARM_MODULE, "unwrap"),
})

CLASSIFICATIONS = frozenset({"VERDICT", "OWNER", "PROJECTION"})


@dataclass(frozen=True)
class Feeder:
    """One remaining production writer of process-global serve state."""

    dotted: str
    owner: str
    distinctive: bool = True
    receivers: tuple[str, ...] = ()


FEEDERS = (
    Feeder("compile_cache.record_compiled_graph_proven", "compile_cache.py"),
    Feeder("compile_cache.record_compiled_graph_quarantined", "compile_cache.py"),
    Feeder(
        "local_compiled_graph_store.store",
        "local_compiled_graph_store.py",
        distinctive=False,
        receivers=("local_compiled_graph_store", "store"),
    ),
    Feeder("local_compiled_graph_store.note_memo", "local_compiled_graph_store.py"),
)


@dataclass(frozen=True)
class OneConstructor:
    """A production object that must have exactly one canonical map."""

    name: str
    constructor: tuple[str, str]
    when_kwargs: tuple[str, ...] = ()
    why: str = ""


ONE_CONSTRUCTOR = (
    OneConstructor(
        "CompileContract",
        ("registry.py", "CompileContract.from_declaration"),
        why="a declaration field reaching one map but not another silently "
        "changes the compiled class",
    ),
    OneConstructor(
        "_ArmOrder",
        ("executor.py", "_ArmOrder.for_artifact"),
        when_kwargs=("selection",),
        why="plan and adopt must not independently map the same arm order",
    ),
    OneConstructor(
        "_CompileArtifactSelection",
        ("executor.py", "_ArmOrder.for_artifact"),
        why="artifact selection is part of the one arm-order map",
    ),
    OneConstructor(
        "ExpectedIdentity",
        ("aot_identity.py", "ExpectedIdentity.named_by"),
        why="artifact naming has one projection from the declaration",
    ),
)


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _scope(parts: list[str]) -> str:
    return ".".join(parts) or "<module>"


@dataclass(frozen=True)
class CallSite:
    file: str
    scope: str
    line: int
    call: str


class Calls(ast.NodeVisitor):
    """Collect only the few call/write shapes this fence owns."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.parts: list[str] = []
        self.calls: list[CallSite] = []
        self.marker_writes: list[CallSite] = []
        self.feeds: list[CallSite] = []
        self.builds: list[CallSite] = []
        self.defined_scopes: set[str] = set()
        self.dispatch_vars: set[tuple[str, str]] = set()

    @property
    def scope(self) -> str:
        return _scope(self.parts)

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        self.parts.append(name)
        self.defined_scopes.add(self.scope)
        self.generic_visit(node)
        self.parts.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node, node.name)

    def _record_assignment(self, target: ast.AST, value: ast.AST) -> None:
        if isinstance(target, ast.Attribute) and target.attr == "_cozy_aot":
            self.marker_writes.append(
                CallSite(self.filename, self.scope, target.lineno, "assignment")
            )
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            return
        constructor = _dotted(value.func).rpartition(".")[2]
        if constructor in {"EntryDispatch", "_dispatch_for"}:
            self.dispatch_vars.add((self.scope, target.id))

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_assignment(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._record_assignment(node.target, node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        path = _dotted(node.func)
        tail = path.rpartition(".")[2]
        receiver = path.rpartition(".")[0]
        dispatch_add = tail == "add" and (
            (self.scope, receiver) in self.dispatch_vars
            or receiver.rpartition(".")[2].endswith("dispatch")
            or (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Call)
                and _dotted(node.func.value.func).rpartition(".")[2]
                == "_dispatch_for"
            )
        )
        call = "dispatch.add" if dispatch_add else tail
        self.calls.append(CallSite(self.filename, self.scope, node.lineno, call))

        if tail in {"setattr", "delattr"} and len(node.args) >= 2:
            marker = _dotted(node.args[1])
            if (
                (self.filename == ARM_MODULE and marker == "_MARKER_ATTR")
                or marker.endswith("aot_serve._MARKER_ATTR")
                or (
                    isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "_cozy_aot"
                )
            ):
                self.marker_writes.append(
                    CallSite(self.filename, self.scope, node.lineno, tail)
                )

        for feeder in FEEDERS:
            bare = feeder.dotted.rpartition(".")[2]
            if tail != bare:
                continue
            feed_receiver = path.rpartition(".")[0].rpartition(".")[2]
            if not feed_receiver:
                if self.filename != feeder.owner:
                    continue
            elif (
                not feeder.distinctive
                and feed_receiver not in feeder.receivers
            ):
                continue
            self.feeds.append(
                CallSite(self.filename, self.scope, node.lineno, feeder.dotted)
            )

        for one in ONE_CONSTRUCTOR:
            if tail != one.name:
                continue
            if one.when_kwargs and not any(
                keyword.arg in one.when_kwargs for keyword in node.keywords
            ):
                continue
            self.builds.append(
                CallSite(self.filename, self.scope, node.lineno, f"{one.name}()")
            )
        self.generic_visit(node)


def _parse_modules(sources: Mapping[str, str]) -> dict[str, Calls]:
    out: dict[str, Calls] = {}
    for filename, source in sources.items():
        visitor = Calls(filename)
        visitor.visit(ast.parse(source, filename=filename))
        out[filename] = visitor
    return out


def audit_arm_seam(sources: Mapping[str, str]) -> list[str]:
    """Return structural TCG arm violations for parsed module sources."""
    modules = _parse_modules(sources)
    owner = modules.get(ARM_MODULE)
    problems: list[str] = []
    if owner is None:
        return [f"ARM SEAM MISSING: {ARM_MODULE}::{ARM_SEAM}"]
    missing_seams = [
        name for name, _required, _forbidden in ARM_SEAMS
        if name not in owner.defined_scopes
    ]
    if missing_seams:
        return [
            f"ARM SEAM MISSING: {ARM_MODULE}::{name}" for name in missing_seams
        ]

    # A seam that delegates the exact-key resolve to the shared helper is
    # credited with the helper's calls: the fence asks whether the arm
    # resolved through TCG, not where the statement sits.
    shared = {site.call for site in owner.calls if site.scope == ARM_RESOLVER}
    for name, required, forbidden in ARM_SEAMS:
        seam_calls = {site.call for site in owner.calls if site.scope == name}
        if ARM_RESOLVER in seam_calls:
            seam_calls |= shared
        missing = sorted(required - seam_calls)
        if missing:
            problems.append(
                f"ARM SEAM INCOMPLETE: {ARM_MODULE}::{name} no longer calls "
                f"{', '.join(missing)}; exact TCG resolve, bind and dispatch "
                "registration are one atomic boundary"
            )
        overreach = sorted(forbidden & seam_calls)
        if overreach:
            problems.append(
                f"ARM SEAM OVERREACH: {ARM_MODULE}::{name} calls "
                f"{', '.join(overreach)}; a store-sourced arm has no "
                "nn.Module, so it must install no wrapper and no pipeline "
                "marker"
            )

    for visitor in modules.values():
        for site in visitor.calls:
            if site.call not in EXCLUSIVE_ARM_CALLS:
                continue
            if site.file == ARM_MODULE and site.scope in ARM_SEAM_SCOPES:
                continue
            problems.append(
                f"{site.file}:{site.line}: ARM STATE WRITE OUTSIDE SEAM: "
                f"{site.scope} calls {site.call}; route the admitted key "
                f"through {ARM_MODULE}::{ARM_SEAM}"
            )
        for site in visitor.marker_writes:
            if (site.file, site.scope) in MARKER_WRITER_SCOPES:
                continue
            problems.append(
                f"{site.file}:{site.line}: DIRECT AOT MARKER WRITE: "
                f"{site.scope} calls {site.call}; module and pipeline marker "
                f"state is installed only by {ARM_MODULE}::{ARM_SEAM}"
            )
    return problems


def _iter_sources(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _site_key(root: Path, filename: str, scope: str, call: str) -> tuple[str, str]:
    path = root / filename
    return str(path.relative_to(REPO)), f"{scope}::{call}"


def audit_classified_sites(
    root: Path,
    sources: Mapping[str, str],
) -> tuple[dict[tuple[str, str], int], list[str]]:
    """Collect remaining global feeds and duplicate constructor maps."""
    modules = _parse_modules(sources)
    sites: dict[tuple[str, str], int] = {}
    problems: list[str] = []
    for filename, visitor in modules.items():
        for site in visitor.feeds + visitor.builds:
            one = next(
                (
                    row for row in ONE_CONSTRUCTOR
                    if site.call == f"{row.name}()"
                ),
                None,
            )
            if one is not None and (filename, site.scope) == one.constructor:
                continue
            key = _site_key(root, filename, site.scope, site.call)
            sites.setdefault(key, site.line)

    for one in ONE_CONSTRUCTOR:
        constructor_visitor = modules.get(one.constructor[0])
        if (
            constructor_visitor is not None
            and one.constructor[1] not in constructor_visitor.defined_scopes
        ):
            problems.append(
                f"ONE CONSTRUCTOR MISSING: {one.constructor[0]}::"
                f"{one.constructor[1]} no longer defines the sole {one.name} map"
            )
    return sites, problems


def load_allowlist(path: Path) -> tuple[dict[tuple[str, str], str], list[str]]:
    """Parse ``<path>::<scope>::<call> <classification> <reason>`` rows."""
    allowed: dict[tuple[str, str], str] = {}
    errors: list[str] = []
    if not path.is_file():
        return allowed, [f"{path} is missing"]
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            errors.append(
                f"{path.name}:{number}: need '<site> <classification> <reason>'"
            )
            continue
        key_text, classification, _reason = parts
        if classification not in CLASSIFICATIONS:
            errors.append(
                f"{path.name}:{number}: unknown classification "
                f"{classification!r}; want {sorted(CLASSIFICATIONS)}"
            )
            continue
        file_part, separator, name = key_text.partition("::")
        if not separator or not name:
            errors.append(
                f"{path.name}:{number}: site key {key_text!r} lacks '::'"
            )
            continue
        allowed[(file_part, name)] = classification
    return allowed, errors


def check_allowlist(
    sites: Mapping[tuple[str, str], int],
    allowed: Mapping[tuple[str, str], str],
) -> list[str]:
    problems: list[str] = []
    for key, line in sorted(sites.items()):
        if key in allowed:
            continue
        call = key[1].rpartition("::")[2]
        one = next(
            (row for row in ONE_CONSTRUCTOR if call == f"{row.name}()"),
            None,
        )
        if one is not None:
            problems.append(
                f"{key[0]}:{line}: SECOND MAP into {one.name} at {key[1]}; "
                f"{one.constructor[0]}::{one.constructor[1]} is canonical. "
                f"{one.why}. A genuinely different source projection may be "
                "classified PROJECTION."
            )
        else:
            problems.append(
                f"{key[0]}:{line}: UNCLASSIFIED process-global feed "
                f"{key[1]}; classify the verdict/owner at its real write site"
            )
    for key in sorted(set(allowed) - set(sites)):
        problems.append(
            f"stale allowlist row {key[0]}::{key[1]} matches no site; delete it"
        )
    return problems


def run_red_proofs() -> int:
    """Exercise the current fence's rejection paths before auditing src."""
    good = """
def _resolve_graph_specialization():
    engine = open_worker_engine()
    engine.resolve()
    engine.runner()
def arm_compiled_graph():
    resolved = _resolve_graph_specialization()
    resolved.runner.bind()
    _marker()
    dispatch = EntryDispatch()
    dispatch.add()
    wrap_module()
def arm_compiled_graph_from_store():
    resolved = _resolve_graph_specialization()
    resolved.runner.bind()
    dispatch = EntryDispatch()
    dispatch.add()
def _marker():
    setattr(object(), _MARKER_ATTR, {})
def wrap_module():
    setattr(object(), _MARKER_ATTR, {})
def unwrap():
    delattr(object(), _MARKER_ATTR)
"""
    cases = (
        (
            {ARM_MODULE: good.replace("    wrap_module()\n", "")},
            "ARM SEAM INCOMPLETE",
        ),
        (
            # the store arm stops resolving through TCG and hand-loads: the
            # exact-key admission is what makes an arm an arm.
            {ARM_MODULE: good.replace(
                "def arm_compiled_graph_from_store():\n"
                "    resolved = _resolve_graph_specialization()\n",
                "def arm_compiled_graph_from_store():\n"
                "    resolved = load_package()\n",
            )},
            "ARM SEAM INCOMPLETE",
        ),
        (
            # the store arm claims a module it does not have.
            {ARM_MODULE: good.replace(
                "def arm_compiled_graph_from_store():\n"
                "    resolved = _resolve_graph_specialization()\n",
                "def arm_compiled_graph_from_store():\n"
                "    resolved = _resolve_graph_specialization()\n"
                "    wrap_module()\n",
            )},
            "ARM SEAM OVERREACH",
        ),
        (
            {ARM_MODULE: good + "\ndef bypass():\n    EntryDispatch()\n"},
            "ARM STATE WRITE OUTSIDE SEAM",
        ),
        (
            {
                ARM_MODULE: good,
                "bypass.py": "def bypass(obj):\n    obj._cozy_aot = {}\n",
            },
            "DIRECT AOT MARKER WRITE",
        ),
    )
    for sources, expected in cases:
        observed = audit_arm_seam(sources)
        if not any(expected in problem for problem in observed):
            raise AssertionError(
                f"red proof did not reject {expected}: {observed!r}"
            )
    if audit_arm_seam({ARM_MODULE: good}):
        raise AssertionError("valid synthetic arm seam did not pass")
    return len(cases)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=REPO / "src" / "gen_worker")
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=REPO / "scripts" / "arm_state_feeders_allowlist.txt",
    )
    args = parser.parse_args(argv)

    red_proofs = run_red_proofs()
    sources = _iter_sources(args.src)
    arm_problems = audit_arm_seam(sources)
    sites, site_problems = audit_classified_sites(args.src, sources)
    allowed, allowlist_errors = load_allowlist(args.allowlist)
    problems = (
        arm_problems
        + site_problems
        + allowlist_errors
        + check_allowlist(sites, allowed)
    )
    if problems:
        _lint_side.report(problems, "pgw#1152 arm-state feeders")
        return 1
    print(
        f"arm-state fence: {red_proofs} red proofs passed; "
        f"{ARM_MODULE} declares {len(ARM_SEAMS)} TCG arm seam(s), all complete; "
        f"{len(sites)} classified global/constructor site(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
