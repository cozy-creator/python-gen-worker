#!/usr/bin/env python3
"""pgw#931 / ruling §1.18: exactly ONE component in this process reads the environment.

Paul, 2026-08-02:

    *"we should NEVER be loading random envs in the middle of code; we should
    only load it from our config pipeline and then pass it around."*

`gen_worker.config` is that component. Every other `os.environ` / `os.getenv`
access in `src/gen_worker` must appear in `scripts/config_reads_allowlist.txt`
with a classification, or this script fails.

Why a repo-owned AST walk rather than ruff's `flake8-tidy-imports` banned-api
-----------------------------------------------------------------------------
Ruff is already pinned and already gating, so TID251 would have been free. Its
exemption unit is `per-file-ignores` — WHOLE FILES — and this repo's accepted
sites are individual lines inside files that also contain violations
(`procsplit/parent.py` holds legitimate child-IPC handoffs next to config reads).
A file-granular allowlist would exempt exactly the files that need checking.

Line granularity is the smaller half of the reason. The larger half: the
allowlist format forces every accepted site to NAME ITS CLASSIFICATION. That is
what stops it from decaying into `config/settings.py`'s old prose exception list,
which named 5 files while the reads lived in 41 — a written boundary that was
10% accurate, which is not a boundary.

The four classifications are §1.18's, and only the first is a defect:

    VIOLATION   a plain config read that should come from the struct
    BOOTSTRAP   read before config can exist; must be a named, tiny set
    IPC         a parent handing a value to a child across an exec boundary
    LIBRARY     the target library reads env at import and offers no other API
    STANDALONE  a CLI that loads no app config
    TRIPWIRE    a guard whose entire purpose is to fire on a misconfiguration

Baselining follows th#1383's precedent on the Go side: the allowlist is seeded
with today's accepted sites so the gate is green on arrival, then burned down. A
gate that fails on day one gets switched off.

Also enforced here (pgw#929): every owned-namespace env name read anywhere in
`src/` must be known to `config.loader` — either bound to a Settings field or
listed in `_OWNED_NON_SETTINGS`. That is what keeps the loader's
unknown-key refusal honest: the refusal is only safe while its exemption set is
derived from the tree rather than hand-maintained.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO / "src" / "gen_worker"
CONFIG_PKG = SRC_ROOT / "config"
ALLOWLIST = REPO / "scripts" / "config_reads_allowlist.txt"

CLASSIFICATIONS = {
    "VIOLATION", "BOOTSTRAP", "IPC", "LIBRARY", "STANDALONE", "TRIPWIRE",
}

#: Namespaces this program owns; see config/loader.py `_OWNED_PREFIXES`.
OWNED_PREFIXES = ("GEN_WORKER_", "TENSORHUB_", "WORKER_", "COZY_")

#: Key for a site whose variable this walk cannot name statically — a bare
#: `os.environ` binding (`dict(os.environ)`) or a computed key.
UNRESOLVED = "<unresolved>"


class EnvVisitor(ast.NodeVisitor):
    """Every `os.environ` / `os.getenv` access, with the name when it is static."""

    def __init__(self) -> None:
        self.hits: List[Tuple[int, str]] = []
        self.consts: Dict[str, str] = {}
        #: `os.environ` attribute nodes already accounted for as the base of a
        #: `.get(...)` / `[...]`. Without this, `os.environ.get("X")` records
        #: BOTH ("X", from the call) and an unnamed bare binding (from the
        #: attribute), which double-counts the census and invents
        #: `<unresolved>` entries for sites whose variable is right there.
        self._consumed: Set[int] = set()

    def load_consts(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            target = value = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign):
                target, value = node.target, node.value
            if (isinstance(target, ast.Name) and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)):
                self.consts[target.id] = value.value

    def _name(self, node: ast.AST | None) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            # A module constant imported from a sibling (`ENV_SOCKET`) cannot be
            # resolved by a single-file walk, but the IDENTIFIER is itself a
            # stable, specific key — and unlike a line number, nobody else's
            # edit moves it.
            return self.consts.get(node.id) or f"${node.id}"
        if isinstance(node, ast.Attribute):
            return self.consts.get(node.attr) or f"${node.attr}"
        return ""

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            base = func.value
            is_environ = isinstance(base, ast.Attribute) and base.attr == "environ"
            is_bare_environ = isinstance(base, ast.Name) and base.id == "environ"
            is_getenv = (isinstance(base, ast.Name) and base.id == "os"
                         and func.attr == "getenv")
            if is_environ or is_bare_environ or is_getenv:
                self._consumed.add(id(base))
                self.hits.append(
                    (node.lineno, self._name(node.args[0] if node.args else None)))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        value = node.value
        if isinstance(value, ast.Attribute) and value.attr == "environ":
            self._consumed.add(id(value))
            self.hits.append((node.lineno, self._name(node.slice)))
        elif isinstance(value, ast.Name) and value.id == "environ":
            self.hits.append((node.lineno, self._name(node.slice)))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """`os.environ` used as a VALUE — `dict(os.environ)`, `source = os.environ`.

        Caught separately because it is the hole a call/subscript-only walk
        leaves, and it is the widest read there is: `topology.py` binds the
        whole mapping and reads a key out of it later, and
        `procsplit/parent.py` copies the entire environment into the compute
        child's and then pops a DENYLIST — so the child inherits every variable
        nobody thought to name. Same shape th#1502 records at `registry.go`.
        Hits dedupe by line, so an `os.environ.get(...)` is not double-counted.
        """
        if (isinstance(node.value, ast.Name) and node.value.id == "os"
                and node.attr == "environ" and id(node) not in self._consumed):
            self.hits.append((node.lineno, ""))
        self.generic_visit(node)


def scan() -> Tuple[Dict[Tuple[str, str], int], Set[str]]:
    """Every env access outside `config/`, keyed by (path, ENV NAME).

    NOT by line number, and that is the whole point. pgw#931 shipped this gate
    keyed on `path:line` and it went red on `dev` within the hour: two sibling
    PRs (#432, #434) merged alongside it and shifted lines in four files nobody
    in this change had touched. A line number is a fact OTHER PEOPLE change
    independently, so pinning an allowlist to one makes the allowlist a second
    carrier that goes stale silently — §4.22, and precisely the defect class
    this gate exists to police. The gate committed the sin it polices.

    (path, name) is stable under unrelated edits, and it is the meaningful unit
    anyway: the classification belongs to "this file reading this variable",
    not to a cursor position. Two sites in one file reading the same name share
    one classification, which is correct — they are one decision.

    The value per key is a representative line number, for DIAGNOSTICS only.
    Never for matching.
    """
    sites: Dict[Tuple[str, str], int] = {}
    names: Set[str] = set()
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if CONFIG_PKG in path.parents or path == CONFIG_PKG:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover - a broken tree is CI's job
            print(f"{path}: syntax error: {exc}", file=sys.stderr)
            return {}, set()
        visitor = EnvVisitor()
        visitor.load_consts(tree)
        visitor.visit(tree)
        rel = str(path.relative_to(REPO))
        for lineno, name in visitor.hits:
            sites.setdefault((rel, name or UNRESOLVED), lineno)
            if name:
                names.add(name)
    return sites, names


def load_allowlist() -> Tuple[Dict[Tuple[str, str], str], List[str]]:
    """Parse `path::ENV_NAME CLASSIFICATION reason` lines."""
    allowed: Dict[Tuple[str, str], str] = {}
    errors: List[str] = []
    if not ALLOWLIST.is_file():
        return allowed, [f"{ALLOWLIST} is missing"]
    for num, raw in enumerate(ALLOWLIST.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            errors.append(
                f"{ALLOWLIST.name}:{num}: expected "
                f"'<path>::<ENV_NAME> <CLASSIFICATION> <reason>', got {raw!r}")
            continue
        site, classification, reason = parts
        if classification not in CLASSIFICATIONS:
            errors.append(
                f"{ALLOWLIST.name}:{num}: unknown classification "
                f"{classification!r} (one of {sorted(CLASSIFICATIONS)})")
            continue
        if not reason.strip():
            errors.append(f"{ALLOWLIST.name}:{num}: a classification needs a reason")
            continue
        path, sep, name = site.partition("::")
        if not path or not sep or not name:
            errors.append(
                f"{ALLOWLIST.name}:{num}: bad site {site!r} — expected "
                f"'<path>::<ENV_NAME>' (use ::{UNRESOLVED} for a bare "
                f"`os.environ` binding or a name this walk cannot resolve)")
            continue
        allowed[(path, name)] = classification
    return allowed, errors


# ---------------------------------------------------------------------------
# pgw#995 — the BEHAVIOUR axis
# ---------------------------------------------------------------------------
#
# The six classifications above answer WHERE a read happens relative to the
# config pipeline. None of them answers whether the read SELECTS BEHAVIOUR, and
# those are orthogonal questions. `GEN_WORKER_PREFER_AOT` was a behaviour switch
# that silently disarmed on a release rebuild and took the entire AOT path dark
# for three pod attempts; nothing WHERE-shaped could have flagged it. The
# allowlist proves the axis was missing rather than implicit: it classified
# `GEN_WORKER_AOT_EXPORT_PARALLEL`/`_REUSE` as LIBRARY (torch has never heard of
# either name) and three serving-hot-path switches as STANDALONE ("a CLI that
# loads no app config").
#
# Paul's rule: env carries CONFIG, SECRETS and TUNING VALUES. A branch selector
# needs typed config, a loud typed observable, and a named threat. This gate
# makes a NEW one fail the build instead of being noticed three pods later.

#: (path, ENV_NAME) -> the named threat this gate defends against.
#: A gate lives here ONLY with a threat a reader can evaluate. "It is useful"
#: and "it is off by default" are not threats.
BEHAVIOUR_GATES: Dict[Tuple[str, str], str] = {
    ("src/gen_worker/host_move_guard.py", "GEN_WORKER_HOST_MOVE_GUARD"):
        "RULED EXCEPTION. Safety guard, ON by default, disabled only with =0. "
        "Threat: a silent host-RAM offload turns a serving pod into a "
        "swap-thrashing one that still answers health checks. Documented in "
        "CLAUDE.md; explicitly out of scope for every env sweep.",
    ("src/gen_worker/procsplit/actions.py", "GEN_WORKER_PROBE"):
        "SECURITY BOUNDARY (pgw#980). Marks the pod a live-edit probe and "
        "DISARMS cell publish in the parent's action allowlist. Threat: a probe "
        "pod publishing a cell minted from hand-edited source into the fleet "
        "store. Deliberately NOT a Settings field: `authorize` is the boundary, "
        "and a guard that depends on a config load having succeeded has an "
        "unarmed window.",
    ("src/gen_worker/procsplit/actions.py", "GEN_WORKER_PROBE_PUBLISH_ARMED"):
        "SECURITY BOUNDARY (pgw#980). The separate second decision that re-arms "
        "publish on a marked probe. Two names so 'this is a probe' and 'this "
        "probe may write' can never be satisfied by one value.",
    ("src/gen_worker/procsplit/__init__.py", "GEN_WORKER_COMPUTE_CHILD"):
        "BOOTSTRAP. Decides WHICH OF TWO PROGRAMS this process is, before "
        "_run_main and before any config can exist. Threat: none — it is not a "
        "policy, it is the process's own identity, and there is no earlier "
        "carrier than the environment it was exec'd with.",
    ("src/gen_worker/supervisor.py", "GEN_WORKER_SUPERVISOR"):
        "BOOTSTRAP. Pre-fork supervisor predicate, runs before the process "
        "entry. Same identity-not-policy reasoning.",
    ("src/gen_worker/supervisor.py", "GEN_WORKER_SUPERVISED"):
        "BOOTSTRAP. Pre-fork re-entry guard; without it the supervisor forks "
        "itself forever.",
    ("src/gen_worker/models/memory.py", "GEN_WORKER_FORBID_CPU_OFFLOAD"):
        "TRIPWIRE at the real placement boundary. Read as env rather than "
        "Settings because a control-plane box exports it box-wide with no worker "
        "config in sight. Threat: a CPU-offloading run on the shared dev box.",
    ("src/gen_worker/aot_wrapper_split.py", "GEN_WORKER_AOT_RUN_IMPL_SPLIT_OFF"):
        "LIVE ON THE FLEET, and that is why it survives its deleted sibling: 5 "
        "SDXL releases declare it and 1 endpoint carries a non-deleted entry "
        "(standing hub, 2026-08-03). Deleting a live switch changes a running "
        "endpoint. Threat: the pgw#811 run_impl split regressing a family, with "
        "no way to unstick it short of a release.",
    ("src/gen_worker/lifecycle.py", "$ENV_VAR"):
        "READ-ONLY WARNING PREDICATE. Re-reads the hub-delivered topology only "
        "to decide whether the 'GPUs are invisible' warning applies. Selects a "
        "log line, never a code path.",
    ("src/gen_worker/models/store.py", "RUNPOD_POD_ID"):
        "DEFECT, listed to keep the gate green while it burns down. A vendor env "
        "used as a proxy for 'managed runtime'. Blocked on pgw#921/th#1488 "
        "RuntimeIdentity.managed; pgw#929 AMBIGUOUS #5 forbids papering over it "
        "with a vendor Settings field.",
    ("src/gen_worker/models/store.py", "RUNPOD_PROVIDER"):
        "DEFECT, same site and same blocker as RUNPOD_POD_ID above.",
    ("src/gen_worker/content_credentials.py", "$env_name"):
        "TRIPWIRE that REFUSES THE BOOT (th#1307). Threat: a C2PA private key "
        "reaching a pod. Carries no behaviour of its own — its only outcome is "
        "a loud refusal.",

    # ---------------------------------------------------------------------
    # th#1887: found by the discriminated-locals pass, which is new. All four
    # were INVISIBLE to the syntactic pass and sat misfiled as STANDALONE ("a
    # CLI that loads no app config") in the allowlist, while living in serving
    # hot-path modules. They are registered here rather than deleted because
    # deletion needs one fact this box cannot produce: whether any published
    # release DECLARES the variable in `endpoint_env_entries`. That is not
    # hypothetical — GEN_WORKER_AOT_RUN_IMPL_SPLIT_OFF above is declared by
    # five live SDXL releases, so "nothing sets it" is a hub query, not an
    # assumption. Deleting a declared switch is exactly the
    # GEN_WORKER_PREFER_AOT failure this whole gate exists to prevent.
    # ---------------------------------------------------------------------
    ("src/gen_worker/models/native_kernels.py", "GEN_WORKER_NATIVE_KERNELS"):
        "th#1887 DELETION TARGET, pending a declaration check. Tri-state "
        "rollout gate (unset/on/off) for the native-kernel path; unset means "
        "the automatic choice, so deletion should be a no-op for every pod "
        "that does not declare it. Threat while it lives: a dormant rollout "
        "switch nobody re-tests, dark exactly like PREFER_AOT was.",
    ("src/gen_worker/models/svdq.py", "GEN_WORKER_SVDQ_ENGINE"):
        "th#1887 DELETION TARGET, pending a declaration check. Self-described "
        "in-code as an 'operational kill-switch' that PINS the svdq engine for "
        "the process; empty (the default) means choose per artifact and host. "
        "Threat while it lives: a pinned engine outliving the incident it was "
        "pinned for, silently serving a stale path on every later boot.",
    ("src/gen_worker/video_encode.py", "GEN_WORKER_VIDEO_ENCODER"):
        "th#1887 DELETION TARGET, pending a declaration check. Selects the "
        "video encoder: 'auto' probes NVENC, 'x264' skips the probe entirely. "
        "Threat while it lives: a pod pinned to x264 keeps encoding on CPU "
        "after its GPU encoder starts working, and nothing reports the gap.",
    ("src/gen_worker/parallel/group.py", "NCCL_NVLS_ENABLE"):
        "READ-ONLY WARNING PREDICATE, same class as lifecycle.py above. Reads "
        "the pre-imposition value only to decide whether to warn that the "
        "settings authority overrode an ambient NCCL setting. Selects a log "
        "line, never a code path — NOT a deletion target.",
}

#: Predicate-shaped function names: a `return <env read>` inside one of these is
#: a behaviour selection even without a syntactic `if`.
_PREDICATE_SUFFIXES = ("enabled", "disabled", "_on", "_off", "armed", "forced")


#: Calls that only NORMALIZE a scalar without changing where it came from.
#: Peeling these is what lets the taint pass see `str(os.environ.get(X) or
#: "auto").strip().lower()` as a read, while refusing to peel `Path(...)`,
#: `json.loads(...)` or a dict literal — which is the entire reason the pass
#: does not drown in value config.
_SCALAR_CASTS = ("str", "int", "float", "bool")
_SCALAR_METHODS = ("strip", "lower", "upper", "lstrip", "rstrip", "casefold")

#: Comparison operators that DISCRIMINATE a value against known alternatives.
#: Truthiness (`if raw:`) is deliberately excluded: that is the dominant shape
#: of genuine value config (`if not cache_dir: cache_dir = default`).
_DISCRIMINATORS = (ast.Eq, ast.NotEq, ast.In, ast.NotIn)


def _peel_scalar(node: ast.AST) -> ast.AST:
    """Strip scalar-normalizing wrappers until the underlying expression shows.

    Closed set on purpose. Anything not listed — `Path(...)`, `json.loads`,
    arithmetic, a dict/list literal — is NOT peeled, so a read wrapped in it is
    not treated as a gate. That single restriction is what keeps
    TORCHINDUCTOR_CACHE_DIR and PYTHONHASHSEED out of the results.
    """
    while True:
        if isinstance(node, ast.Call):
            fn = node.func
            if (isinstance(fn, ast.Name) and fn.id in _SCALAR_CASTS
                    and len(node.args) == 1):
                node = node.args[0]
                continue
            if (isinstance(fn, ast.Attribute) and fn.attr in _SCALAR_METHODS):
                node = fn.value
                continue
        # `os.environ.get(X) or "auto"` — the default-value idiom.
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and node.values:
            node = node.values[0]
            continue
        return node


def _env_read_name(node: ast.AST, consts: Dict[str, str]) -> str | None:
    """The env var name iff `node` ITSELF is an env read — never a nested one.

    Root-only on purpose. Probing a whole subtree instead would treat
    `Path(os.environ.get(X) or "")` and `{"seed": os.environ.get(X)}` as reads
    bound to a local, which is how a directory path and a hash seed end up
    accused of being behaviour gates. The read must survive peeling as the
    WHOLE right-hand side.
    """
    probe = EnvVisitor()
    probe.consts = dict(consts)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            base = func.value
            if ((isinstance(base, ast.Attribute) and base.attr == "environ")
                    or (isinstance(base, ast.Name) and base.id == "environ")
                    or (isinstance(base, ast.Name) and base.id == "os"
                        and func.attr == "getenv")):
                return probe._name(node.args[0] if node.args else None)
        return None
    if isinstance(node, ast.Subscript):
        value = node.value
        if ((isinstance(value, ast.Attribute) and value.attr == "environ")
                or (isinstance(value, ast.Name) and value.id == "environ")):
            return probe._name(node.slice)
    return None


def _literal_collections(tree: ast.AST) -> Set[str]:
    """Module-level names bound to a collection of literals (svdq's SVDQ_ENGINES)."""
    found: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if (isinstance(target, ast.Name)
                and isinstance(value, (ast.Tuple, ast.List, ast.Set))
                and value.elts
                and all(isinstance(e, ast.Constant) for e in value.elts)):
            found.add(target.id)
    return found


def _is_literal_alternatives(node: ast.AST, collections: Set[str]) -> bool:
    """True when `node` is a set of known alternatives to discriminate against."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, int, bool)) or node.value is None
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return bool(node.elts) and all(isinstance(e, ast.Constant) for e in node.elts)
    # A module-level collection of literals, e.g. svdq.py's SVDQ_ENGINES.
    return isinstance(node, ast.Name) and node.id in collections


class BehaviourVisitor(ast.NodeVisitor):
    """Env reads whose value reaches a CONDITIONAL rather than a value slot.

    Deliberately syntactic and conservative. What it catches is stated, and so
    is what it does not — a gate that pretends to completeness it does not have
    is worse than one whose reach is stated.

    TWO passes, because one was not enough:

    1. `_scan_conditions` — the read sits syntactically inside an `if`/`while`/
       ternary/`assert` test, a comprehension guard, or the `return` of a
       predicate-shaped function. Every switch deleted by pgw#995 was this
       shape.

    2. `_scan_discriminated_locals` (th#1887) — the read is assigned to a local
       and that local is then discriminated against known alternatives
       (`==`/`!=`/`in`/`not in`, or a `match` subject). This shape was
       STRUCTURALLY INVISIBLE to pass 1, and three real gates were hiding in
       it: GEN_WORKER_NATIVE_KERNELS, GEN_WORKER_SVDQ_ENGINE and
       GEN_WORKER_VIDEO_ENCODER, all misfiled as STANDALONE. A registry that
       cannot see a whole gate SHAPE is a guard that cannot fire, which is
       worse than the three gates it missed.

    Still out of reach, stated honestly: a read stored in a module-level
    constant and branched on elsewhere; a read passed as a call argument to a
    callee that branches; `BoolOp` short-circuit used as control flow; a read
    hidden in a collection and reached by lookup.
    """

    def __init__(self) -> None:
        self.hits: List[Tuple[int, str]] = []

    def _scan_discriminated_locals(
        self, tree: ast.AST, consts: Dict[str, str], collections: Set[str]
    ) -> None:
        """Pass 2: read -> local -> discriminated against known alternatives."""
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Locals bound to a (scalar-normalized) env read, ROOT-ONLY: the
            # read must be the whole right-hand side once wrappers are peeled.
            tainted: Dict[str, Tuple[int, str]] = {}
            for node in ast.walk(scope):
                targets: List[ast.expr] = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    targets = [node.target]
                if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                    continue
                if node.value is None:
                    continue
                peeled = _peel_scalar(node.value)
                name = _env_read_name(peeled, consts)
                if name is not None:
                    tainted[targets[0].id] = (peeled.lineno, name)
            if not tainted:
                continue
            for node in ast.walk(scope):
                local = ""
                if isinstance(node, ast.Compare) and isinstance(node.ops[0], _DISCRIMINATORS):
                    left, right = node.left, node.comparators[0]
                    if isinstance(left, ast.Name) and _is_literal_alternatives(right, collections):
                        local = left.id
                    elif isinstance(right, ast.Name) and _is_literal_alternatives(left, collections):
                        local = right.id
                elif isinstance(node, ast.Match) and isinstance(node.subject, ast.Name):
                    local = node.subject.id
                if local in tainted:
                    self.hits.append(tainted.pop(local))

    def _scan_conditions(self, tree: ast.AST, consts: Dict[str, str]) -> None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
                self._collect_with(node.test, consts)
            elif isinstance(node, ast.comprehension):
                for cond in node.ifs:
                    self._collect_with(cond, consts)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                returns_bool = (
                    isinstance(node.returns, ast.Name)
                    and node.returns.id == "bool")
                predicate_name = node.name.lower().endswith(_PREDICATE_SUFFIXES)
                if not (returns_bool or predicate_name):
                    continue
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Return) and inner.value is not None:
                        self._collect_with(inner.value, consts)

    def _collect_with(self, node: ast.AST, consts: Dict[str, str]) -> None:
        sub = EnvVisitor()
        sub.consts = dict(consts)
        sub.visit(node)
        self.hits.extend(sub.hits)


def scan_behaviour() -> Dict[Tuple[str, str], int]:
    """Every env read outside `config/` that feeds a conditional."""
    sites: Dict[Tuple[str, str], int] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if CONFIG_PKG in path.parents or path == CONFIG_PKG:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - the other pass reports it
            continue
        consts = EnvVisitor()
        consts.load_consts(tree)
        visitor = BehaviourVisitor()
        visitor._scan_conditions(tree, consts.consts)
        visitor._scan_discriminated_locals(
            tree, consts.consts, _literal_collections(tree))
        rel = str(path.relative_to(REPO))
        for lineno, name in visitor.hits:
            sites.setdefault((rel, name or UNRESOLVED), lineno)
    return sites


def check_behaviour_gates() -> List[str]:
    """Paul's rule, enforced: env carries values, never a branch selection."""
    found = scan_behaviour()
    errors: List[str] = []
    for key in sorted(set(found) - set(BEHAVIOUR_GATES)):
        path, name = key
        errors.append(
            f"{path}:{found[key]} reads {name} from the environment and feeds it "
            f"to a CONDITIONAL. Env vars are for CONFIG and SECRETS, never logic "
            f"or behaviour switches (Paul, standing rule). GEN_WORKER_PREFER_AOT "
            f"was exactly this: it gated the mint recipe and cell discovery, "
            f"silently disarmed on a release rebuild, and cost three pod attempts "
            f"before anyone noticed the AOT path was dark.\n"
            f"    Fix it: make the branch unconditional (if its default is ON and "
            f"nothing declares it, deleting the gate changes NO pod's behaviour), "
            f"or move the decision to typed config with a loud typed observable.\n"
            f"    If it genuinely must stay, add ('{path}', '{name}') to "
            f"BEHAVIOUR_GATES in {Path(__file__).name} with the THREAT it defends "
            f"against — not 'it is useful' and not 'it is off by default'.")
    for key in sorted(set(BEHAVIOUR_GATES) - set(found)):
        path, name = key
        errors.append(
            f"BEHAVIOUR_GATES lists ('{path}', '{name}') but no such conditional "
            f"env read exists any more. Delete the entry — a stale exemption is "
            f"the second carrier this whole gate exists to prevent (§4.22).")
    for key, threat in BEHAVIOUR_GATES.items():
        if len(threat.strip()) < 40:
            errors.append(
                f"BEHAVIOUR_GATES{key}: the threat must be specific enough for a "
                f"reader to evaluate, got {threat!r}")
    return errors


def check_owned_names_known(names: Set[str]) -> List[str]:
    """Every owned-namespace name read in src/ must be known to the loader."""
    sys.path.insert(0, str(REPO / "src"))
    try:
        from gen_worker.config.loader import (  # noqa: PLC0415 - a linter's own import
            _ENV_ALIASES, _ENV_TO_FIELD, _OWNED_NON_SETTINGS,
        )
    except Exception as exc:  # pragma: no cover
        return [f"could not import gen_worker.config.loader: {exc}"]
    known = set(_ENV_TO_FIELD) | set(_ENV_ALIASES) | set(_OWNED_NON_SETTINGS)
    unknown = sorted(
        n for n in names
        if any(n.startswith(p) for p in OWNED_PREFIXES) and n not in known
    )
    return [
        f"{name} is read in src/ but is unknown to config.loader — bind it to a "
        f"Settings field or list it in `_OWNED_NON_SETTINGS`. The loader's "
        f"unknown-key refusal is only safe while that set covers the tree."
        for name in unknown
    ]


def main() -> int:
    sites, names = scan()
    allowed, errors = load_allowlist()
    errors.extend(check_behaviour_gates())

    for key in sorted(set(sites) - set(allowed)):
        path, name = key
        errors.append(
            f"{path}:{sites[key]} reads {name} from the process environment "
            f"outside gen_worker/config. Ruling §1.18: config is loaded once by "
            f"the pipeline and PASSED. If this site is genuinely legitimate, add "
            f"'{path}::{name} <CLASSIFICATION> <reason>' to {ALLOWLIST.name} — "
            f"and say which classification it is.")

    for key in sorted(set(allowed) - set(sites)):
        path, name = key
        errors.append(
            f"{ALLOWLIST.name}: {path}::{name} is allowlisted but no longer "
            f"reads the environment. Delete the line — a stale allowlist is how "
            f"an exception list stops describing the tree.")

    errors.extend(check_owned_names_known(names))

    if errors:
        print("config-read guard (§1.18) failed:\n", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        print(
            f"\n{len(errors)} problem(s). See scripts/lint_config_reads.py.",
            file=sys.stderr)
        return 1

    print(f"config-read guard: OK — {len(sites)} accepted (file, variable) "
          f"pair(s), all classified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
