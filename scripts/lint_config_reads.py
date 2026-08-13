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
        "DISARMS compiled_graph publish in the parent's action allowlist. Threat: a probe "
        "pod publishing a compiled_graph minted from hand-edited source into the fleet "
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
    ("src/gen_worker/executor.py", "RUNPOD_POD_ID"):
        "DEFECT, listed to keep the gate green while it burns down. A vendor env "
        "used as a proxy for 'managed runtime'. Blocked on pgw#921/th#1488 "
        "RuntimeIdentity.managed; pgw#929 AMBIGUOUS #5 forbids papering over it "
        "with a vendor Settings field.",
    ("src/gen_worker/executor.py", "RUNPOD_PROVIDER"):
        "DEFECT, same site and same blocker as RUNPOD_POD_ID above.",
    ("src/gen_worker/content_credentials.py", "$env_name"):
        "TRIPWIRE that REFUSES THE BOOT (th#1307). Threat: a C2PA private key "
        "reaching a pod. Carries no behaviour of its own — its only outcome is "
        "a loud refusal.",

}

#: Predicate-shaped function names: a `return <env read>` inside one of these is
#: a behaviour selection even without a syntactic `if`.
_PREDICATE_SUFFIXES = ("enabled", "disabled", "_on", "_off", "armed", "forced")


class BehaviourVisitor(ast.NodeVisitor):
    """Env reads whose value reaches a CONDITIONAL rather than a value slot.

    Deliberately syntactic and conservative. It cannot follow a read through a
    variable into an `if` three functions away, and it does not try — a gate
    that pretends to completeness it does not have is worse than one whose
    reach is stated. What it DOES catch is every shape the four switches
    deleted by pgw#995 were written in, which is the shape this defect keeps
    being written in.
    """

    def __init__(self) -> None:
        self.hits: List[Tuple[int, str]] = []

    def _collect(self, node: ast.AST) -> None:
        sub = EnvVisitor()
        sub.visit(node)
        self.hits.extend(sub.hits)

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
            f"was exactly this: it gated the mint recipe and compiled_graph discovery, "
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
