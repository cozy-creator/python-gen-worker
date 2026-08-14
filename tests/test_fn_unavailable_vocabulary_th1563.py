"""The FnUnavailable reason vocabulary is a CONTRACT, and it drifts.

The wire contract carries a list of the reason tokens this worker may put in
``FnUnavailable.reason``. The hub branches on those tokens — each has a
lifetime, and the ones it treats as hardware facts are never re-probed at all —
so a reason the list forgets is a reason nobody checked the hub's handling of.
The list has been provably wrong more than once, including for reasons the hub
deliberately never decays.

The list is therefore not maintained by memory. This test DERIVES the vocabulary
from the source that emits it — every literal that can reach the reason slot of
``ExecutorState.unavailable``, which ``lifecycle._emit_unavailable`` puts on the
wire verbatim — and requires the vendored ``.proto`` to document each token.
Adding a new refusal reason without a contract entry fails here, in this repo,
offline.

The derivation reads the three shapes the emitting code actually uses: the
literal written inline, the local ``code``/``reason`` variable a branch picked,
and ``getattr(exc, "reason", ...)`` resolving through the gate exception
classes. It is deliberately scoped to the ENCLOSING function — a module-wide
name scan would sweep in every unrelated ``reason`` variable in the file and
turn the contract into noise.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PROTO = _ROOT / "proto" / "worker_scheduler.proto"
_SOURCES = (
    _ROOT / "src" / "gen_worker" / "executor.py",
    _ROOT / "src" / "gen_worker" / "capability.py",
    # The third file declaring a reason token on an exception class.
    # `_mark_setup_failed` writes `exc.reason` for it exactly the way it does
    # for the capability gates, so leaving this file out would have made the
    # NEW token the one case the contract test cannot see — which is precisely
    # the blindness th#1562/th#1563 wrote this test to end.
    _ROOT / "src" / "gen_worker" / "models" / "envelope.py",
)

_FunctionNode = (ast.FunctionDef, ast.AsyncFunctionDef)


def _string_literals(node: ast.AST) -> set[str]:
    """String literals directly reachable from one reason expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    # `"cuda_unavailable" if plan.run_mode == RUN_CPU else "insufficient_vram"`:
    # BOTH arms are emissions.
    if isinstance(node, ast.IfExp):
        return _string_literals(node.body) | _string_literals(node.orelse)
    if isinstance(node, ast.BoolOp):
        out: set[str] = set()
        for value in node.values:
            out |= _string_literals(value)
        return out
    if isinstance(node, ast.Call):
        func = node.func
        # getattr(exc, "reason", "<default>") — the default is a real emission.
        if isinstance(func, ast.Name) and func.id == "getattr" and len(node.args) == 3:
            return _string_literals(node.args[2])
    return set()


def _name_bindings(scope: ast.AST) -> dict[str, set[str]]:
    """Every ``NAME = "literal"`` binding inside one scope, including the
    tuple-unpacked form (``reason, axes = "retry_exhausted", {}``)."""
    out: dict[str, set[str]] = {}

    def bind(target: ast.expr, value: ast.AST) -> None:
        if isinstance(target, ast.Name):
            literals = _string_literals(value)
            if literals:
                out.setdefault(target.id, set()).update(literals)

    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for sub, val in zip(target.elts, node.value.elts):
                        bind(sub, val)
                else:
                    bind(target, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind(node.target, node.value)
    return out


def _reason_slot_expressions(scope: ast.AST) -> list[ast.expr]:
    """Every expression written into ``...unavailable[...]``'s reason slot."""
    slots: list[ast.expr] = []
    for node in ast.walk(scope):
        # self.unavailable[name] = (reason, detail, axes)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "unavailable"
                    and node.value.elts
                ):
                    slots.append(node.value.elts[0])
        # self.unavailable.setdefault(name, (reason, detail, axes))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            if (
                func.attr == "setdefault"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "unavailable"
                and len(node.args) == 2
                and isinstance(node.args[1], ast.Tuple)
                and node.args[1].elts
            ):
                slots.append(node.args[1].elts[0])
    return slots


def _module_bindings(tree: ast.Module) -> dict[str, set[str]]:
    """Module-level ``NAME = "literal"`` constants, for reasons referred to by
    constant (``_CANCEL_UNWIND_REASON``) rather than written inline."""
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    out.setdefault(target.id, set()).add(node.value.value)
    return out


def _gate_exception_reasons(tree: ast.Module) -> set[str]:
    """``reason = "..."`` declared on a gate exception class or its base: these
    reach the wire through ``getattr(exc, "reason", ...)``, whichever subclass
    the gate raised."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for name, literals in _name_bindings_direct(node).items():
            if name == "reason":
                out |= literals
    return out


def _name_bindings_direct(cls: ast.ClassDef) -> dict[str, set[str]]:
    """Class-body assignments only — not the bodies of its methods."""
    out: dict[str, set[str]] = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and isinstance(stmt.value.value, str):
                    out.setdefault(target.id, set()).add(stmt.value.value)
        elif (
            isinstance(stmt, ast.AnnAssign)
            and stmt.value is not None
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.target, ast.Name)
            and isinstance(stmt.value.value, str)
        ):
            out.setdefault(stmt.target.id, set()).add(stmt.value.value)
    return out


def emitted_reason_vocabulary() -> set[str]:
    reasons: set[str] = set()
    for path in _SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_names = _module_bindings(tree)
        reasons |= _gate_exception_reasons(tree)

        scopes: list[ast.AST] = [tree]
        scopes += [n for n in ast.walk(tree) if isinstance(n, _FunctionNode)]
        for scope in scopes:
            slots = _reason_slot_expressions(scope)
            if not slots:
                continue
            local_names = _name_bindings(scope)
            for slot in slots:
                reasons |= _string_literals(slot)
                if isinstance(slot, ast.Name):
                    reasons |= local_names.get(slot.id, set())
                    reasons |= module_names.get(slot.id, set())
    return reasons


def _fn_unavailable_message() -> str:
    body = _PROTO.read_text(encoding="utf-8")
    start = body.index("message FnUnavailable {")
    return body[start : body.index("\n}", start)]


def test_the_derivation_still_finds_the_refusals() -> None:
    """The scan itself is load-bearing: a refactor that moves the refusal map
    must not silently turn this whole file into a no-op."""
    reasons = emitted_reason_vocabulary()
    for anchor in ("native_crash_streak", "setup_failed", "compute_capability_unmet"):
        assert anchor in reasons, (
            f"the reason scan no longer finds {anchor!r}, which `executor.py` "
            f"demonstrably emits — the derivation has gone blind. Found: {sorted(reasons)}"
        )
    assert len(reasons) >= 15, f"suspiciously few reasons derived: {sorted(reasons)}"


@pytest.mark.parametrize("reason", sorted(emitted_reason_vocabulary()))
def test_every_emitted_reason_is_in_the_wire_contract(reason: str) -> None:
    message = _fn_unavailable_message()
    assert reason in message, (
        f"this worker can emit FnUnavailable.reason={reason!r}, and the wire "
        "contract's reason vocabulary does not mention it. The hub gives each "
        "token its own lifetime (th#1100), so an undocumented reason is one "
        "nobody checked the hub's handling of — the exact defect th#1562 and "
        "th#1563 each had to repair. Add it to "
        "tensorhub:internal/orchestrator/grpc/proto/worker_scheduler.proto and "
        "re-vendor with scripts/vendor-proto.sh."
    )
