"""pgw#853 — A COMPILE FEATURE MUST NEVER BE ABLE TO BREAK SERVING.

The defect, found by pgw#834: the ONLY mechanism the platform has for making
an export declaration visible to a pod is importing a module that calls
``register_export_declaration`` at module scope — and ltx-2.3, qwen-image and
z-image expressed their mint blockers by RAISING ``MintRefused`` at that same
module scope. So wiring those three to the AOT lane the way sdxl is wired
would take the endpoint DOWN AT BOOT. A refusal to MINT was being expressed
as a refusal to IMPORT; the two have different blast radii, and conflating
them is the bug.

The invariant this file exists to establish, and the reason it boots a real
worker rather than asserting on a function: a declaration that refuses must
cost the AOT lane and NOTHING ELSE. It is written so a FUTURE declaration
that raises at import cannot take an endpoint down.

pgw#1107 retired the mechanism, not the invariant. The refusal used to be a
THUNK evaluated when the mint asked; every family is now folded onto
``@endpoint(compile=)``, which takes a ``Compile`` and never a callable, so
the refusal is DATA (``Compile.blockers``, pgw#1115) and the registry accessor
cannot raise at all. What survives from the thunk is its construction check —
an export declaration that names no graph classes has nothing to derive from —
and that check is asserted here, on the path the decorator actually walks.

Deliberately real: ``harness.blocked_declaration`` still raises at module
scope (the OTHER failure, a declaration file that throws for any reason),
``harness.blocked_declaration_endpoints`` imports it the way an endpoint must,
and the serving proof is a real job dispatched over the hub-double's gRPC wire
into a real ``Worker``. Nothing about the raise is simulated.
"""

from __future__ import annotations

import importlib
from typing import List, Tuple

import msgspec
import pytest

from gen_worker.aot_mint import MintRefused
from gen_worker.api.export_contract import (
    DeclarationError, export_declaration, has_export_declaration,
    import_export_declaration, register_export_declaration,
    registered_export_families, reset_export_declarations,
)
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.hub_double import hub_double, is_ready, is_result_for

BLOCKED_MODULE = "harness.blocked_declaration"
ENDPOINT_MODULE = "harness.blocked_declaration_endpoints"


# ---------------------------------------------------------------------------
# 0. The stimulus is REAL — this module genuinely raises at import.
# ---------------------------------------------------------------------------


def test_the_declaration_module_really_raises_at_import() -> None:
    """Without this, every assertion below is vacuous."""
    import sys

    sys.modules.pop(BLOCKED_MODULE, None)
    with pytest.raises(MintRefused) as excinfo:
        importlib.import_module(BLOCKED_MODULE)
    assert "UNRESOLVED mint blocker" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 1. THE INVARIANT — an endpoint carrying a raising declaration still SERVES.
#    RED without `import_export_declaration`: collection dies, the worker
#    never reaches READY, and no job is ever accepted.
# ---------------------------------------------------------------------------


def test_an_endpoint_whose_declaration_raises_still_serves_real_jobs() -> None:
    with hub_double(modules=(ENDPOINT_MODULE,)) as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-blocked", attempt=1, function_name="echo",
            input_payload=msgspec.msgpack.encode({"text": "marco"})))
        res = conn.wait_for(is_result_for("r-blocked")).job_result

    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    assert msgspec.msgpack.decode(res.inline)["response"] == "served:marco"


def test_the_guarded_import_reports_the_failure_instead_of_raising() -> None:
    """Degrade, loudly — the pod must SAY the AOT lane is off, not die and
    not go quiet. The event carries the exception text verbatim."""
    from gen_worker import activity

    seen: List[Tuple[str, str, str]] = []
    real_emit = activity.emit_event
    activity.emit_event = (  # type: ignore[assignment]
        lambda kind, detail, phase="", duration_ms=0, **_kw: seen.append(
            (kind, phase, detail)))
    try:
        import sys

        sys.modules.pop(BLOCKED_MODULE, None)
        ok = import_export_declaration(BLOCKED_MODULE)
    finally:
        activity.emit_event = real_emit  # type: ignore[assignment]

    assert ok is False
    kinds = [k for k, _, _ in seen]
    assert "aot_declaration_import_failed" in kinds, seen
    detail = next(d for k, _, d in seen if k == "aot_declaration_import_failed")
    assert "UNRESOLVED mint blocker" in detail
    assert BLOCKED_MODULE in detail


def test_endpoint_collection_registers_the_blocked_declaration() -> None:
    """The blocked family is DECLARED — it just refuses to mint. Collection
    must register it and read it back without detonating (pgw#853:
    `register_declared_exports` used to read it back through an accessor that
    evaluated a thunk)."""
    from gen_worker.registry import collect_endpoints

    reset_export_declarations()
    try:
        mod = importlib.import_module(ENDPOINT_MODULE)
        register_export_declaration(mod.BLOCKED_DECLARATION, replace=True)

        specs = collect_endpoints([ENDPOINT_MODULE])

        assert specs, "collection produced no endpoint"
        assert mod.BLOCKED_FAMILY in registered_export_families()
        assert has_export_declaration(mod.BLOCKED_FAMILY)
        decl = export_declaration(mod.BLOCKED_FAMILY)
        assert decl is not None
        assert [b.id for b in decl.open_blockers] == ["B1-harness"]
    finally:
        reset_export_declarations()


# ---------------------------------------------------------------------------
# 2. THE REFUSAL STILL HAPPENS — and every word of it survives.
# ---------------------------------------------------------------------------


def test_a_blocked_family_refuses_at_the_MINT_not_at_the_registry_read() -> None:
    """The refusal is readable, typed, and costs the read nothing."""
    from harness.blocked_declaration_parts import BLOCKER, build_declaration

    reset_export_declarations()
    try:
        register_export_declaration(
            build_declaration(family="harness-blocked", blockers=(BLOCKER,)))
        assert has_export_declaration("harness-blocked")

        decl = export_declaration("harness-blocked")

        assert decl is not None, "reading a blocked declaration returned None"
        assert [b.id for b in decl.open_blockers] == ["B1-harness"]
        assert "not pytree-representable" in decl.open_blockers[0].what
    finally:
        reset_export_declarations()


def test_a_CALLABLE_declaration_is_refused_and_says_what_to_use_instead() -> None:
    """pgw#1107 deleted the thunk. A callable must not be accepted-and-ignored
    (the fold onto `@endpoint(compile=)` cannot carry one), and the refusal has
    to name the replacement or the author has nowhere to go."""
    from harness.blocked_declaration_parts import build_declaration

    reset_export_declarations()
    try:
        with pytest.raises(DeclarationError) as excinfo:
            register_export_declaration(build_declaration, family="harness-x")
        assert "blockers" in str(excinfo.value)
        assert not has_export_declaration("harness-x")
    finally:
        reset_export_declarations()


def test_a_declaration_whose_family_does_not_match_is_refused_by_name() -> None:
    from harness.blocked_declaration_parts import build_declaration

    reset_export_declarations()
    try:
        with pytest.raises(DeclarationError) as excinfo:
            register_export_declaration(
                build_declaration(), family="not-that-one")
        assert "not-that-one" in str(excinfo.value)
    finally:
        reset_export_declarations()


def test_the_thunks_classes_check_survives_the_thunks_deletion() -> None:
    """`_Thunk.build()` refused a declaration that named no graph classes.
    That check is the one behaviour the deletion had to carry forward — it is
    now the registration invariant, and it is REACHABLE, which it was not
    while the decorator gate filtered class-less declarations out."""
    from gen_worker import Compile, Dim, Input

    reset_export_declarations()
    try:
        with pytest.raises(DeclarationError) as excinfo:
            register_export_declaration(Compile(
                family="harness-classless", targets=("transformer",),
                text_len=77, shapes=((1024, 1024),),
                dims=(Dim("B", carried_by=(("hidden_states", 0),)),),
                inputs=(Input("hidden_states", shape=("B", 4, 128, 128),
                              dtype="bfloat16"),),
                shape_strategy="static-rows"))
        assert "no graph classes" in str(excinfo.value)
    finally:
        reset_export_declarations()


# ---------------------------------------------------------------------------
# 3. THE MINT GATE and the REGISTRATION-IDEMPOTENCE regression both moved on.
#
#    §3 used to prove a thunk refusal lands as a typed `self_mint_skipped`
#    with its text intact; that is now `test_declared_blockers_pgw1115.py`
#    (`test_the_recipe_gate_declines_a_blocked_family_and_names_the_ids` +
#    `test_the_blocked_pod_keeps_serving`), which asserts the same event under
#    the `declaration_blocked` phase on the declared form. §4 proved a THUNK
#    re-registered from the same source file was idempotent — a `Compile` is a
#    frozen msgspec Struct and value-equal, which is what made two thunks the
#    hard case; the value-equality half is covered by
#    `test_export_contract_pgw739.py`. Neither is repeated here.
# ---------------------------------------------------------------------------


FAMILY = "harness-blocked-family"


# ---------------------------------------------------------------------------
# 5. THE WALKER DOOR — an in-wheel declaration is imported by DISCOVERY, not
#    only by main.py, so `import_export_declaration` cannot be the only guard.
# ---------------------------------------------------------------------------


def test_a_package_whose_declaration_submodule_refuses_still_serves() -> None:
    """The shape ltx-2.3 / qwen-image / z-image actually ship.

    `discovery/walk.py` imports EVERY submodule of an endpoint package and
    raises `EndpointImportError` on any failure. So for an in-wheel
    declaration the walker imports it DIRECTLY — `main.py`'s guarded import is
    not on that path at all, so the declaration itself has to be import-safe,
    which is what `Compile(blockers=...)` buys. Proven the only way that means
    anything: a real worker, a real package, a real job over the wire.
    """
    with hub_double(modules=("harness.blocked_pkg",)) as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-pkg", attempt=1, function_name="pkg-echo",
            input_payload=msgspec.msgpack.encode({"text": "marco"})))
        res = conn.wait_for(is_result_for("r-pkg")).job_result

    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    assert msgspec.msgpack.decode(res.inline)["response"] == "pkg-served:marco"


def test_the_walker_registers_the_blocked_family_and_it_still_refuses() -> None:
    """Registered by the walk, and still refusing — both halves, one pass."""
    from gen_worker.registry import collect_endpoints

    # The module may already be in sys.modules from a sibling test, and a
    # cached module does not re-run its registration. Re-assert it so this
    # test states one thing (collection registers the blocked family) rather
    # than accidentally testing import order.
    mod = importlib.import_module("harness.blocked_pkg.aot_declaration")
    reset_export_declarations()
    register_export_declaration(mod.DECLARATION)
    try:
        specs = collect_endpoints(["harness.blocked_pkg"])

        assert specs, "the package collected no endpoint"
        assert FAMILY in registered_export_families()
        decl = export_declaration(FAMILY)
        assert decl is not None
        assert [b.id for b in decl.open_blockers] == ["B1-harness"]
    finally:
        reset_export_declarations()
