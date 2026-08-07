"""pgw#853 — A COMPILE FEATURE MUST NEVER BE ABLE TO BREAK SERVING.

The defect, found by pgw#834: the ONLY mechanism the platform has for making
an export declaration visible to a pod is importing a module that calls
``register_export_declaration`` at module scope — and ltx-2.3, qwen-image and
z-image express their mint blockers by RAISING ``MintRefused`` at that same
module scope. So wiring those three to the AOT lane the way sdxl is wired
would take the endpoint DOWN AT BOOT. A refusal to MINT was being expressed
as a refusal to IMPORT; the two have different blast radii, and conflating
them is the bug.

The invariant this file exists to establish, and the reason it boots a real
worker rather than asserting on a function: a declaration that raises must
cost the AOT lane and NOTHING ELSE. It is written so a FUTURE declaration
that raises at import cannot take an endpoint down.

Deliberately real: ``harness.blocked_declaration`` raises exactly the way
z-image's file does, ``harness.blocked_declaration_endpoints`` imports it the
way an endpoint must, and the serving proof is a real job dispatched over the
hub-double's gRPC wire into a real ``Worker``. Nothing about the raise is
simulated.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import msgspec
import pytest

from gen_worker import aot_cells, fleet_cells
from gen_worker.aot_mint import MintRefused
from gen_worker.api.export_contract import (
    DeclarationError, export_declaration, has_export_declaration,
    import_export_declaration, register_export_declaration,
    registered_export_families, reset_export_declarations,
)
from gen_worker import config as gw_config
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.cell_adopt import AdoptOutcome

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
        lambda kind, detail, phase="", duration_ms=0: seen.append(
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


def test_endpoint_collection_registers_the_thunk_without_evaluating_it() -> None:
    """The blocked family is DECLARED — it just refuses to mint. Reading the
    registry must not detonate it (pgw#853: `register_declared_exports` used
    to read it back with `export_declaration`)."""
    from gen_worker.registry import collect_endpoints

    reset_export_declarations()
    try:
        importlib.import_module(ENDPOINT_MODULE)
        mod = importlib.import_module(ENDPOINT_MODULE)
        register_export_declaration(
            mod._blocked_thunk, family=mod.THUNK_FAMILY, replace=True)

        specs = collect_endpoints([ENDPOINT_MODULE])

        assert specs, "collection produced no endpoint"
        assert mod.THUNK_FAMILY in registered_export_families()
        assert has_export_declaration(mod.THUNK_FAMILY)
    finally:
        reset_export_declarations()


# ---------------------------------------------------------------------------
# 2. THE REFUSAL STILL HAPPENS — and every word of it survives.
# ---------------------------------------------------------------------------


def test_a_thunk_refuses_where_the_mint_asks_not_where_python_imports() -> None:
    reset_export_declarations()
    try:
        register_export_declaration(_raising_thunk, family="harness-blocked")
        assert has_export_declaration("harness-blocked")
        with pytest.raises(MintRefused) as excinfo:
            export_declaration("harness-blocked")
        assert "B1-harness" in str(excinfo.value)
    finally:
        reset_export_declarations()


def test_a_callable_registration_requires_an_explicit_family() -> None:
    reset_export_declarations()
    try:
        with pytest.raises(DeclarationError):
            register_export_declaration(_raising_thunk)
    finally:
        reset_export_declarations()


def test_a_thunk_that_builds_the_wrong_family_is_refused_by_name() -> None:
    from harness.blocked_declaration_parts import build_declaration

    reset_export_declarations()
    try:
        register_export_declaration(build_declaration, family="not-that-one")
        with pytest.raises(DeclarationError) as excinfo:
            export_declaration("not-that-one")
        assert "not-that-one" in str(excinfo.value)
    finally:
        reset_export_declarations()


def test_a_thunk_that_builds_is_evaluated_once_and_memoized() -> None:
    calls = {"n": 0}

    def _once():
        from harness.blocked_declaration_parts import build_declaration

        calls["n"] += 1
        return build_declaration()

    reset_export_declarations()
    try:
        register_export_declaration(_once, family="harness-blocked-family")
        first = export_declaration("harness-blocked-family")
        second = export_declaration("harness-blocked-family")
        assert first is second
        assert calls["n"] == 1
    finally:
        reset_export_declarations()


# ---------------------------------------------------------------------------
# 3. THE MINT GATE — a thunk refusal lands as a TYPED event, text intact.
# ---------------------------------------------------------------------------


FAMILY = "harness-blocked-family"


class _Pipe:
    pass


@dataclass
class _Cfg:
    family: str = FAMILY
    lora_bucket: int = 64
    shapes: Tuple[Tuple[int, int], ...] = ((1024, 1024),)
    targets: Tuple[str, ...] = ("transformer",)
    text_lens: Tuple[int, ...] = (77,)
    guidance_scales: Tuple[float, ...] = (1.0, 5.0)
    regional: bool = False


class _Publisher:
    base_url = "http://hub.invalid"

    def enabled(self) -> bool:
        return True

    def worker_jwt(self) -> str:
        return "jwt"


def _raising_thunk():
    from harness.blocked_declaration_parts import BLOCKER_TEXT

    raise MintRefused(BLOCKER_TEXT)


@pytest.fixture()
def _events(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str, str]]:
    seen: List[Tuple[str, str, str]] = []
    monkeypatch.setattr(
        fleet_cells.activity_mod, "emit_event",
        lambda kind, detail, phase="", duration_ms=0: seen.append(
            (kind, phase, detail)))
    return seen


@pytest.fixture()
def _miss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """A real AOT discovery miss on an otherwise mint-capable pod."""
    gw_config.reload_for_test()
    monkeypatch.setattr(aot_cells, "discover", lambda *a, **k: None)
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: AdoptOutcome.miss("no_cell"))
    monkeypatch.setattr(fleet_cells.cc, "has_compile_target", lambda p, c: True)
    monkeypatch.setattr(fleet_cells.cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(fleet_cells.cc, "delivered_cell_seeded", lambda: False)
    monkeypatch.setattr(fleet_cells.cc, "apply_lora_execution_lane", lambda p, b: None)
    monkeypatch.setattr(fleet_cells.cc, "drop_lora_execution_lane", lambda p: None)
    monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
    monkeypatch.setattr(fleet_cells, "_PENDING", {})
    monkeypatch.setattr(
        fleet_cells.cell_key, "compute",
        lambda *a, **k: type("_K", (), {"digest": "ck1-" + "a" * 56})())
    monkeypatch.setattr(fleet_cells.cc, "mandatory_serving", lambda p: False)
    monkeypatch.setattr(
        fleet_cells.cc, "begin_fleet_mint", lambda p, c, capture: None)
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "w8a8")
    reset_export_declarations()
    yield
    reset_export_declarations()
    gw_config.reload_for_test()


def test_the_mint_gate_declines_typed_and_keeps_the_blocker_text(
    _miss: Any, _events: List[Tuple[str, str, str]],
) -> None:
    """The whole point of moving the refusal: it must still be SAID, with
    its evidence, on the wire — not swallowed by a try/except."""
    register_export_declaration(_raising_thunk, family=FAMILY)

    outcome = fleet_cells.enable_compiled(
        _Pipe(), _Cfg(), publisher=_Publisher(), delegate=True)  # type: ignore[arg-type]

    pending = outcome.self_mint
    assert pending is not None
    assert pending.recipe == fleet_cells.RECIPE_DYNAMO, (
        "an AOT cell cannot be minted for a blocked family — it must fall "
        "back, not proceed")
    skipped = [(p, d) for k, p, d in _events if k == "self_mint_skipped"]
    assert skipped, _events
    phases = [p for p, _ in skipped]
    assert "declaration_refused" in phases, phases
    detail = next(d for p, d in skipped if p == "declaration_refused")
    assert "B1-harness" in detail, detail
    assert "RESOLVES WHEN" in detail, detail


def test_the_pod_serves_eager_rather_than_failing_the_arm(
    _miss: Any, _events: List[Tuple[str, str, str]],
) -> None:
    register_export_declaration(_raising_thunk, family=FAMILY)

    outcome = fleet_cells.enable_compiled(
        _Pipe(), _Cfg(), publisher=_Publisher(), delegate=True)  # type: ignore[arg-type]

    assert outcome.armed is False
    assert outcome.self_mint is not None


# ---------------------------------------------------------------------------
# 4. THE REGRESSION THIS FIX ALMOST SHIPPED WITH — registering a thunk twice
#    from the same source file must be IDEMPOTENT.
# ---------------------------------------------------------------------------


def test_the_same_declaration_file_registered_twice_is_idempotent() -> None:
    """Measured on qwen-image, and it is the same defect class this whole
    file exists to kill, entering through a different door.

    `discovery/walk.py` imports EVERY submodule of an endpoint package, so an
    in-wheel declaration module is reachable under more than one module name.
    A `Compile` re-registered from a second execution is VALUE-equal (msgspec
    Struct) so it was always idempotent; two THUNKS are two distinct function
    objects, so the naive `==` raised `DeclarationError` on the second
    registration and the walk died with `EndpointImportError` — a pod that
    does not boot, for a compile feature.
    """
    import importlib.util
    from pathlib import Path

    src = Path(__file__).resolve().parent / "harness" / "blocked_declaration_parts.py"

    def _exec_under(name: str):
        spec = importlib.util.spec_from_file_location(name, src)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    reset_export_declarations()
    try:
        first = _exec_under("parts_alias_one")
        second = _exec_under("parts_alias_two")
        assert first.build_declaration is not second.build_declaration, (
            "the two module objects must really be distinct, or this test "
            "proves nothing")

        register_export_declaration(first.build_declaration, family=FAMILY)
        # Must NOT raise: same function, same source file, same declaration.
        register_export_declaration(second.build_declaration, family=FAMILY)

        assert registered_export_families() == (FAMILY,)
        assert export_declaration(FAMILY).family == FAMILY
    finally:
        reset_export_declarations()


def test_a_genuinely_different_thunk_still_conflicts_by_name() -> None:
    """Source identity must not be so loose that two real declarations for
    one family register over each other silently."""
    reset_export_declarations()
    try:
        register_export_declaration(_raising_thunk, family=FAMILY)
        with pytest.raises(DeclarationError):
            register_export_declaration(_other_thunk, family=FAMILY)
        # ...and the owner can still say so explicitly.
        register_export_declaration(_other_thunk, family=FAMILY, replace=True)
    finally:
        reset_export_declarations()


def _other_thunk():
    from harness.blocked_declaration_parts import build_declaration

    return build_declaration()


# ---------------------------------------------------------------------------
# 5. THE WALKER DOOR — an in-wheel declaration is imported by DISCOVERY, not
#    only by main.py, so `import_export_declaration` cannot be the only guard.
# ---------------------------------------------------------------------------


def test_a_package_whose_declaration_submodule_refuses_still_serves() -> None:
    """The shape ltx-2.3 / qwen-image / z-image actually ship.

    `discovery/walk.py` imports EVERY submodule of an endpoint package and
    raises `EndpointImportError` on any failure. So for an in-wheel
    declaration the walker imports it DIRECTLY — `main.py`'s guarded import is
    not on that path at all, and the thunk is what actually makes these
    families safe. Proven the only way that means anything: a real worker, a
    real package, a real job over the wire.
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


def test_the_walker_registers_the_blocked_family_and_it_refuses_at_the_mint() -> None:
    """Registered by the walk, and still refusing — both halves, one pass."""
    from gen_worker.registry import collect_endpoints

    # The module may already be in sys.modules from a sibling test, and a
    # cached module does not re-run its registration. Re-assert it so this
    # test states one thing (collection does not detonate the thunk) rather
    # than accidentally testing import order.
    mod = importlib.import_module("harness.blocked_pkg.aot_declaration")
    reset_export_declarations()
    register_export_declaration(mod._declaration, family=FAMILY)
    try:
        specs = collect_endpoints(["harness.blocked_pkg"])

        assert specs, "the package collected no endpoint"
        assert FAMILY in registered_export_families()
        with pytest.raises(MintRefused) as excinfo:
            export_declaration(FAMILY)
        assert "B1-harness" in str(excinfo.value)
    finally:
        reset_export_declarations()
