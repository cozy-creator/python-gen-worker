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
from gen_worker.config import get_settings
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
    monkeypatch.setenv("GEN_WORKER_PREFER_AOT", "1")
    get_settings.cache_clear()
    monkeypatch.setattr(aot_cells, "discover", lambda *a, **k: None)
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: False)
    monkeypatch.setattr(fleet_cells.cc, "has_compile_target", lambda p, c: True)
    monkeypatch.setattr(fleet_cells.cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(fleet_cells.cc, "delivered_cell_seeded", lambda: False)
    monkeypatch.setattr(fleet_cells.cc, "apply_lora_lane", lambda p, b: None)
    monkeypatch.setattr(fleet_cells.cc, "drop_lora_lane", lambda p: None)
    monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
    monkeypatch.setattr(fleet_cells, "_PENDING", {})
    monkeypatch.setattr(
        fleet_cells.cell_key, "compute",
        lambda *a, **k: type("_K", (), {"digest": "ck5-" + "a" * 56})())
    monkeypatch.setattr(fleet_cells.cc, "mandatory_serving", lambda p: False)
    monkeypatch.setattr(
        fleet_cells.cc, "begin_fleet_mint", lambda p, c, capture: None)
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "w8a8")
    reset_export_declarations()
    yield
    reset_export_declarations()
    get_settings.cache_clear()


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
