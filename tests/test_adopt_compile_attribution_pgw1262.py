"""The ADOPT's device spans are named as COMPILES, so their deaths are not
charged to the tenant.

`postmortem.attribute_signal_death` already implements the right rule — *"when
a ``compile`` marker is in flight the streak is recorded ONLY against the
compile marker(s) … charging the request's function would refuse serving and
condemn the SKU for a software bug"* — and `executor` already acts on it
(pgw#714: a prior compile-attributed death reboots the pod EAGER-ONLY instead of
re-running the native crash). Both were live before this change. What was
missing is that **the adopt never wrote such a marker**: `hot_swap` was the
SDK's only writer, so an adopt death fell through to the tenant's function, the
eager-only reboot never fired, and the pod re-minted the same key and
crash-looped.

Measured 2026-08-14 on the standing master stack (pgw#1255): a z-image pod died
in `probe_compiled_graph` -> `gate_compiled_graph_numerics` -> `arm_aot` and reported
`native_crash_streaks: {"generate": 1}` — the tenant's function, for a fault in
the adopt.

These tests drive the REAL `provision.arm_aot` and the REAL postmortem registry.
The doubles are `aot_serve.enable`, `gate_compiled_graph_numerics` and the CUDA reading —
the GPU work this box may not do (§4.33 local-rig bounds: the compile phase is
stubbed, nothing is minted or compiled here).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest

from gen_worker import activity as activity_mod
from gen_worker import mint_workers, postmortem
from gen_worker.compiled_graph_adopt import AdoptOutcome
from gen_worker.models import provision


_META: Dict[str, Any] = {
    "family": "z-image",
    "weight_lane": "lora128",
    "compiled_graph_key": "cg-key-v1-" + "0" * 56,
    "entry": {"name": "transformer/e0", "target": "transformer"},
}


@pytest.fixture(autouse=True)
def _isolate_registries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Iterator[None]:
    """Never touch the real pod paths."""
    monkeypatch.setattr(postmortem, "INFLIGHT_PATH", tmp_path / "inflight.json")
    monkeypatch.setattr(
        postmortem, "CRASH_REGISTRY_PATH", tmp_path / "crashes.json")
    postmortem.clear_inflight()
    yield
    postmortem.clear_inflight()


def _live_compile_markers() -> List[Dict[str, Any]]:
    """The compile markers a dying process would find, WITHOUT consuming them."""
    import json

    try:
        raw = json.loads(Path(postmortem.INFLIGHT_PATH).read_text())
    except (OSError, ValueError):
        return []
    return [
        row for row in (raw.get("active") or [])
        if isinstance(row, dict)
        and str(row.get("kind") or "") == postmortem.COMPILE_KIND
    ]


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_enable: Optional[Callable[[], None]] = None,
    on_gate: Optional[Callable[[], None]] = None,
    armed: bool = True,
    gate_passes: bool = True,
    enable_raises: Optional[BaseException] = None,
) -> None:
    """Real arm_aot; doubled CUDA reading, load and §4.32 gate."""
    from gen_worker import aot_serve

    monkeypatch.setattr(mint_workers, "adopt_watermark", lambda _d=None: (0, 0))
    monkeypatch.setattr(mint_workers, "device_of", lambda _p: 0)

    def _enable(*_a: Any, **_k: Any) -> AdoptOutcome:
        if on_enable is not None:
            on_enable()
        if enable_raises is not None:
            raise enable_raises
        return (AdoptOutcome.hit("armed") if armed
                else AdoptOutcome.miss("load_failed", "no"))

    def _gate(*_a: Any, **_k: Any) -> bool:
        if on_gate is not None:
            on_gate()
        return gate_passes

    monkeypatch.setattr(aot_serve, "enable", _enable)
    monkeypatch.setattr(aot_serve, "armed_metadata", lambda _p: dict(_META))
    monkeypatch.setattr(aot_serve, "unwrap", lambda _p: None)
    monkeypatch.setattr(aot_serve, "disarm_entry", lambda *a, **k: None)
    monkeypatch.setattr(aot_serve, "entry_states", lambda _p: {})
    monkeypatch.setattr(provision, "gate_compiled_graph_numerics", _gate)
    monkeypatch.setattr(activity_mod, "emit_event", lambda *a, **k: None)


def _arm(tmp_path: Path, **kw: Any) -> AdoptOutcome:
    return provision.arm_aot(
        object(), type("Cfg", (), {"family": "z-image"})(), None,
        tmp_path / "cell.tar.gz", 0, dict(_META), **kw)  # cell-spelling: on-disk artifact name read by cozy-local's compiled-graphs CLI


# --------------------------------------------------------------------------
# The two device spans are NAMED while they run.
# --------------------------------------------------------------------------


def test_the_LOAD_span_holds_a_compile_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`aot_serve.enable` is the adopt's first device touch."""
    seen: List[List[Dict[str, Any]]] = []
    _install(monkeypatch, on_enable=lambda: seen.append(_live_compile_markers()))
    _arm(tmp_path)

    assert seen and seen[0], (
        "no compile marker was in flight during the adopt's LOAD — a signal "
        "death here is charged to whatever tenant request is running")
    assert seen[0][0]["function"] == "compile:adopt:z-image"


def test_the_NUMERICS_GATE_span_holds_a_compile_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The §4.32 gate runs two forwards beside the just-loaded runner — the
    adopt's peak device moment, and where the 2026-08-14 death landed."""
    seen: List[List[Dict[str, Any]]] = []
    _install(monkeypatch, on_gate=lambda: seen.append(_live_compile_markers()))
    _arm(tmp_path, verify_numerics=True)

    assert seen and seen[0], (
        "no compile marker was in flight during `gate_compiled_graph_numerics` — this is "
        "the exact frame the z-image pod died in")
    assert seen[0][0]["function"] == "compile:adopt:z-image"


# --------------------------------------------------------------------------
# THE CONSEQUENCE: pgw#714's eager-only reboot now covers the adopt.
# --------------------------------------------------------------------------


def test_a_death_during_the_gate_is_a_COMPILE_crash_not_the_tenant_s(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The production report, reproduced: a tenant request in flight, a fault
    inside the adopt, and the streak must land on the COMPILE."""
    registry = tmp_path / "crashes.json"
    captured: Dict[str, Any] = {}

    def _die() -> None:
        # The process dies HERE. This is what the parent's post-mortem runs.
        captured.update(postmortem.attribute_signal_death(
            signal_name="SIGSEGV",
            inflight_path=postmortem.INFLIGHT_PATH,
            registry_path=registry))

    _install(monkeypatch, on_gate=_die)
    # A tenant request is in flight, exactly as `a4e6f6a4…#1` was.
    postmortem.note_inflight("request", "generate", request_id="a4e6f6a4")
    _arm(tmp_path, verify_numerics=True)

    streaks = captured.get("native_crash_streaks") or {}
    assert "generate" not in streaks, (
        f"the tenant's serving function was charged for an adopt fault: "
        f"{streaks} — this is the misattribution that kept pgw#714's "
        f"eager-only reboot from ever firing")
    assert "compile:adopt:z-image" in streaks, streaks

    rows = postmortem.compile_crash_rows(registry)
    assert rows, (
        "compile_crash_rows() is empty, so executor's "
        "`disable_process_compiles()` never fires and the pod re-mints the "
        "same arm_key and crash-loops")
    assert "compile:adopt:z-image" in rows


# --------------------------------------------------------------------------
# The marker must never LEAK — a stale one misattributes the next death.
# --------------------------------------------------------------------------


def test_the_marker_is_retired_on_a_clean_arm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _install(monkeypatch)
    _arm(tmp_path, verify_numerics=True)
    assert _live_compile_markers() == []


def test_the_marker_is_retired_when_the_load_RAISES(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A leaked marker would make the NEXT unrelated death read as a compile
    crash — the same misattribution, pointed the other way."""
    _install(monkeypatch, enable_raises=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        _arm(tmp_path)
    assert _live_compile_markers() == []


def test_the_marker_is_retired_when_the_gate_REFUSES(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _install(monkeypatch, gate_passes=False)
    _arm(tmp_path, verify_numerics=True)
    assert _live_compile_markers() == []


# --------------------------------------------------------------------------
# Shape guards.
# --------------------------------------------------------------------------


def test_a_boot_adopt_names_the_load_but_runs_no_gate_span(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`verify_numerics=False` is every adopter; it pays the load and no gate,
    so it opens exactly one span and never a vacuous second one."""
    gate_ran: List[int] = []
    seen: List[List[Dict[str, Any]]] = []
    _install(
        monkeypatch,
        on_enable=lambda: seen.append(_live_compile_markers()),
        on_gate=lambda: gate_ran.append(1))
    _arm(tmp_path, verify_numerics=False)

    assert seen and seen[0], "the boot adopt's load must still be named"
    assert gate_ran == [], "§4.32: an adopter runs no numerics gate"
    assert _live_compile_markers() == []


def test_the_marker_names_the_FAMILY_so_the_reboot_report_is_legible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`disable_process_compiles` prints `last=<marker>`; a nameless marker
    would make the pod's eager-only reboot unexplainable."""
    seen: List[List[Dict[str, Any]]] = []
    _install(monkeypatch, on_enable=lambda: seen.append(_live_compile_markers()))
    provision.arm_aot(
        object(), type("Cfg", (), {"family": ""})(), None,
        tmp_path / "cell.tar.gz", 0, {"entry": _META["entry"]})  # cell-spelling: on-disk artifact name read by cozy-local's compiled-graphs CLI

    assert seen[0][0]["function"] == "compile:adopt:unknown"


def test_the_adopt_marker_is_namespaced_away_from_serving_functions() -> None:
    """It must never collide with — or refuse — a real function name."""
    assert postmortem.compile_marker("adopt:z-image").startswith("compile:")
    assert postmortem.compile_crash_rows.__doc__
