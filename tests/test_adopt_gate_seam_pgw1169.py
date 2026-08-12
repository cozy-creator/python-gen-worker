"""pgw#1169 — the adopt headroom gate reaches EVERY arm route, and declining
serves eager instead of taking the process.

pgw#1164 gated the SELF-MINT adopt only. A BOOT adopt that could not fit the
cell had no way to decline: it ran the load and died. Because an AOTI load that
exhausts the card SIGSEGVs rather than raising, nothing downstream can catch it
— so one bad cell becomes a fleet-wide crash loop, every serving pod on the
release taking the same death.

This is §4.31 applied where it could not previously reach, not new policy: a
cell-attributable failure de-arms and serves eager, and an adopt that cannot fit
is that failure in its worst-behaved form.

THE TWO DIRECTIONS BOTH MATTER, and the second one more than usual: an
over-refusing adopt gate would silently return the whole fleet to eager serving
and look like nothing at all — the AOT program's failure mode with the best
manners. So every "refuses" row here is paired with a "still adopts" row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from gen_worker import mint_budget
from gen_worker.cell_adopt import AdoptOutcome
from gen_worker.models import provision

_GIB = 1 << 30


@pytest.fixture(autouse=True)
def _clean_state():
    mint_budget._ADOPT_PEAKS.clear()
    mint_budget._ADOPT_DECLINED.clear()
    yield
    mint_budget._ADOPT_PEAKS.clear()
    mint_budget._ADOPT_DECLINED.clear()


_META: Dict[str, Any] = {
    "family": "sdxl",
    "weight_lane": "w8a8",
    "mode": "",
    "cell_key": "ck1-testtesttest",
    "entries": {f"unet/e{i}": {"target": "unet"} for i in range(36)},
}


def _device(free_gib: float, allocated_gib: float = 32.0, peak_gib: float = 36.0):
    allocated = int(allocated_gib * _GIB)
    peak = int(peak_gib * _GIB)
    measured = max(0, peak - allocated)
    return mint_budget._DeviceRead(
        free_bytes=int(free_gib * _GIB),
        allocated=allocated,
        cache_slack=0,
        measured_activation=measured,
        activation=max(
            measured,
            int(allocated * mint_budget._UNMEASURED_ACTIVATION_FRACTION)),
    )


def _install(monkeypatch, events: List[Tuple[str, str, str]],
             loads: List[int], *, free_gib: float):
    from gen_worker import aot_serve

    monkeypatch.setattr(mint_budget, "_read_device", lambda _d: _device(free_gib))
    monkeypatch.setattr(mint_budget, "device_of", lambda _p: 0)
    monkeypatch.setattr(mint_budget, "adopt_watermark", lambda _d: (0, 0))

    def _enable(*_a: Any, **_k: Any):
        loads.append(1)
        return AdoptOutcome.hit("armed")

    monkeypatch.setattr(aot_serve, "enable", _enable)
    monkeypatch.setattr(aot_serve, "armed_metadata", lambda _p: dict(_META))
    monkeypatch.setattr(aot_serve, "unwrap", lambda _p: None)
    monkeypatch.setattr(provision, "gate_cell_numerics", lambda *a, **k: True)
    monkeypatch.setattr(
        provision.activity_mod, "emit_event",
        lambda kind, detail, phase="", **_k: events.append((kind, phase, detail)))


def _arm(tmp_path: Path, **kw: Any):
    return provision.arm_aot(
        object(), type("Cfg", (), {"family": "sdxl"})(), None,
        tmp_path / "cell.tar.gz", 0, dict(_META), **kw)


# --------------------------------------------------------------------------
# Direction 1 — a pod that CANNOT fit serves eager instead of dying.
# --------------------------------------------------------------------------


def test_a_BOOT_ADOPT_that_cannot_fit_DECLINES_instead_of_loading(
    monkeypatch, tmp_path,
) -> None:
    """THE LOAD-BEARING ROW. `verify_numerics=False` is the boot adopt — the
    route the whole serving fleet takes. The load must NOT be reached."""
    events: List[Tuple[str, str, str]] = []
    loads: List[int] = []
    _install(monkeypatch, events, loads, free_gib=15.0)
    mint_budget.record_adopt_peak("sdxl", "w8a8", int(21.0 * _GIB))

    outcome = _arm(tmp_path, verify_numerics=False)

    assert loads == [], (
        "aot_serve.enable RAN ANYWAY — on a real pod this is the SIGSEGV the "
        "gate exists to prevent, and it would repeat on every pod in the fleet")
    assert not outcome.armed
    assert outcome.reason == "insufficient_adopt_vram", outcome.reason


def test_the_refusal_is_TYPED_and_EMITTED(monkeypatch, tmp_path) -> None:
    """A fleet-wide fact must be visible, not inferred from an absence of
    adoptions."""
    events: List[Tuple[str, str, str]] = []
    loads: List[int] = []
    _install(monkeypatch, events, loads, free_gib=15.0)
    mint_budget.record_adopt_peak("sdxl", "w8a8", int(21.0 * _GIB))

    _arm(tmp_path, verify_numerics=False)

    rows = [(p, d) for k, p, d in events if k == "cell_adopt_declined"]
    assert len(rows) == 1, f"expected one typed refusal row, got {events!r}"
    phase, detail = rows[0]
    assert phase == "insufficient_adopt_vram"
    assert "entries=36" in detail, detail
    assert "serves EAGER" in detail


def test_the_refusal_is_STICKY_across_later_arms(monkeypatch, tmp_path) -> None:
    """§4.31: a pod must not re-run a load it has already decided it cannot
    survive. The second arm refuses even though the card now looks roomy."""
    events: List[Tuple[str, str, str]] = []
    loads: List[int] = []
    _install(monkeypatch, events, loads, free_gib=15.0)
    mint_budget.record_adopt_peak("sdxl", "w8a8", int(21.0 * _GIB))
    _arm(tmp_path, verify_numerics=False)
    assert mint_budget.adopt_declined("sdxl", "w8a8")

    # The card frees up — a transient dip in someone else's residency is not
    # evidence that the doomed load has become survivable.
    monkeypatch.setattr(mint_budget, "_read_device", lambda _d: _device(400.0))
    outcome = _arm(tmp_path, verify_numerics=False)

    assert loads == [], "a sticky refusal was retried — that is the crash loop"
    assert outcome.reason == "insufficient_adopt_vram"


def test_stickiness_is_scoped_to_the_family_and_lane(
    monkeypatch, tmp_path,
) -> None:
    """One family's refusal must not mute another's adopt."""
    events: List[Tuple[str, str, str]] = []
    loads: List[int] = []
    _install(monkeypatch, events, loads, free_gib=15.0)
    mint_budget.record_adopt_peak("sdxl", "w8a8", int(21.0 * _GIB))
    _arm(tmp_path, verify_numerics=False)

    assert mint_budget.adopt_declined("sdxl", "w8a8")
    assert not mint_budget.adopt_declined("micro-diffusion", "plain")
    assert not mint_budget.adopt_declined("sdxl", "")


# --------------------------------------------------------------------------
# Direction 2 — THE OVER-REFUSAL FENCE. A gate that never adopts looks like
# nothing at all, and would return the whole fleet to eager serving silently.
# --------------------------------------------------------------------------


def test_a_pod_that_CAN_fit_STILL_ADOPTS(monkeypatch, tmp_path) -> None:
    """THE FENCE. If this row ever goes green while the others do too by
    refusing everything, the AOT program has silently ended."""
    events: List[Tuple[str, str, str]] = []
    loads: List[int] = []
    _install(monkeypatch, events, loads, free_gib=64.0)
    mint_budget.record_adopt_peak("sdxl", "w8a8", int(21.0 * _GIB))

    outcome = _arm(tmp_path, verify_numerics=False)

    assert loads == [1], "a card with room did not reach the load"
    assert outcome.armed, "a card with room did not adopt"
    assert not [k for k, _p, _d in events if k == "cell_adopt_declined"]


def test_an_UNMEASURED_family_on_a_roomy_card_still_adopts(
    monkeypatch, tmp_path,
) -> None:
    """No banked figure and room to spare must NOT refuse — the unmeasured
    floor is `2 * activation`, never a guess about the entry runners."""
    events: List[Tuple[str, str, str]] = []
    loads: List[int] = []
    _install(monkeypatch, events, loads, free_gib=64.0)

    outcome = _arm(tmp_path, verify_numerics=False)

    assert loads == [1] and outcome.armed


def test_an_UNPROBEABLE_device_never_blocks_an_adopt(
    monkeypatch, tmp_path,
) -> None:
    """A CPU rig keeps today's behaviour exactly — every budget in this module
    fits by construction with no CUDA to read."""
    events: List[Tuple[str, str, str]] = []
    loads: List[int] = []
    _install(monkeypatch, events, loads, free_gib=1.0)
    monkeypatch.setattr(mint_budget, "_read_device", lambda _d: None)
    mint_budget.record_adopt_peak("sdxl", "w8a8", int(900.0 * _GIB))

    outcome = _arm(tmp_path, verify_numerics=False)

    assert loads == [1] and outcome.armed


def test_the_SELF_MINT_route_is_gated_by_the_same_authority(
    monkeypatch, tmp_path,
) -> None:
    """One authority, many callers. The self-mint route reaches the same
    decision through the same function, so the two cannot drift."""
    events: List[Tuple[str, str, str]] = []
    loads: List[int] = []
    _install(monkeypatch, events, loads, free_gib=15.0)
    mint_budget.record_adopt_peak("sdxl", "w8a8", int(21.0 * _GIB))

    outcome = _arm(tmp_path, verify_numerics=True)

    assert loads == []
    assert outcome.reason == "insufficient_adopt_vram"
