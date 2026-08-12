"""pgw#1164 / th#1825 — the ADOPT-ARM gets its own headroom budget.

th#1825, measured: an A40 passed its pre-mint budget, compiled 36/36 sdxl
entries over 1 h 37 m, and then SIGSEGV'd in `finalize` with 1.9 MB free of
47.7 GB. `publish_self_mint` refuses unless `_state["minted"]` is set and the
only writer of that key is `adopt_delegated_mint`, so the ONLY path from
"packed on disk" to "durable" ran through a GPU load that nothing budgeted.

These rows drive the REAL `mint_budget.adopt_headroom`, the REAL
`fleet_cells.adopt_delegated_mint` and the REAL banking registry. The only
double is the CUDA reading itself (`_read_device`), which is the one thing a
box with no usable GPU cannot supply — every decision under test is the
production one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from gen_worker import fleet_cells, mint_budget

_GIB = 1 << 30


@pytest.fixture(autouse=True)
def _clean_bank():
    mint_budget._ADOPT_PEAKS.clear()
    yield
    mint_budget._ADOPT_PEAKS.clear()


def _device(free_gib: float, allocated_gib: float, peak_gib: float):
    """One fake CUDA reading, in the shape `_read_device` returns."""
    allocated = int(allocated_gib * _GIB)
    peak = int(peak_gib * _GIB)
    measured_activation = max(0, peak - allocated)
    return mint_budget._DeviceRead(
        free_bytes=int(free_gib * _GIB),
        allocated=allocated,
        cache_slack=0,
        measured_activation=measured_activation,
        activation=max(
            measured_activation,
            int(allocated * mint_budget._UNMEASURED_ACTIVATION_FRACTION),
        ),
    )


# --------------------------------------------------------------------------
# The budget itself.
# --------------------------------------------------------------------------


def test_unprobeable_device_never_blocks_an_adopt(monkeypatch) -> None:
    """Every budget in this module fits by construction with no CUDA to read.
    A CPU rig must keep today's behaviour exactly."""
    monkeypatch.setattr(mint_budget, "_read_device", lambda _d: None)
    verdict = mint_budget.adopt_headroom("sdxl", "w8a8")
    assert verdict.fits and not verdict.probed


def test_an_unmeasured_adopt_states_its_basis_and_does_not_invent_a_number(
    monkeypatch,
) -> None:
    """THE HONESTY ROW. With no banked peak the need is `2 * activation` — the
    verify's own two-forward working set, every term measured — and `measured`
    is False so no reader can mistake the floor for a prediction."""
    monkeypatch.setattr(
        mint_budget, "_read_device",
        lambda _d: _device(free_gib=15.0, allocated_gib=32.0, peak_gib=36.0))
    verdict = mint_budget.adopt_headroom("sdxl", "w8a8")
    assert verdict.probed and not verdict.measured
    assert verdict.need_bytes == 2 * verdict.activation_bytes, (
        "the unmeasured floor must be exactly two activation working sets — "
        "any other figure means a constant was smuggled in")
    assert verdict.activation_bytes == int(8.0 * _GIB), (
        "activation is `_read_device`'s own figure (max of the measured "
        "high-water and the pre-forward fraction), not a new one")
    assert not verdict.fits  # 15 GiB free against a 16 GiB floor


def test_a_BANKED_peak_becomes_the_ask_and_can_refuse(monkeypatch) -> None:
    """THE ROW THAT SAVES THE SECOND POD. Once one adopt has completed
    anywhere on this (family, lane), the ask is that FACT, and a card that
    cannot meet it refuses before the arm."""
    monkeypatch.setattr(
        mint_budget, "_read_device",
        lambda _d: _device(free_gib=15.0, allocated_gib=32.0, peak_gib=36.0))
    mint_budget.record_adopt_peak("sdxl", "w8a8", int(21.0 * _GIB))
    verdict = mint_budget.adopt_headroom("sdxl", "w8a8")
    assert verdict.measured, "a banked peak must report basis=measured"
    assert verdict.need_bytes == int(21.0 * _GIB)
    assert not verdict.fits, (
        "21 GiB of measured adopt cost against 15 GiB free must NOT fit — "
        "this is exactly the th#1825 shape and it has to refuse")


def test_the_bank_is_monotone(monkeypatch) -> None:
    """An adopt that peaked higher once can peak that high again; a lucky run
    must never lower the ask."""
    mint_budget.record_adopt_peak("sdxl", "w8a8", int(20.0 * _GIB))
    mint_budget.record_adopt_peak("sdxl", "w8a8", int(3.0 * _GIB))
    assert mint_budget.adopt_peak("sdxl", "w8a8") == int(20.0 * _GIB)


def test_the_bank_is_keyed_per_family_and_lane() -> None:
    mint_budget.record_adopt_peak("sdxl", "w8a8", int(20.0 * _GIB))
    assert mint_budget.adopt_peak("sdxl", "") == 0
    assert mint_budget.adopt_peak("wan-2.2", "w8a8") == 0


# --------------------------------------------------------------------------
# The CALL SITE. These are the rows that go red if the gate is disconnected.
# --------------------------------------------------------------------------


class _Pending:
    def __init__(self, target: Path) -> None:
        self.family = "sdxl"
        self.target = target
        self.cache_dir = target.parent
        self.arm_token = "arm2-deadbeef"
        self.arm_key = None
        self.cfg = type("Cfg", (), {"lora_bucket": 0})()
        self.armed_at = 0.0
        self.mint_root = target.parent
        self.publisher = None
        self._state: Dict[str, Any] = {}


@pytest.fixture()
def _pending(tmp_path: Path) -> "_Pending":
    art = tmp_path / "cell.tar.gz"
    art.write_bytes(b"not a real cell - the arm must never be reached")
    return _Pending(art)


def _install(monkeypatch, free_gib: float, events: List[Tuple[str, str, str]],
             arm_calls: List[int]) -> None:
    monkeypatch.setattr(
        mint_budget, "_read_device",
        lambda _d: _device(free_gib=free_gib, allocated_gib=32.0, peak_gib=36.0))
    monkeypatch.setattr(mint_budget, "device_of", lambda _p: 0)
    monkeypatch.setattr(mint_budget, "adopt_watermark", lambda _d: (0, 0))
    monkeypatch.setattr(
        fleet_cells.cc, "cell_base_execution_lane",
        lambda _p: "w8a8")

    def _arm(*_a: Any, **_k: Any):
        arm_calls.append(1)
        return True, {"cell_key": "ck1-x"}, ("", "")

    monkeypatch.setattr(fleet_cells, "_arm_exported_cell", _arm)
    monkeypatch.setattr(
        fleet_cells.activity_mod, "emit_event",
        lambda kind, detail, phase="", **_k: events.append((kind, phase, detail)))


def test_a_measured_shortfall_REFUSES_BEFORE_THE_ARM(monkeypatch, _pending) -> None:
    """THE LOAD-BEARING ROW. The whole point of th#1825 is that the arm is the
    step that cannot survive being wrong, so the refusal must land with
    `_arm_exported_cell` NEVER ENTERED."""
    events: List[Tuple[str, str, str]] = []
    arm_calls: List[int] = []
    _install(monkeypatch, free_gib=15.0, events=events, arm_calls=arm_calls)
    mint_budget.record_adopt_peak("sdxl", "w8a8", int(21.0 * _GIB))

    out = fleet_cells.adopt_delegated_mint(
        object(), _pending, _pending.target)

    assert out is None, "a declined adopt must produce no SelfMint"
    assert arm_calls == [], (
        "THE ARM RAN ANYWAY — the gate is downstream of the thing it protects")
    assert _pending._state["adopt_refusal"][0] == "insufficient_adopt_vram"
    phases = [p for _k, p, _d in events]
    assert "insufficient_adopt_vram" in phases, (
        f"the refusal must be typed and countable; got phases {phases!r}")


def test_the_refusal_publishes_NOTHING(monkeypatch, _pending) -> None:
    """A declined adopt must leave `minted` unset, because that is the key
    `publish_self_mint` refuses on — an unverified cell must never ship."""
    events: List[Tuple[str, str, str]] = []
    _install(monkeypatch, free_gib=15.0, events=events, arm_calls=[])
    mint_budget.record_adopt_peak("sdxl", "w8a8", int(21.0 * _GIB))

    fleet_cells.adopt_delegated_mint(object(), _pending, _pending.target)

    assert "minted" not in _pending._state


def test_an_UNMEASURED_first_adopt_is_NOT_refused(monkeypatch, _pending) -> None:
    """THE NEGATIVE THAT KEEPS THIS HONEST. With nothing banked the budget has
    no fact about the loaded ENTRY RUNNERS, so on a card that clears the
    measured floor it must let the first adopt through and MEASURE it.
    Refusing here would mean refusing on an invented per-entry constant, which
    is the magic number this codebase forbids — 64 GiB free clears the 16 GiB
    floor, and nothing else is known."""
    events: List[Tuple[str, str, str]] = []
    arm_calls: List[int] = []
    _install(monkeypatch, free_gib=64.0, events=events, arm_calls=arm_calls)

    fleet_cells.adopt_delegated_mint(object(), _pending, _pending.target)

    assert arm_calls == [1], (
        "the first adopt on a family was refused on a number nobody has "
        "measured — that is a guess, not a budget")


def test_the_adopt_cost_is_BANKED_AND_EMITTED(monkeypatch, _pending) -> None:
    """One adopt teaches the next, and teaches th#1820's placement floor: the
    bank dies with the pod, so the row has to carry the number too."""
    events: List[Tuple[str, str, str]] = []
    arm_calls: List[int] = []
    _install(monkeypatch, free_gib=64.0, events=events, arm_calls=arm_calls)
    # A real watermark pair this time: 17 GiB of high-water above resident.
    monkeypatch.setattr(
        mint_budget, "adopt_watermark",
        lambda _d: (0, int(17.0 * _GIB)) if arm_calls else (0, 0))

    fleet_cells.adopt_delegated_mint(object(), _pending, _pending.target)

    assert mint_budget.adopt_peak("sdxl", "w8a8") == int(17.0 * _GIB), (
        "the adopt's measured device cost was not banked, so the next adopt "
        "on this pod is as blind as this one was")
    kinds = [k for k, _p, _d in events]
    assert "cell_adopt_budget" in kinds, (
        f"the number never reached the wire; got {kinds!r}")
