"""pgw#1168 — the adopt's device cost is measured at the ONE seam every arm
route passes, and its two terms are reported apart.

pgw#1164 put the measurement in `fleet_cells.adopt_delegated_mint`, i.e. on the
SELF-MINT adopt only. The boot adopt, the local-store adopt and the re-arm run
the identical `aot_serve.enable` -> `load_and_wrap` and reported nothing — the
"emitter wired on one of N paths" shape this program keeps producing. It lives
in `provision.arm_aot` now, which owns both the load and the §4.32 gate.

Why the SPLIT matters (th#1825): `load` is what EVERY adopting pod pays for the
life of the arm and is the term that decides whether a cell fits the fleet it
was built for; `verify` is the parity gate's two forwards and is paid only by
the minting pod. A boot-adopt row on a 48 GB card is therefore the EMPIRICAL
answer to "does this cell fit", where before there was only arithmetic over a
single blended number.

These rows drive the REAL `arm_aot` and the REAL banking registry. The doubles
are the CUDA reading and `aot_serve.enable`/`gate_cell_numerics` — the GPU work
this box may not do.
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
def _clean_bank():
    mint_budget._ADOPT_PEAKS.clear()
    yield
    mint_budget._ADOPT_PEAKS.clear()


_META: Dict[str, Any] = {
    "family": "sdxl",
    "weight_lane": "w8a8",
    "mode": "",
    "cell_key": "ck1-testtesttest",
    "entries": {f"unet/e{i}": {"target": "unet"} for i in range(36)},
}


def _install(
    monkeypatch,
    events: List[Tuple[str, str, str]],
    *,
    watermarks: List[Tuple[int, int]],
    armed: bool = True,
    gate_passes: bool = True,
):
    """Real arm_aot; doubled CUDA reading, load and gate."""
    from gen_worker import aot_serve

    calls = {"n": 0}

    def _watermark(_device: Any = None) -> Tuple[int, int]:
        i = min(calls["n"], len(watermarks) - 1)
        calls["n"] += 1
        return watermarks[i]

    monkeypatch.setattr(mint_budget, "adopt_watermark", _watermark)
    monkeypatch.setattr(mint_budget, "device_of", lambda _p: 0)
    monkeypatch.setattr(
        aot_serve, "enable",
        lambda *a, **k: (AdoptOutcome.hit("armed") if armed
                         else AdoptOutcome.miss("load_failed", "no")))
    monkeypatch.setattr(aot_serve, "armed_metadata", lambda _p: dict(_META))
    monkeypatch.setattr(aot_serve, "unwrap", lambda _p: None)
    monkeypatch.setattr(
        provision, "gate_cell_numerics", lambda *a, **k: gate_passes)
    monkeypatch.setattr(
        provision.activity_mod, "emit_event",
        lambda kind, detail, phase="", **_k: events.append((kind, phase, detail)))


def _arm(tmp_path: Path, **kw: Any):
    return provision.arm_aot(
        object(), type("Cfg", (), {"family": "sdxl"})(), None,
        tmp_path / "cell.tar.gz", 0, dict(_META), **kw)


def _row(events: List[Tuple[str, str, str]]) -> str:
    rows = [d for k, _p, d in events if k == "cell_adopt_budget"]
    assert len(rows) == 1, f"expected exactly ONE budget row, got {len(rows)}"
    return rows[0]


# --------------------------------------------------------------------------
# GAP 1 — the boot-adopt path reports at all.
# --------------------------------------------------------------------------


def test_the_BOOT_ADOPT_path_emits_a_budget_row(monkeypatch, tmp_path) -> None:
    """THE LOAD-BEARING ROW. `verify_numerics=False` is the boot adopt, the
    local-store adopt and the re-arm — every route that is NOT the self-mint.
    Before pgw#1168 none of them reported, so the cheap measurement on the card
    the fleet actually serves on did not exist."""
    events: List[Tuple[str, str, str]] = []
    _install(monkeypatch, events,
             watermarks=[(10 * _GIB, 10 * _GIB), (0, 27 * _GIB), (0, 27 * _GIB)])

    _arm(tmp_path, verify_numerics=False)

    detail = _row(events)
    assert "load=17.000GiB" in detail, detail
    assert "verified=False" in detail


def test_a_boot_adopt_reports_verify_ZERO_by_construction(
    monkeypatch, tmp_path,
) -> None:
    """An adopter runs no parity gate (§4.32), so its row must attribute the
    whole cost to `load`. That is what makes the row comparable to a serving
    pod's requirement rather than to a minting pod's."""
    events: List[Tuple[str, str, str]] = []
    _install(monkeypatch, events,
             watermarks=[(10 * _GIB, 10 * _GIB), (0, 27 * _GIB), (0, 27 * _GIB)])

    _arm(tmp_path, verify_numerics=False)

    assert "verify=0.000GiB" in _row(events)


# --------------------------------------------------------------------------
# GAP 2 — the two terms are reported APART.
# --------------------------------------------------------------------------


def test_load_and_verify_are_reported_SEPARATELY(monkeypatch, tmp_path) -> None:
    """THE ROW THAT DECIDES WHETHER THE CELL OR THE GATE IS THE PROBLEM.
    A single blended number cannot answer it; `R + load` is the serving
    requirement and `verify` is the minting pod's alone."""
    events: List[Tuple[str, str, str]] = []
    _install(monkeypatch, events,
             watermarks=[(10 * _GIB, 10 * _GIB), (0, 27 * _GIB), (0, 33 * _GIB)])

    _arm(tmp_path, verify_numerics=True)

    detail = _row(events)
    assert "load=17.000GiB" in detail, detail
    assert "verify=6.000GiB" in detail, detail
    assert "adopt_device_peak=23.000GiB" in detail, detail


def test_the_row_names_the_entry_count(monkeypatch, tmp_path) -> None:
    """36 entries and 3 entries must never be pooled — the whole fit question
    is about how the cost scales with entry count."""
    events: List[Tuple[str, str, str]] = []
    _install(monkeypatch, events,
             watermarks=[(0, 0), (0, 5 * _GIB), (0, 5 * _GIB)])
    _arm(tmp_path, verify_numerics=False)
    assert "entries=36" in _row(events)


# --------------------------------------------------------------------------
# Banking, dedup, and the refused arm.
# --------------------------------------------------------------------------


def test_the_seam_feeds_the_BANK_under_the_cell_s_own_lane(
    monkeypatch, tmp_path,
) -> None:
    """The lane key is read from the cell's recorded `weight_lane`, so this
    bank agrees with `mint_budget`'s other three without importing
    compile_cache into provision."""
    events: List[Tuple[str, str, str]] = []
    _install(monkeypatch, events,
             watermarks=[(0, 0), (0, 21 * _GIB), (0, 21 * _GIB)])

    _arm(tmp_path, verify_numerics=False)

    assert mint_budget.adopt_peak("sdxl", "w8a8") == 21 * _GIB


def test_a_REFUSED_arm_still_reports_what_it_paid(monkeypatch, tmp_path) -> None:
    """A refusal is exactly when the number is most worth having — the device
    high-water was paid either way, and th#1825 died at a refusal."""
    events: List[Tuple[str, str, str]] = []
    _install(monkeypatch, events, armed=False,
             watermarks=[(0, 0), (0, 12 * _GIB), (0, 12 * _GIB)])

    _arm(tmp_path, verify_numerics=False)

    detail = _row(events)
    assert "armed=False" in detail
    assert "load=12.000GiB" in detail


def test_a_numerics_REFUSAL_reports_ONCE_with_both_terms(
    monkeypatch, tmp_path,
) -> None:
    """The gate-refused path falls through to the tail emit; the dedup must
    stop it reporting twice."""
    events: List[Tuple[str, str, str]] = []
    _install(monkeypatch, events, gate_passes=False,
             watermarks=[(0, 0), (0, 17 * _GIB), (0, 23 * _GIB)])

    _arm(tmp_path, verify_numerics=True)

    detail = _row(events)  # asserts exactly one
    assert "armed=False" in detail
    assert "verify=6.000GiB" in detail
