"""The adopt's device cost is measured at the ONE seam every arm route passes,
and its two terms are reported apart.

The measurement lives in `provision.arm_aot`, which owns both the load and the
§4.32 gate. Putting it in `fleet_cells.adopt_delegated_mint` covers the
SELF-MINT adopt only — the boot adopt, the local-store adopt and the re-arm run
the identical `aot_serve.enable` -> `arm_entry` and would report nothing.

Why the SPLIT matters: `load` is what EVERY adopting pod pays for the life of
the arm and is the term that decides whether a cell fits the fleet it was built
for; `verify` is the parity gate's two forwards and is paid only by the minting
pod. A boot-adopt row on a 48 GB card is therefore the EMPIRICAL answer to "does
this cell fit", not arithmetic over a single blended number.

These rows drive the REAL `arm_aot` and the REAL banking registry. The doubles
are the CUDA reading and `aot_serve.enable`/`gate_cell_numerics` — the GPU work
this box may not do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from gen_worker import mint_workers
from gen_worker.cell_adopt import AdoptOutcome
from gen_worker.models import provision

_GIB = 1 << 30


_META: Dict[str, Any] = {
    "family": "sdxl",
    "weight_lane": "w8a8",
    "compiled_graph_key": "cg-key-v1-" + "0" * 56,
    "entry": {"name": "unet/e0", "target": "unet"},
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

    monkeypatch.setattr(mint_workers, "adopt_watermark", _watermark)
    monkeypatch.setattr(mint_workers, "device_of", lambda _p: 0)
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
    rows = [d for k, _p, d in events if k == "compiled_graph_adopt_budget"]
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


def test_the_row_names_ONE_entry_because_that_is_all_an_artifact_carries(
    monkeypatch, tmp_path,
) -> None:
    """INVERTED by pgw#1176. It read: *"36 entries and 3 entries must never be
    pooled — the whole fit question is about how the cost scales with entry
    count"*, and asserted `entries=36`. An artifact now carries ONE graph
    class, so a 36-entry envelope is a shape production cannot construct and
    this row is always `entries=1` — the point, not a degenerate case.

    The count is still READ from the metadata rather than printed as a
    constant, which is what the row below pins."""
    events: List[Tuple[str, str, str]] = []
    _install(monkeypatch, events,
             watermarks=[(0, 0), (0, 5 * _GIB), (0, 5 * _GIB)])
    _arm(tmp_path, verify_numerics=False)
    assert "entries=1" in _row(events)


def test_each_armed_CLASS_gets_ITS_OWN_row_never_a_pooled_one(
    monkeypatch, tmp_path,
) -> None:
    """The surviving guard from the inversion above. *"36 entries and 3 entries
    must never be pooled"* is still true and is now structural rather than a
    field: N classes arm through N `arm_aot` calls and emit N rows, so a
    refused class shows up as its own missing row instead of disappearing into
    one blended total."""
    events: List[Tuple[str, str, str]] = []
    _install(monkeypatch, events,
             watermarks=[(0, 0), (0, 5 * _GIB), (0, 5 * _GIB)])
    _arm(tmp_path, verify_numerics=False)
    _arm(tmp_path, verify_numerics=False)

    rows = [d for k, _p, d in events if k == "compiled_graph_adopt_budget"]
    assert len(rows) == 2, rows
    assert all("entries=1" in r for r in rows), rows


# --------------------------------------------------------------------------
# Banking, dedup, and the refused arm.
# --------------------------------------------------------------------------


def test_the_row_is_keyed_by_the_cell_s_OWN_recorded_lane(
    monkeypatch, tmp_path,
) -> None:
    """The lane comes off the cell's recorded `weight_lane`, so a reader can
    line these rows up per (family, lane) without provision importing
    compile_cache.

    this row USED to feed `mint_budget._ADOPT_PEAKS`, which was then
    divided into free VRAM to refuse the next adopt. The bank is deleted; the
    ROW is not. It is the only instrument that answers where a loaded cell's
    device memory goes, which is exactly what §4.33's ~8 GiB target has to be
    checked against — a measurement, kept, with the prediction it fed removed.
    """
    events: List[Tuple[str, str, str]] = []
    _install(monkeypatch, events,
             watermarks=[(0, 0), (0, 21 * _GIB), (0, 21 * _GIB)])

    _arm(tmp_path, verify_numerics=False)

    detail = _row(events)
    assert "family=sdxl lane=w8a8" in detail
    assert "adopt_device_peak=21.000GiB" in detail


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
