"""pgw#1171 / th#1828 — the adopt LOAD travels, so a pod that has never seen a
cell can still refuse an adopt it cannot survive.

pgw#1169 gates the adopt, but on a PROCESS-LOCAL bank — and a fresh pod's bank
is empty. So on the first wave the gate had only its unmeasured floor, which
omits the loaded-runner term: the term that actually exhausts the card
(th#1825, 1.9 MB free of 47.7 GB, SIGSEGV, $0.81). An AOTI load that exhausts
the card takes the process rather than raising, so nothing downstream catches
it and ONE bad cell takes every serving pod on the release.

The loop this closes, end to end:

    arm seam measures `load`  ->  banked locally (pgw#1168/#1171)
      ->  publish-intent carries it  ->  hub stores it on cell_store (th#1828)
      ->  resolve answers with it    ->  a COLD pod seeds its budget
      ->  pgw#1169's gate refuses BEFORE the load

PRODUCER AND CONSUMER ARE PROVEN IN ONE FILE deliberately: a producer with no
consumer is this program's most common defect, and pgw#1168 existed precisely
because an emitter had been wired on one of two paths.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import cell_resolve, mint_budget

_GIB = 1 << 30


@pytest.fixture(autouse=True)
def _clean_state():
    for d in (mint_budget._ADOPT_PEAKS, mint_budget._ADOPT_LOAD,
              mint_budget._FLEET_ADOPT_LOAD):
        d.clear()
    mint_budget._ADOPT_DECLINED.clear()
    yield
    for d in (mint_budget._ADOPT_PEAKS, mint_budget._ADOPT_LOAD,
              mint_budget._FLEET_ADOPT_LOAD):
        d.clear()
    mint_budget._ADOPT_DECLINED.clear()


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


# --------------------------------------------------------------------------
# The WIRE: what the hub says becomes what the worker knows.
# --------------------------------------------------------------------------


def _body(**extra: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "found": True, "family": "sdxl", "cell_key": "ck1-x",
        "cell_ref": "root/family-sdxl#ck1-x", "checkpoint_id": "sha256:cc",
        "content_digest": "sha256:dd", "artifact_path": "cell.tar.gz",
        "size_bytes": 4096, "publisher_org": "org", "publisher_tier": "platform",
        "graph_contract": "gc", "toolchain_digest": "tc", "env_seal_digest": "es",
        "identity_axes": {"lane": "w8a8"}, "sm": "sm_86", "sku": "a40",
        "lane": "w8a8", "receipt": "jws",
        "transport": {"snapshot_digest": "sha256:cc", "files": []},
    }
    body.update(extra)
    return body


def test_the_resolve_answer_carries_the_load() -> None:
    cell = cell_resolve._cell_from(_body(adopt_load_bytes=21 * _GIB))
    assert cell.adopt_load_bytes == 21 * _GIB


def test_an_answer_WITHOUT_the_field_is_no_evidence() -> None:
    """Every cell published before th#1828 is in this case, and the hub OMITS
    the key rather than sending 0. It must read as no evidence, not as a
    measurement of zero."""
    cell = cell_resolve._cell_from(_body())
    assert cell.adopt_load_bytes == 0


# --------------------------------------------------------------------------
# The CONSUMER: a fleet measurement gates a pod that measured nothing itself.
# --------------------------------------------------------------------------


def test_a_FLEET_measurement_gates_a_pod_with_an_EMPTY_bank(monkeypatch) -> None:
    """THE LOAD-BEARING ROW — the whole reason th#1828 exists. Nothing local
    has ever been measured here; without the fleet figure the budget would fall
    back to its floor and admit an adopt that kills the process."""
    # 18 GiB free: the unmeasured floor (2 x 8 GiB activation = 16 GiB) ADMITS,
    # so the ONLY thing that can refuse below is the fleet figure.
    monkeypatch.setattr(mint_budget, "_read_device", lambda _d: _device(18.0))
    assert mint_budget.adopt_headroom("sdxl", "w8a8").fits, (
        "precondition: with no evidence at all this card is admitted")

    mint_budget.note_fleet_adopt_load("sdxl", "w8a8", 21 * _GIB)
    verdict = mint_budget.adopt_headroom("sdxl", "w8a8")

    assert not verdict.fits, (
        "a fleet-measured 21 GiB load against 18 GiB free must REFUSE — this "
        "is exactly the th#1825 shape, on a pod that has never seen the cell")
    assert verdict.measured, "a fleet figure is a measurement, and must say so"
    assert verdict.need_bytes == 21 * _GIB


def test_no_fleet_evidence_leaves_TODAY_S_behaviour_untouched(monkeypatch) -> None:
    """THE DEGRADATION FENCE. Absent is not a pass and not a refusal — it is
    the unmeasured floor, exactly as before th#1828."""
    monkeypatch.setattr(mint_budget, "_read_device", lambda _d: _device(18.0))
    mint_budget.note_fleet_adopt_load("sdxl", "w8a8", 0)  # the hub said nothing

    verdict = mint_budget.adopt_headroom("sdxl", "w8a8")

    assert verdict.fits and not verdict.measured
    assert verdict.need_bytes == 2 * verdict.activation_bytes


def test_a_ZERO_report_records_NOTHING(monkeypatch) -> None:
    """The guard is at the door, not only downstream. A 0 must not create an
    entry at all — a stored 0 is a measurement nobody took, and the next reader
    of this map (there will be one) must not find one."""
    mint_budget.note_fleet_adopt_load("sdxl", "w8a8", 0)
    mint_budget.note_fleet_adopt_load("sdxl", "w8a8", -5)
    assert ("sdxl", "w8a8") not in mint_budget._FLEET_ADOPT_LOAD, (
        "a zero/negative report was BANKED — absent must stay absent in the "
        "store, not merely be filtered by whoever happens to read it next")


def test_the_floor_is_what_NO_evidence_falls_back_to(monkeypatch) -> None:
    """THE DEGRADATION PATH ITSELF. With nothing banked and nothing from the
    fleet, `need` must be the unmeasured floor — not 0, which would admit
    everything, and not a refusal."""
    monkeypatch.setattr(mint_budget, "_read_device", lambda _d: _device(18.0))
    verdict = mint_budget.adopt_headroom("sdxl", "w8a8")
    assert verdict.need_bytes == 2 * verdict.activation_bytes > 0
    assert verdict.fits and not verdict.measured


def test_a_fleet_figure_never_LOWERS_a_local_measurement(monkeypatch) -> None:
    """THE OVER-TRUST FENCE. The local bank is load+verify and the fleet figure
    is load alone, so where both exist the local one is larger and must win —
    a remote number must never talk this pod into an adopt its own hardware
    already proved it cannot do."""
    monkeypatch.setattr(mint_budget, "_read_device", lambda _d: _device(40.0))
    mint_budget.record_adopt_peak("sdxl", "w8a8", 45 * _GIB)
    mint_budget.note_fleet_adopt_load("sdxl", "w8a8", 10 * _GIB)

    verdict = mint_budget.adopt_headroom("sdxl", "w8a8")

    assert verdict.need_bytes == 45 * _GIB
    assert not verdict.fits


def test_the_fleet_bank_is_monotone_and_scoped() -> None:
    mint_budget.note_fleet_adopt_load("sdxl", "w8a8", 21 * _GIB)
    mint_budget.note_fleet_adopt_load("sdxl", "w8a8", 3 * _GIB)
    assert mint_budget.fleet_adopt_load("sdxl", "w8a8") == 21 * _GIB
    assert mint_budget.fleet_adopt_load("sdxl", "") == 0
    assert mint_budget.fleet_adopt_load("micro-diffusion", "w8a8") == 0


# --------------------------------------------------------------------------
# The PRODUCER: what the pod measured is what the publish sends.
# --------------------------------------------------------------------------


def test_the_publish_sends_the_locally_measured_LOAD(monkeypatch) -> None:
    """Producer and consumer in one change: the arm seam banks `load`, and the
    publish carries THAT — never the load+verify total, which only a minting
    pod pays and would over-state what an adopter needs."""
    from gen_worker import fleet_cells

    mint_budget.record_adopt_load("sdxl", "w8a8", 17 * _GIB)
    mint_budget.record_adopt_peak("sdxl", "w8a8", 23 * _GIB)  # load+verify
    sent: List[Tuple[Any, ...]] = []

    class _Pub:
        def publish(self, family, artifact, meta, mint_duration_ms=0,
                    adopt_load_bytes=0):
            sent.append((family, adopt_load_bytes))
            return "ckpt"

    monkeypatch.setattr(fleet_cells, "_note_durable", lambda *a, **k: None)
    monkeypatch.setattr(fleet_cells.activity_mod, "emit_event",
                        lambda *a, **k: None)
    import pathlib
    art = pathlib.Path("/nonexistent/cell.tar.gz")
    t = fleet_cells._publish_async(
        _Pub(), "sdxl", art, {"weight_lane": "w8a8"}, cell_key_digest="ck1-x")
    t.join(timeout=10)

    assert sent == [("sdxl", 17 * _GIB)], (
        f"the publish must carry the LOAD term (17 GiB), not the load+verify "
        f"total (23 GiB) and not 0; got {sent!r}")
