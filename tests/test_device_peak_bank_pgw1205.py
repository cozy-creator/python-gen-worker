"""What a compile costs a card, banked per GRAPH CLASS with provenance.

An estimate may never act as a floor a measurement is not allowed to correct,
and an absent measurement means NO EVIDENCE. A mint is activation-scale, not
weight-scale, so the honest number is what a compile actually costs a card.

WHY THE MACHINE AND NOT THE CELL MANIFEST
-----------------------------------------
The manifest is written when a cell SEALS. **The mint that most needs measuring
is the one that OOMed and sealed nothing**, so a manifest-only bank has a writer
that dies exactly when the interesting data exists. And the consumer is local:
K is decided in the mint child, on this card, so a fleet table reachable only
through a hub is the wrong shape to read it from — and cozy-local has no hub.

So: two sinks, one measurement, each answering only the question it can. The
machine banks the row for "what did this cost HERE"; `aot_mint_phases` carries
the identical bytes to the hub for the fleet view. The artifact schema is
untouched.

WHAT THIS IS NOT: it sizes nothing. `entry_workers` is still f(cores, measured
child RSS), and no width, placement or admission decision reads a banked row.
Wiring one in would re-create the prediction layer §4.33 deleted.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from gen_worker import mint_delegate, mint_workers


@pytest.fixture(autouse=True)
def _clean_bank() -> Any:
    mint_workers._forget_device_peaks()
    yield
    mint_workers._forget_device_peaks()


PROVENANCE: Dict[str, str] = {
    "card": "h100-80gb-hbm3",
    "sm": "sm_90",
    "toolchain": "e4b2b170438af354",
    "gen_worker": "0.113.2",
    "phase": "entry_compile",
}


def _phases(rows: Dict[str, Dict[str, int]], **prov: str) -> Dict[str, Any]:
    """A mint phase table shaped exactly as `_pool_facts` writes one."""
    return {"pool": {
        "peak_child_device_bytes": max(
            [r.get("reserved_bytes", 0) for r in rows.values()] or [0]),
        "entry_device_peaks": rows,
        "device_peak_provenance": {**PROVENANCE, **prov},
    }}


def _key(graph_class: str, **over: str) -> mint_workers.DevicePeakKey:
    base = dict(
        graph_class=graph_class, card=PROVENANCE["card"], sm=PROVENANCE["sm"],
        toolchain=PROVENANCE["toolchain"], gen_worker=PROVENANCE["gen_worker"],
        weight_lane="w8a8", phase=PROVENANCE["phase"])
    base.update(over)
    return mint_workers.DevicePeakKey(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The bank: a reading, per class, with every axis that makes it meaningful
# ---------------------------------------------------------------------------


def test_an_unmeasured_row_is_NONE_and_not_a_zero() -> None:
    """§4.33: an absent measurement is NO EVIDENCE. A zero-valued row would be
    read as "measured at zero", which is a different fact and a dangerous one."""
    assert mint_workers.entry_device_peak(_key("unet")) is None


def test_both_readings_are_kept_because_the_GAP_is_the_information() -> None:
    """`allocated` is what the compile needed; `reserved` is what the caching
    allocator HELD and therefore what a concurrent sibling could not have. The
    gap between them is fragmentation, which one number hides."""
    mint_workers.record_entry_device_peak(_key("unet"), 2_000_000, 3_500_000)

    peak = mint_workers.entry_device_peak(_key("unet"))
    assert peak is not None
    assert peak.allocated_bytes == 2_000_000
    assert peak.reserved_bytes == 3_500_000


def test_the_bank_is_MONOTONE_so_a_lucky_run_cannot_talk_it_down() -> None:
    mint_workers.record_entry_device_peak(_key("unet"), 9_000_000, 9_500_000)
    mint_workers.record_entry_device_peak(_key("unet"), 1_000, 2_000)

    peak = mint_workers.entry_device_peak(_key("unet"))
    assert peak == mint_workers.DevicePeak(9_000_000, 9_500_000)


def test_each_reading_widens_INDEPENDENTLY() -> None:
    """Maxed per field: widening can only ever make a reading more honest, and
    a bank that under-reports is the failure mode that matters."""
    mint_workers.record_entry_device_peak(_key("unet"), 9_000_000, 1_000)
    mint_workers.record_entry_device_peak(_key("unet"), 1_000, 9_500_000)

    assert mint_workers.entry_device_peak(_key("unet")) == \
        mint_workers.DevicePeak(9_000_000, 9_500_000)


@pytest.mark.parametrize(
    "axis,value",
    [("card", "a4500"), ("sm", "sm_86"), ("toolchain", "deadbeefdeadbeef"),
     ("gen_worker", "0.114.0"), ("weight_lane", "bf16"),
     ("phase", "export")],
)
def test_every_provenance_axis_SEPARATES_a_reading(axis: str, value: str) -> None:
    """The number is meaningless without its conditions. The same graph class
    costs a different amount on a different card, under a different toolchain,
    at a different weight lane — and an EXPORT high-water and an entry COMPILE
    high-water are different questions about the same card, which is why the
    phase is on the key and not implied."""
    mint_workers.record_entry_device_peak(_key("unet"), 5_000, 6_000)

    assert mint_workers.entry_device_peak(_key("unet", **{axis: value})) is None
    assert mint_workers.entry_device_peak(_key("unet")) is not None


def test_graph_classes_do_not_share_a_row() -> None:
    """The whole reason this exists: `peak_child_device_bytes` was ONE number
    for a whole cell — 18 classes on sdxl — which cannot answer "what does this
    class cost"."""
    mint_workers.record_entry_device_peak(_key("unet"), 5_000, 6_000)
    mint_workers.record_entry_device_peak(_key("vae.decode"), 100, 200)

    unet = mint_workers.entry_device_peak(_key("unet"))
    vae = mint_workers.entry_device_peak(_key("vae.decode"))
    assert unet is not None and unet.reserved_bytes == 6_000
    assert vae is not None and vae.reserved_bytes == 200


def test_a_row_with_no_subject_is_refused() -> None:
    """It could not be looked up, and it would silently accumulate every class
    into one entry."""
    mint_workers.record_entry_device_peak(_key(""), 5_000, 6_000)
    assert mint_workers.device_peak_rows() == {}


def test_an_empty_reading_banks_nothing() -> None:
    mint_workers.record_entry_device_peak(_key("unet"), 0, 0)
    assert mint_workers.device_peak_rows() == {}


def test_the_returned_rows_are_a_COPY() -> None:
    """The bank is append-and-widen only; a caller must not be able to lower a
    reading by holding the map."""
    mint_workers.record_entry_device_peak(_key("unet"), 5_000, 6_000)
    rows = mint_workers.device_peak_rows()
    rows.clear()
    assert mint_workers.entry_device_peak(_key("unet")) is not None


# ---------------------------------------------------------------------------
# THE TWO-SINK PROOF
# ---------------------------------------------------------------------------


class _Outcome:
    """The shape `build_cell` reads: a report on a terminus, a snapshot always."""

    def __init__(self, report: Any = None, partial: Any = None) -> None:
        self.report = report
        self.partial_phases = partial or {}


class _Report:
    def __init__(self, phases: Dict[str, Any]) -> None:
        self.mint_phases = phases


def test_a_FAILED_mint_leaves_a_bank_row() -> None:
    """The deliverable, and the argument that chose this design over the cell
    manifest: a mint that OOMs seals no cell and writes no manifest, and it is
    the attempt whose reading the next attempt most needs.

    The killed child writes no report at all — only the phase SNAPSHOT — so the
    row has to come off that path or it does not exist.
    """
    rows = {"unet": {"allocated_bytes": 7_000_000, "reserved_bytes": 8_100_000}}
    killed = _Outcome(report=None, partial=_phases(rows))
    assert killed.report is None, "a killed child writes no report — only a snapshot"

    banked = mint_delegate._bank_device_peaks(
        killed.partial_phases, weight_lane="w8a8")

    assert banked == 1, "a killed mint's snapshot must still bank"
    peak = mint_workers.entry_device_peak(_key("unet"))
    assert peak == mint_workers.DevicePeak(7_000_000, 8_100_000)


def test_a_mint_that_REACHED_a_terminus_banks_from_its_report() -> None:
    """The other source. `build_cell` drains BOTH — the report when the child
    reached a terminus under its own power, the snapshot always — so neither
    kind of ending loses its measurement."""
    rows = {"unet": {"allocated_bytes": 1_500, "reserved_bytes": 2_500}}
    refused = _Outcome(report=_Report(_phases(rows)), partial={})

    banked = mint_delegate._bank_device_peaks(
        refused.report.mint_phases, weight_lane="w8a8")

    assert banked == 1
    assert mint_workers.entry_device_peak(_key("unet")) == \
        mint_workers.DevicePeak(1_500, 2_500)


def test_the_HUB_row_and_the_LOCAL_row_are_the_same_bytes() -> None:
    """The two sinks cannot disagree, because they are not two measurements.

    `_bank_device_peaks` reads the rows out of the mint phase table, and
    `_emit_aot_phases` sends that SAME table to the hub. So the proof is
    identity, not reconciliation: whatever the hub is told, this machine
    banked, field for field.
    """
    rows = {
        "unet": {"allocated_bytes": 2_000_000, "reserved_bytes": 3_500_000},
        "vae.decode": {"allocated_bytes": 400, "reserved_bytes": 900},
    }
    phases = _phases(rows)

    banked = mint_delegate._bank_device_peaks(phases, weight_lane="w8a8")
    assert banked == 2

    # What the hub receives, verbatim out of the same table.
    hub = phases["pool"]["entry_device_peaks"]
    prov = phases["pool"]["device_peak_provenance"]

    for graph_class, hub_row in hub.items():
        local = mint_workers.entry_device_peak(
            mint_workers.DevicePeakKey(
                graph_class=graph_class, card=prov["card"], sm=prov["sm"],
                toolchain=prov["toolchain"], gen_worker=prov["gen_worker"],
                weight_lane="w8a8", phase=prov["phase"]))
        assert local is not None, f"{graph_class} reached the hub and not the bank"
        assert local.allocated_bytes == hub_row["allocated_bytes"]
        assert local.reserved_bytes == hub_row["reserved_bytes"]


def test_a_table_with_no_device_rows_banks_nothing_and_says_so() -> None:
    """"No rows" and "banked nothing" must be distinguishable — the same reason
    `entry_device_peak` returns None rather than a zero."""
    assert mint_delegate._bank_device_peaks({}, weight_lane="w8a8") == 0
    assert mint_delegate._bank_device_peaks(
        {"pool": {"peak_child_rss_bytes": 5}}, weight_lane="w8a8") == 0
    assert mint_workers.device_peak_rows() == {}


def test_a_malformed_table_never_costs_a_mint() -> None:
    """This runs on the outcome path of a mint that may already be failing; it
    must not add a second failure to the one being reported."""
    for junk in (None, {"pool": None}, {"pool": {"entry_device_peaks": "no"}},
                 {"pool": {"entry_device_peaks": {"unet": "no"}}}):
        assert mint_delegate._bank_device_peaks(junk, weight_lane="w") == 0


def test_the_parent_supplies_the_LANE_because_the_child_does_not_state_it() -> None:
    """A fact belongs to whoever knows it first-hand. The child states card,
    sm, toolchain, version and phase — everything about the process that ran on
    the card. The lane is the parent's, and it is the axis it already keys the
    RSS bank by."""
    rows = {"unet": {"allocated_bytes": 10, "reserved_bytes": 20}}
    assert "weight_lane" not in _phases(rows)["pool"]["device_peak_provenance"]

    mint_delegate._bank_device_peaks(_phases(rows), weight_lane="bf16")

    assert mint_workers.entry_device_peak(_key("unet", weight_lane="bf16")) \
        is not None
    assert mint_workers.entry_device_peak(_key("unet", weight_lane="w8a8")) is None


# ---------------------------------------------------------------------------
# The fence: this measures, it does not size
# ---------------------------------------------------------------------------


def test_no_width_or_placement_decision_reads_the_bank() -> None:
    """§4.33's whole lesson. `entry_workers` is f(cores, measured child RSS);
    reintroducing a device divisor is what pgw#1175 deleted, and a banked
    number is exactly the shape that invites it back.

    Asserted structurally rather than by comment: nothing outside this module
    and its tests may call the reader.
    """
    import pathlib
    import re

    # The READER only. `record_entry_device_peak` contains the reader's name as
    # a substring, and the writer is the whole point — a guard that cannot tell
    # them apart names its own plumbing and then gets deleted for crying wolf.
    reader = re.compile(r"(?<!record_)entry_device_peak\s*\(")
    root = pathlib.Path(mint_workers.__file__).resolve().parent
    callers = [
        path.name for path in sorted(root.rglob("*.py"))
        if path.name != "mint_workers.py"
        and reader.search(path.read_text())
    ]
    assert callers == [], (
        f"{callers} reads the device bank — if a width, placement or admission "
        f"decision now divides by it, that is mint_budget returning and §4.33 "
        f"deleted it on measured evidence")
