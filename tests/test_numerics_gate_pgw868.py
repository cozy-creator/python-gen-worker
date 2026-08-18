"""THE NUMERICS GATE. Everything that runs without a card.

The requirement: a compiled graph must be refusable for being WRONG, not only for being
UNUSABLE.

The trap this file keeps shut: `numerics_ladder.gate()`
opens `if comparison is None: return None`. Wiring the call without producing a
measurement passes EVERY compiled graph, always, while looking correct in the diff and in
the call graph. So the tests that matter here are not "is `gate` called" — they
are:

* a compiled graph BELOW its declared floor does not arm, and the pipeline is left eager;
* a compiled graph BETWEEN floor and warn arms *with the warning on the wire*;
* a compiled graph that cannot be MEASURED does not arm either — "nobody could ask" is
  not "it passed";
* and the verdict is bisectable to ONE named axis (one entry x one shape row x
  one lane) — a whole-compiled graph fail nobody can split invites a confident wrong
  diagnosis.

Everything below drives the REAL arm path — `provision.arm_aot` ->
`aot_serve.enable` -> stage/verify/bind/wrap -> the gate — against a real packed
artifact, a real registered `Compile` declaration, real `torch` tensors and the
mint's own input builder. The ONE substitution is `_load_package`: an AOTI
`.so` needs a GPU, and it is the only piece deferred to the pod. The subject it
returns is a CALIBRATED blend of the eager output at an exact cosine, which is
what lets a test name the rung it is aiming at; the real numbers come from a
pod, and no test here may be cited as evidence about a real compiled graph's numerics.
"""

from __future__ import annotations

# ``declared`` and ``events`` are imported pytest fixtures, so test parameters
# intentionally reuse their fixture names.
# ruff: noqa: F811

from typing import Any, List

import pytest
import torch

from gen_worker import aot_serve as aot
from gen_worker import numerics_probe
from gen_worker.api.export_contract import reset_export_declarations

# The rig this file grew moved to `tests/harness/exported_compiled_graph.py`.
# Three other modules already imported it from here as `rig868`, and the
# adopt-path rig needs the same packed artifact — a shared vehicle belongs
# where shared vehicles live. Nothing about it changed; the names below are
# the same objects.
from harness.exported_compiled_graph import (  # noqa: F401 — imported pytest fixtures
    FAMILY, FLOOR, ROWS, RUNTIME, TARGET, WARN,
    ProbeDenoiser, ProbePackage, ProbePipeline,
    arm, artifact, blend, declaration, declared, entry_name, events,
    metadata, numerics_rows,
)


# ---------------------------------------------------------------------------
# THE RED: a compiled graph below its declared floor must not serve
# ---------------------------------------------------------------------------


def test_a_cell_below_its_declared_floor_REFUSES_TO_ARM(
        tmp_path, monkeypatch, declared, events):
    """The headline. Before this gate existed, this exact compiled graph ARMED.

    RED PROOF (recorded rather than asserted, because it is a property of the
    absent call site): with the `gate_compiled_graph_numerics(...)` call removed from
    `provision.arm_aot`, this test fails on its first assertion — `arm_aot`
    returns True and the 0.99-cosine compiled graph serves every subsequent request.
    """
    packages = {entry_name(h, w): ProbePackage(cosine=0.99) for h, w in ROWS}
    pipeline, module, outcomes = arm(tmp_path, monkeypatch, declared, packages)

    # One verdict PER GRAPH CLASS. Every class here is 1% off, so
    # every class refuses — and the pipeline serves nothing compiled, which is
    # now a CONSEQUENCE of every entry refusing rather than a rule that one
    # refusal condemns the rest.
    assert [o.armed for o in outcomes] == [False, False], (
        "a compiled graph that lost 1% of the output's direction armed")
    assert aot.is_armed(pipeline) is False
    assert aot.armed_entries(pipeline) == {}
    assert isinstance(module.forward(torch.zeros(8, 8), torch.tensor(1.0)),
                      torch.Tensor)

    # The adopt ledger cannot need closing, because nothing is
    # announced until the arm is FINAL. `enable` used to say `armed` before
    # this gate ran, so a reader counting armed adoptions over-counted every
    # numerics refusal and a second "retraction" row existed only to correct
    # the first. The arm returns ONE outcome, with the gate's verdict in it.
    assert outcomes[0].reason == "numerics_refused"
    assert "is not published" in outcomes[0].detail

    rows = numerics_rows(events)
    assert rows, "a refused compiled graph said nothing on the wire"
    detail, phase = rows[-1]
    assert phase == "refused"
    # The verdict carries its inputs: the floor, its source, and the AXIS.
    assert "floor=0.995" in detail and "warn=0.999" in detail
    assert "source=declared" in detail
    assert entry_name(*ROWS[0]) in detail or entry_name(*ROWS[1]) in detail
    assert "0.99" in detail


def test_the_gray_band_CONFESSES_AND_REFUSES_TO_PUBLISH(
        tmp_path, monkeypatch, declared, events):
    """The gray band still confesses — a fleet-wide rate is only countable from
    activity records — but since §4.32 it does not SHIP.

    pgw#1141 moved this gate to the minting pod and made it strict: identical
    or refuse. A degraded compiled graph is one an adopter can never re-check (adoption
    runs no quality gate at all), so publishing it would export an unmeasured
    degradation to every pod that pulls the key. Before that ruling this exact
    compiled graph armed and shipped."""
    packages = {entry_name(h, w): ProbePackage(cosine=0.997) for h, w in ROWS}
    pipeline, _module, outcomes = arm(tmp_path, monkeypatch, declared, packages)

    assert not any(o.armed for o in outcomes), (
        "a gray-band compiled graph was published to the fleet")
    assert aot.is_armed(pipeline) is False
    rows = numerics_rows(events)
    # One row per ENTRY, each on its own single axis: the gate runs at the
    # moment that entry exists, never "after all 36" (pgw#1176 / §4.32).
    assert [p for _d, p in rows] == ["degraded", "degraded"], rows
    detail = rows[0][0]
    assert "cosine=0.997" in detail
    assert "axes=1/1" in detail


def test_a_faithful_compiled_graph_arms_AND_THE_PASS_IS_ANNOUNCED(
        tmp_path, monkeypatch, declared, events):
    """A silent pass is indistinguishable from a gate that never ran, which is
    this program's signature failure. So the pass is a hub row too, carrying
    every axis it was taken on."""
    packages = {entry_name(h, w): ProbePackage() for h, w in ROWS}
    pipeline, _module, outcomes = arm(tmp_path, monkeypatch, declared, packages)

    assert all(o.armed for o in outcomes)
    assert aot.is_armed(pipeline) is True
    assert set(aot.armed_entries(pipeline)) == {
        entry_name(h, w) for h, w in ROWS}
    rows = numerics_rows(events)
    assert [p for _d, p in rows] == ["checked", "checked"], rows
    # Per-axis readings, named — the PoolWidth discipline. One row per class,
    # each naming its own class and carrying its own key.
    for (detail, _phase), (h, w) in zip(rows, ROWS):
        assert "axes=1/1" in detail
        assert f"family={FAMILY}" in detail and "key=cg-key-v1-" in detail
        assert entry_name(h, w) in detail
        assert "cos=1.00000" in detail


# ---------------------------------------------------------------------------
# fail-closed: an unmeasurable compiled graph is NOT a passing compiled graph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package,reason", [
    (ProbePackage(raises="dlopen: undefined symbol"), "compiled_graph_forward_failed"),
    (ProbePackage(drop_output=True), "output_structure_differs"),
])
def test_a_cell_that_cannot_be_MEASURED_does_not_arm(
        tmp_path, monkeypatch, declared, events, package, reason):
    """"Nobody could ask" must never collapse into "it passed". Staying eager
    is the ordinary miss policy of every other adopt gate, so the cost of a
    probe defect is an un-armed compiled graph — never a silently degraded one."""
    packages = {entry_name(h, w): package for h, w in ROWS}
    pipeline, _module, outcomes = arm(tmp_path, monkeypatch, declared, packages)

    assert not any(o.armed for o in outcomes)
    assert aot.is_armed(pipeline) is False
    detail, phase = numerics_rows(events)[-1]
    assert phase == "unmeasurable"
    assert reason in detail
    assert "not a pass" in detail


def test_an_undeclared_family_cannot_be_probed_AND_THEREFORE_CANNOT_ARM(
        tmp_path, monkeypatch, events):
    """No export declaration => no feed => no comparison. That is exactly the
    state the whole fleet was in, and it must read as a refusal."""
    reset_export_declarations()
    packages = {entry_name(h, w): ProbePackage() for h, w in ROWS}
    _pipeline, _module, outcomes = arm(
        tmp_path, monkeypatch, declaration(), packages)

    assert not any(o.armed for o in outcomes)
    detail, phase = numerics_rows(events)[-1]
    assert phase == "unmeasurable"
    assert "no_input_contract" in detail


def test_the_report_cannot_report_a_pass_it_did_not_take():
    """The structural guard against the trap, asserted on the type itself.

    `CompiledGraphNumerics.measured` is the predicate the arm consults, and it
    must be False for every shape of partial or absent evidence — an empty
    report, a report short of its own axis count, and a report whose axes
    errored. If this ever returns True for one of these, the gate is passing
    compiled graphs nobody measured.

    pgw#1176 KEPT this row and narrowed what it guards. The all-axes-of-the-
    COMPILED GRAPH rule is gone (one unmeasurable class must not condemn 35 measured
    ones); "absent evidence is never a pass" is the part that was always the
    point, and it is what these three shapes assert.
    """
    from gen_worker.numerics_probe import (
        AxisVerdict,
        CompiledGraphNumerics,
        ProbeAxis,
    )

    thresholds = numerics_probe.numerics_ladder.DEFAULT_THRESHOLDS
    axis = ProbeAxis(entry="e", target=TARGET)
    empty = CompiledGraphNumerics(
        FAMILY, "k", thresholds, "declared", (), 2)
    assert empty.measured is False
    assert empty.comparison() is None

    errored = CompiledGraphNumerics(
        FAMILY, "k", thresholds, "declared",
        (AxisVerdict(axis=axis, reason="compiled_graph_forward_failed"),), 1)
    assert errored.measured is False
    assert errored.comparison() is None


# ---------------------------------------------------------------------------
# bisectability: one entry x one shape row x one lane
# ---------------------------------------------------------------------------


def test_the_verdict_is_bisectable_to_ONE_named_axis(
        tmp_path, monkeypatch, declared, events):
    """A verdict that cannot be split is the artifact that produced three
    wrong confident diagnoses here. pgw#1176 makes the split STRUCTURAL rather
    than a reporting convention: two shape rows, ONE of them bad, and the bad
    one refuses ALONE while the good one arms and serves. The refusal still
    names the class, because a verdict that did not would be unbisectable
    however small its subject.
    """
    good, bad = entry_name(*ROWS[0]), entry_name(*ROWS[1])
    packages = {good: ProbePackage(), bad: ProbePackage(cosine=0.90)}
    pipeline, module, outcomes = arm(tmp_path, monkeypatch, declared, packages)

    # THE point of the whole change: one bad class costs itself.
    assert outcomes[0].armed is True
    assert outcomes[1].armed is False
    assert set(aot.armed_entries(pipeline)) == {good}
    assert aot.entry_states(pipeline)[bad]["state"] == "de_armed"
    assert aot.entry_states(pipeline)[bad]["reason"] == "numerics_refused"

    rows = numerics_rows(events)
    assert [p for _d, p in rows] == ["checked", "refused"], rows
    detail = rows[-1][0]
    assert f"worst axis: entry={bad}" in detail
    assert "row=h=16,w=16" in detail
    assert "cosine=0.90" in detail
    # The healthy class's reading is on the record too, in its OWN row.
    assert f"{good}[h=8,w=8]: cos=1.00000" in detail.replace(rows[-1][0], "") \
        or f"{good}[h=8,w=8]: cos=1.00000" in rows[0][0]

    # And the healthy class still SERVES compiled, which is the property the
    # old all-or-nothing arm made impossible to have.
    module(torch.zeros(ROWS[0]), torch.tensor(1.0))
    assert packages[good].invocations == 2  # the gate's forward, then a serve


def test_an_axis_names_its_entry_row_execution_lane_and_seed_and_reproduces():
    """Every verdict carries its inputs. The seed is DERIVED from the axis, so
    two rows of one compiled graph are never fed the same latent and any reader can
    rebuild the exact feed."""
    from gen_worker.numerics_probe import ProbeAxis, axes_from_meta

    # ONE artifact, ONE axis. The seed's independence across CLASSES
    # is the property that matters and it is unchanged — it is derived from
    # the axis, so two artifacts of one declaration still never share a feed.
    (a,) = axes_from_meta(metadata(ROWS[0]))
    (b,) = axes_from_meta(metadata(ROWS[1]))
    assert [a.name, b.name] == [entry_name(*ROWS[0]), entry_name(*ROWS[1])]
    assert a.execution_lane == "w8a8" and a.target == TARGET
    assert {a.shape_row, b.shape_row} == {"h=8,w=8", "h=16,w=16"}
    assert a.seed != b.seed, "two axes share a seed; a shape-independent bug " \
                             "would read as agreement"
    # Reproducible across processes: the seed is a pure function of the axis.
    assert ProbeAxis(entry=a.entry, target=a.target, execution_lane=a.execution_lane).seed == a.seed
    assert str(a).startswith(f"entry={a.entry} target={TARGET}")


def test_the_probe_does_not_disturb_the_serving_RNG(
        tmp_path, monkeypatch, declared, events):
    """pgw#784: the tenant keeps being served throughout an arm. A probe that
    advanced the global generator would change a paying request's output in
    order to check a compiled graph."""
    torch.manual_seed(4242)
    before = torch.get_rng_state().clone()
    packages = {entry_name(h, w): ProbePackage() for h, w in ROWS}
    _pipeline, _module, outcomes = arm(tmp_path, monkeypatch, declared, packages)
    assert all(o.armed for o in outcomes)
    assert torch.equal(torch.get_rng_state(), before)


# ---------------------------------------------------------------------------
# the wiring, tested separately from the helper — this issue's signature failure
# ---------------------------------------------------------------------------


def test_a_REAL_arm_reaches_the_gate(tmp_path, monkeypatch, declared, events):
    """The tests above would all still pass if `arm_aot` never called the gate
    and something else did. This one asserts the CALL SITE: drive the real
    `provision.arm_aot` and require that the compiled graph's own forward was invoked by
    the probe — i.e. a measurement was actually taken on the arm path.

    RED when `gate_compiled_graph_numerics(pipe, cfg)` is removed from `arm_aot`:
    `invocations` stays 0 and `armed` is True.
    """
    packages = {entry_name(h, w): ProbePackage() for h, w in ROWS}
    _pipeline, _module, outcomes = arm(tmp_path, monkeypatch, declared, packages)
    assert all(o.armed for o in outcomes)
    assert all(p.invocations == 1 for p in packages.values()), (
        "a REAL arm did not run the compiled graph against its eager reference: "
        f"{ {k: p.invocations for k, p in packages.items()} }")
    assert numerics_rows(events), "a REAL arm emitted no compiled_graph_numerics row"


def test_the_gate_is_never_reached_without_a_comparison(monkeypatch):
    """The trap, asserted directly: `numerics_ladder.gate(None, ...)` returns
    None and refuses nothing. So `gate_compiled_graph_numerics` must never call it with
    a `None` comparison — every path to `gate()` goes through
    `report.measured`, and every other path refuses.
    """
    from gen_worker.models import provision
    from gen_worker import numerics_ladder

    seen: List[Any] = []
    monkeypatch.setattr(
        numerics_ladder, "gate",
        lambda comparison, **kw: seen.append(comparison))
    monkeypatch.setattr(
        numerics_probe, "probe_compiled_graph",
        lambda *a, **k: (_ for _ in ()).throw(
            numerics_probe.ProbeUnavailable("not_armed", "nothing armed")))
    assert provision.gate_compiled_graph_numerics(object(), declaration()) is False
    assert seen == [], "the ladder was consulted with no measurement to judge"
