"""pgw#923: a boot-attached adoption produces a MEASURED wire event.

The defect, from both live stacks: `worker_activity_events` held **zero**
`compile_cache_adopt` rows — the kind th#1329/th#1352 built two partial indexes
and a p50/p95/max admin surface on — while every adoption that actually
happened rode a free-text `aot_adopt` row at `duration_ms=0`. The cause was
reachability, not correctness: the only worker-side sender of the measured
`ModelEvent{ADOPTED}` was the hub-commanded `ADOPT_COMPILE_CACHE` handler, and
no stack has ever dispatched that operation. Adoptions happen at BOOT, through
`fleet_compiled_graphs`, and that path sent no `ModelEvent` at all.

So a reader of the th#1329 surface concluded adoption never happens, a reader of
`aot_adopt` got no numbers, and the percentile endpoint aggregated a population
with no members. Nothing went red.

These tests hold an adoption open for a KNOWN interval and a warmup for a
second known interval, and assert the stored numbers. The emitter that shipped
for a year cannot pass them: it reported zero.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import boot_phases, fleet_compiled_graphs
from gen_worker.compiled_graph_adopt import AdoptOutcome, CompiledGraphAdoption
from gen_worker.executor import Executor, _InjectionResult
from gen_worker.pb import worker_scheduler_pb2 as pb

REF = "root/family-sdxl#ek1-" + "b" * 56
DIGEST = "blake3:" + "c" * 64

#: The arm is INDUCED to take this long, and the floor asserted against it is a
#: share of that induced quantity rather than a bare constant (pgw#795). This is
#: a LOWER bound on work the test itself produced: a slow runner only raises the
#: measured value, so nothing here can fail because the machine was busy.
_INDUCED_ARM_S = 0.05
_MEASURED_ARM_FLOOR_MS = int(_INDUCED_ARM_S * 1000 * 0.8)


@pytest.fixture(autouse=True)
def _reset() -> Any:
    boot_phases.reset_for_tests()
    yield
    boot_phases.reset_for_tests()


def _report(adoptions: List[CompiledGraphAdoption], proof: Dict[int, Tuple[int, int, int]],
            warm_ms: int) -> List[pb.ModelEvent]:
    """Drive the REAL executor method and read the REAL protobuf it sends."""
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor.__new__(Executor)
    ex._boot_warm_ms = warm_ms          # type: ignore[attr-defined]
    ex._send = _send                    # type: ignore[assignment]
    inj = _InjectionResult(kwargs={}, loaded={})
    inj.adoptions.extend(adoptions)
    asyncio.run(ex._report_adoptions(inj, proof))
    return [m.model_event for m in sent if m.WhichOneof("msg") == "model_event"]


def test_an_armed_adoption_reports_its_arm_time_AND_its_warm_time() -> None:
    """The acceptance: both halves are non-zero and both are the real numbers.

    `duration_ms` is what arming cost; `warmup_s` is what the compiled graph STILL costs
    once armed — the "with-compiled graph side of the trade" th#1329 exists to price and
    has never had a single sample of.
    """
    row = CompiledGraphAdoption(
        ref=REF, snapshot_digest=DIGEST, artifact_kind="aot-inductor",
        arm_ms=412, armed=True, pipeline_id=7)
    events = _report([row], {7: (5, 4, 1)}, warm_ms=1_750)

    assert len(events) == 1
    ev = events[0]
    assert ev.state == pb.MODEL_STATE_ADOPTED
    assert ev.ref == REF and ev.snapshot_digest == DIGEST
    assert ev.duration_ms == 412, "the arm reported no time"
    assert ev.warmup_s == pytest.approx(1.75), "the warmup reported no time"
    assert (ev.cache_hits, ev.cache_misses) == (4, 1)
    # The wire contract's own name for a boot-attached compiled graph. The hub stores
    # these without being taught a second spelling.
    assert ev.operation_id == "" and ev.target_incarnation_id == ""


def test_a_refused_adoption_reports_the_classified_reason_on_the_same_execution_lane() -> None:
    """th#1352's half. `adopt_failed:<reason>` is the same grammar the
    hub-commanded path uses, so ONE `kind=compile_cache_adopt` query returns
    the whole outcome distribution instead of two half-populations."""
    row = CompiledGraphAdoption(
        ref=REF, snapshot_digest=DIGEST, artifact_kind="aot-inductor",
        arm_ms=88, armed=False, reason="key_mismatch",
        detail="compiled_graph was minted for sm_90", pipeline_id=9)
    events = _report([row], {}, warm_ms=0)

    assert len(events) == 1
    ev = events[0]
    assert ev.state == pb.MODEL_STATE_FAILED
    assert ev.error == "adopt_failed:key_mismatch"
    assert ev.ref == REF and ev.snapshot_digest == DIGEST
    assert ev.duration_ms == 88


def test_an_adoption_with_no_candidate_identity_is_not_reported() -> None:
    """A measurement the hub cannot attribute to a compiled graph answers nothing, and
    the hub drops it from its side too. Not reporting it is cheaper than
    storing a row that has to be filtered out of every query."""
    row = CompiledGraphAdoption(
        ref="", snapshot_digest="", artifact_kind="", arm_ms=5, armed=True)
    assert _report([row], {}, warm_ms=10) == []


def test_each_adoption_is_its_own_sample() -> None:
    """Two attempts are two measurements — a discovered compiled graph that refused and
    a delivered compiled graph that armed are different facts about the same boot."""
    events = _report(
        [
            CompiledGraphAdoption(ref=REF, snapshot_digest=DIGEST,
                         artifact_kind="aot-inductor", arm_ms=30,
                         armed=False, reason="no_arm_for_mode", pipeline_id=1),
            CompiledGraphAdoption(ref=REF + "-b", snapshot_digest=DIGEST,
                         artifact_kind="", arm_ms=90, armed=True,
                         pipeline_id=1),
        ],
        {1: (2, 2, 0)}, warm_ms=500)
    assert [e.state for e in events] == [
        pb.MODEL_STATE_FAILED, pb.MODEL_STATE_ADOPTED]
    assert events[0].error == "adopt_failed:no_arm_for_mode"
    assert events[1].warmup_s == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# the arm itself: measured once, in the one place that arms (pgw#923/#924)
# ---------------------------------------------------------------------------


class _Pipe:
    pass


class _Cfg:
    family = "sdxl"
    lora_bucket = 0


def test_the_arm_is_measured_and_recorded_as_the_compiled_graph_arm_boot_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """`compiled_graph_arm` was a DECLARED boot phase with no producer anywhere in the
    SDK — only unit tests constructed it. It now brackets the real arm, and its
    duration is the same interval the adoption reports, so the boot ladder and
    the hub's adoption measurement cannot disagree about what an arm cost.
    """
    artifact = tmp_path / "compiled_graph.tar.gz"
    artifact.write_bytes(b"compiled_graph")

    def _slow_arm(pipe: Any, cfg: Any, cache_dir: Any, art: Any) -> AdoptOutcome:
        time.sleep(_INDUCED_ARM_S)
        return AdoptOutcome.hit("family=sdxl key=ck1-abc")

    monkeypatch.setattr(fleet_compiled_graphs.provision, "enable_compiled", _slow_arm)
    outcome = fleet_compiled_graphs.enable_compiled(
        _Pipe(), _Cfg(), artifact=artifact,
        delivered_ref=REF, delivered_digest=DIGEST)

    assert outcome.armed
    assert len(outcome.adoptions) == 1
    assert outcome.adoptions[0].arm_ms >= _MEASURED_ARM_FLOOR_MS

    arm_rows = [r for r in boot_phases.recorded_rows()
                if r.phase == boot_phases.PHASE_COMPILED_GRAPH_ARM and r.terminal]
    assert len(arm_rows) == 1, "the arm recorded no compiled_graph_arm boot phase"
    assert arm_rows[0].ref == REF
    assert arm_rows[0].artifact_key == DIGEST
    assert arm_rows[0].duration_ms >= _MEASURED_ARM_FLOOR_MS
    assert arm_rows[0].outcome == boot_phases.OUTCOME_OK


def test_a_refused_arm_records_the_reason_on_its_boot_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    artifact = tmp_path / "compiled_graph.tar.gz"
    artifact.write_bytes(b"compiled_graph")
    monkeypatch.setattr(
        fleet_compiled_graphs.provision, "enable_compiled",
        lambda *a, **k: AdoptOutcome.miss("host_isa_unsupported", "sparc64"))
    monkeypatch.setattr(fleet_compiled_graphs, "_cuda_ready", lambda: False)

    outcome = fleet_compiled_graphs.enable_compiled(
        _Pipe(), _Cfg(), artifact=artifact,
        delivered_ref=REF, delivered_digest=DIGEST)

    assert not outcome.armed
    assert [r.reason for r in outcome.adoptions] == ["host_isa_unsupported"]
    arm_rows = [r for r in boot_phases.recorded_rows()
                if r.phase == boot_phases.PHASE_COMPILED_GRAPH_ARM and r.terminal]
    assert len(arm_rows) == 1
    # A typed refusal, never a failure: the worker declined this compiled graph and goes
    # on serving eager.
    assert arm_rows[0].outcome == boot_phases.OUTCOME_REFUSED
    assert arm_rows[0].reason == "host_isa_unsupported"


def test_an_arm_with_no_candidate_records_no_compiled_graph_arm_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`provision.enable_compiled` with no artifact also covers the seeded and
    ALLOW_COLD inductor lanes. Bracketing those would put a near-zero
    `compiled_graph_arm` row on every compile-declaring boot — the same default-read-as-a-
    fact defect pgw#924 closes for `warmup`."""
    monkeypatch.setattr(
        fleet_compiled_graphs.provision, "enable_compiled",
        lambda *a, **k: AdoptOutcome.hit())
    outcome = fleet_compiled_graphs.enable_compiled(_Pipe(), _Cfg())

    assert outcome.armed
    assert outcome.adoptions == ()
    assert not [r for r in boot_phases.recorded_rows()
                if r.phase == boot_phases.PHASE_COMPILED_GRAPH_ARM]
