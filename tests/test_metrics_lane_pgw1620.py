"""pgw#1620: a completed request NAMES the lane that served it.

THE DEFECT WAS A SILENCE. anima 0.5.0 completed a real request on a rented
A4500 — real image, real card, real money — and `payload.metrics.lane` was
absent, `request_state.execution_lane` was empty, and no `applied_lane` row
existed. Nothing failed. Under the measurement law an absent metric is
UNPROVEN, so that run proved nothing about which lane executed, and every lane
claim on every pgw#1599-surface endpoint was unfalsifiable from the hub.

Root cause: `metrics.lane` is `JobResult.metrics.lane` (proto field 13), which
the v1 executor stamped from `ServedIdentity`. The v1 hardcut deleted the
executor and the v2 worker built a `JobMetrics` of two numbers.

THE CALLER HERE IS THE POD'S (pgw#1551's lesson, and tonight's): the object
under test is a REAL `ServeLoop` over a REAL projected tensorfs tree, driven
through the REAL `Worker._run_one` — the same except chain, the same
`pb.JobResult` construction, the same wire bound. Only the socket is captured.
A test that stamped a context itself and then read it back would prove the
getter and nothing else, which is exactly how this hole opened.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from gen_worker import activity  # noqa: E402
from gen_worker.models.store import ModelStore  # noqa: E402
from gen_worker.pb import worker_scheduler_pb2 as pb  # noqa: E402
from gen_worker.worker import Worker  # noqa: E402

# The pod-shaped fixture: a real ServeLoop, a real CAS, a real projected tree.
# Re-exported under the fixture names pytest resolves by, aliased on import so
# the test signatures below are parameters rather than redefinitions.
from test_pod_serve_loop_streams import LANE, REF, pod_serve_loop  # noqa: E402
from test_pod_serve_loop_streams import bound_store as _bound_store  # noqa: E402
from test_pod_serve_loop_streams import projected as _projected  # noqa: E402

projected = _projected
bound_store = _bound_store

#: `sdxl.diffusers@1+plain.bf16@1` and the fixture's `sd15.diffusers@1+plain.bf16@1`
#: both resolve to `plain.bf16@1`, and `lane_ladder.rule_body` maps that RULE to
#: ONE ranked body (pgw#1621: keyed on the rule, not on the dtype spelling —
#: `cozy.fp8-storage@1` and `cozy.fp8-rowwise@1` share a dtype and do not share
#: a body).
#: This is the hub's own vocabulary (proto 13: "fp8-w8a8-dynamic+compiled"),
#: which is what old-surface endpoints report today — so old and new surfaces
#: stay comparable in the measurement relation instead of forking by spelling.
EXPECTED = "bf16-w16a16+eager"


class _PodWorker:
    """`Worker._run_one` over the REAL pod serve loop, with the socket captured.

    Everything that produces the lane is production code: the ladder resolves
    it inside the instance build, `ServeLoop.invoke` stamps it on the request
    context, and `Worker._served_lane` reads it at the terminal.
    """

    def __init__(self, loop: Any) -> None:
        self.sent: List[pb.WorkerMessage] = []
        w = object.__new__(Worker)
        w._jobs = {}
        w._canceled = set()
        w._dispatch = None
        w.draining = False
        w.lanes = frozenset({LANE})
        w.file_base_url = ""
        w.serve = loop  # type: ignore[assignment]
        w.loaded = loop.loaded  # type: ignore[assignment]
        w.adoption = None  # type: ignore[attr-defined]

        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        w._send = _send  # type: ignore[method-assign]
        self.w = w

    def run(self, job_id: str = "lane-1") -> pb.JobResult:
        run = pb.RunJob(request_id=job_id, attempt=1, function_name="probe")
        run.models.add(slot="model", ref=REF)
        asyncio.run(self.w._run_one(run, (job_id, 1)))
        results = [m.job_result for m in self.sent if m.HasField("job_result")]
        assert len(results) == 1, self.sent
        return results[0]


@pytest.fixture()
def emitted(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, str]]:
    """Every typed activity event this process emits, captured at the seam."""
    rows: List[Dict[str, str]] = []
    real = activity.emit_event

    def _capture(kind: str, detail: str, phase: str = "", *args: Any,
                 **kwargs: Any) -> None:
        rows.append({"kind": kind, "detail": detail, "phase": phase,
                     "family": str(kwargs.get("family", ""))})
        real(kind, detail, phase, *args, **kwargs)

    monkeypatch.setattr(activity, "emit_event", _capture)
    return rows


def test_a_completed_request_carries_the_lane_that_served_it(
    projected: Dict[str, Any], bound_store: ModelStore, tmp_path: Path,
) -> None:
    """RED before pgw#1620: `metrics.lane` is the empty string on a COMPLETED
    request from an endpoint that declares a lane — which the hub omits from
    the payload entirely, and which every dashboard reads exactly the way it
    reads 'not measured yet'."""
    result = _PodWorker(pod_serve_loop(projected, tmp_path)).run()

    assert result.status == pb.JOB_STATUS_OK, result.safe_message
    assert result.metrics.lane == EXPECTED, (
        f"a completed request on a lanes-declaring endpoint reported "
        f"lane={result.metrics.lane!r}; absent means UNPROVEN, so this run "
        f"would prove nothing about which lane executed"
    )


def test_the_regime_is_MEASURED_not_assumed(
    projected: Dict[str, Any], bound_store: ModelStore, tmp_path: Path,
) -> None:
    """`+eager` here is the dispatch counter's answer, not a default: this
    fixture adopts nothing, so `armed_modules == 0` and the pod says eager
    because it counted, which is the same instrument that caught twelve
    'compiled' images being served entirely eager (pgw#1491)."""
    result = _PodWorker(pod_serve_loop(projected, tmp_path)).run()
    assert result.metrics.lane.endswith("+eager")
    body, _, regime = result.metrics.lane.partition("+")
    # `reduce.go` groups the measurement relation on `split_part(lane,'+',1)`,
    # so the body half must stay a member of the fleet's ranked table.
    from gen_worker.models.execution_lanes import known_execution_lane_bodies

    assert body in known_execution_lane_bodies(), body
    assert regime in ("eager", "compiled")


def test_the_ladders_confession_leaves_the_pod(
    projected: Dict[str, Any], bound_store: ModelStore, tmp_path: Path,
    emitted: List[Dict[str, str]],
) -> None:
    """The other half of the filing. `resolved.confession()` — the audit line
    pgw#1606 wrote so a ladder that reports only its winner cannot exist — was
    a `logger.info` on a RunPod pod, which has no logs API: measured ZERO times
    in the hub log and in zero `worker_activity_events` rows."""
    _PodWorker(pod_serve_loop(projected, tmp_path)).run()

    lanes = [row for row in emitted if row["kind"] == activity.KIND_APPLIED_LANE]
    assert lanes, (
        "no `applied_lane` event: the platform resolved a lane and forwarded "
        "it nowhere that survives the pod"
    )
    row = lanes[0]
    assert row["phase"] == "bf16-w16a16", row
    assert row["family"] == LANE, row
    # The confession names what it did NOT do, or it is not an audit line.
    assert "LANE=" in row["detail"] and "rejected=" in row["detail"], row
    assert "contract=" in row["detail"], row


def test_a_lane_that_was_never_resolved_is_reported_ABSENT(tmp_path: Path) -> None:
    """Absence must stay possible and stay honest: a weightless entrypoint
    resolves no lane, and inventing one would be the worse failure. `""` is
    what the hub stores as omitted rather than as an empty claim."""
    w = object.__new__(Worker)
    w._jobs = {}
    w._dispatch = None
    assert w._served_lane([], solo=True) == ""
    ctx = types.SimpleNamespace(_resolved_lane=None)
    assert w._served_lane([ctx], solo=True) == ""
