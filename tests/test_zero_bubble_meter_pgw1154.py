"""pgw#1154: the zero-bubble program's two meters, on the real serve path.

Paul's bar is *"the GPU should remain hot at all times ... no wasted time in
between requests."* Two things have to be true before that bar can even be
scored, and neither was:

1. **The stage window must exist on the endpoints that matter.** Step marks
   wired from ``diffusers_step_callback`` only leave every endpoint driving its
   own step loop and reporting it with ``ctx.progress(..., step=, total=)`` —
   minimax-h3, ltx-video's stage 1, anima, hidream-o1-image, i.e. the whole
   DiffSynth half of the fleet and the entire video lane — with no denoise
   window at all: 98.5% of an H3 handler lands in ``resid.unattributed``, with
   ``class.gpu_busy`` 0 and no ``total.prep`` / ``total.tail`` key whatsoever.
   The number the whole pipelining program is sized against is structurally
   absent exactly where the round-trip gap is.

2. **The gap BETWEEN requests must be measured, not inferred.** Nothing in the
   platform reported it. ``gpu_permit_wait`` measures a request waiting for the
   card; nothing measured the card waiting for a request.

Both assertions here come off the wire in ``JobResult.metrics.stage_ms``.
"""

from __future__ import annotations

import time

import msgspec

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.stage_timing import (
    PRE_HANDLER_STAGES,
    StageTimer,
    reconciliation,
    stage_ms_for_metrics,
)

from harness.hub_double import hub_double, is_ready, is_result_for
from harness.stage_endpoints import DECODE_S, STEP_S, STEPS, TEXT_ENCODE_S
from harness.upload_sink import DedupUploadSink, serve_upload_sink

ORG = "00000000-0000-0000-0000-000000000001"


def _payload() -> bytes:
    return msgspec.msgpack.encode({"prompt": "a cat"})


def _run(conn, request_id: str, function_name: str):
    conn.send(run_job=pb.RunJob(
        request_id=request_id, attempt=1, function_name=function_name,
        input_payload=_payload(), media_bytes=pb.MEDIA_BYTES_INLINE,
        org=ORG, capability_token="cap-token"))
    return conn.wait_for(is_result_for(request_id)).job_result


def test_a_progress_only_endpoint_gets_a_denoise_window_with_no_endpoint_change() -> None:
    """The Defect-A fix. ``ProgressOnly`` brackets NOTHING and never touches
    ``diffusers_step_callback``; it only calls ``ctx.progress(step=, total=)``,
    exactly as the DiffSynth families do. Before pgw#1154 this produced no
    ``total.prep``, no ``total.tail``, no ``denoise`` and ``class.gpu_busy``
    == 0."""
    httpd, base_url = serve_upload_sink()
    try:
        with hub_double(modules=("harness.stage_endpoints",),
                        file_base_url=base_url) as (scheduler, _h):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            res = _run(conn, "r-progress", "progress-generate")
    finally:
        httpd.shutdown()
        DedupUploadSink.requests_seen = []

    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    stages = dict(res.metrics.stage_ms)

    # The denoise window exists and is derived from the progress marks.
    assert stages["denoise"] >= int(STEPS * STEP_S * 1000) - 15, stages
    assert stages.get("flag.denoise_estimated") == 1, stages
    step_mean = stages["denoise.step_mean"]
    assert abs(step_mean - int(STEP_S * 1000)) <= 25, step_mean

    # And therefore so do the two numbers pipelining is sized against.
    assert stages["total.prep"] >= int(TEXT_ENCODE_S * 1000) - 5, stages
    assert stages["total.tail"] >= int(DECODE_S * 1000) - 5, stages
    assert stages["class.gpu_busy"] >= int(STEPS * STEP_S * 1000) - 15, stages

    # The instrument still closes: a new mark producer must not double-count.
    attributed, total = reconciliation(stages)
    assert total == res.metrics.runtime_ms
    assert abs(attributed - total) <= 5, (attributed, total, stages)


def test_the_shared_callback_and_ctx_progress_do_not_double_mark() -> None:
    """``diffusers_step_callback`` marks the step AND then calls
    ``ctx.progress``, which now marks too. A duplicated index would inflate the
    mark count and halve the derived per-step mean — the instrument would lie
    about the one number the JIT gate and every pipelining estimate read."""
    httpd, base_url = serve_upload_sink()
    try:
        with hub_double(modules=("harness.stage_endpoints",),
                        file_base_url=base_url) as (scheduler, _h):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            res = _run(conn, "r-dedup", "staged-generate")
    finally:
        httpd.shutdown()
        DedupUploadSink.requests_seen = []

    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    stages = dict(res.metrics.stage_ms)
    step_mean = stages["denoise.step_mean"]
    # Halved-by-duplication would land at ~STEP_S/2; the band excludes it.
    assert abs(step_mean - int(STEP_S * 1000)) <= 25, (step_mean, stages)
    assert stages["denoise"] >= int(STEPS * STEP_S * 1000) - 15, stages


def test_step_marks_are_idempotent_per_index() -> None:
    timer = StageTimer()
    timer.handler_open()
    for i in range(1, 5):
        timer.mark_step("denoise", i)
        timer.mark_step("denoise", i)  # the second producer
        time.sleep(0.02)
    timer.handler_close()
    out = timer.snapshot()
    # 4 distinct marks span 3 gaps of ~20 ms -> mean ~20. Marking each index
    # twice makes it 8 marks over the same 60 ms wall -> mean ~8.6, which this
    # band excludes. A sleep is a floor, so the ceiling carries the slack.
    assert out["denoise.step_mean"] >= 14, out
    assert out["denoise.step_mean"] <= 45, out


def test_gpu_idle_before_measures_the_gap_between_two_permit_holders() -> None:
    """The bubble meter itself. The permit path is unreachable from a CPU-only
    harness (th#1111 hit the same wall and covered ``gpu_permit_wait`` the same
    way), so this drives ``_PermitLedger`` — the object that computes the
    number — through the exact take/drop sequence two back-to-back requests
    produce on one card.

    The first holder must report NOTHING: an unmeasured gap and a zero gap are
    different answers, and a worker's first request would otherwise report its
    entire boot as a bubble.
    """
    import asyncio

    from gen_worker.executor import _PermitLedger

    sem = asyncio.Semaphore(1)
    ledger = _PermitLedger(1)

    token = ledger.take(sem, "request A")
    assert ledger.consume_idle(sem) is None, "first holder invented a bubble"

    ledger.drop(sem, token)
    gap_s = 0.25
    time.sleep(gap_s)
    ledger.take(sem, "request B")

    idle = ledger.consume_idle(sem)
    assert idle is not None
    # A sleep is a floor at any box load: assert the floor and a ceiling loose
    # enough to survive contention, never a tight band.
    assert idle >= gap_s - 0.02, idle
    assert idle <= gap_s + 5.0, idle
    # Read-once: the span belongs to the request that closed it, not to every
    # later reader.
    assert ledger.consume_idle(sem) is None


def test_no_bubble_is_recorded_while_any_holder_remains() -> None:
    """A multi-permit group only idles when the card is wholly unheld. One of
    two holders leaving is not a bubble, and recording it as one would report
    idle time on a card that never stopped computing."""
    import asyncio

    from gen_worker.executor import _PermitLedger

    sem = asyncio.Semaphore(2)
    ledger = _PermitLedger(2)
    a = ledger.take(sem, "request A")
    ledger.take(sem, "request B")

    ledger.drop(sem, a)
    time.sleep(0.05)
    ledger.take(sem, "request C")
    assert ledger.consume_idle(sem) is None, "counted idle while B still held"


def test_gpu_idle_before_rides_the_stage_map_as_a_pre_handler_stage() -> None:
    """It is the gap BEFORE this request — reported on it, charged to no
    request's ``runtime_ms``, and excluded from the reconciliation on BOTH
    sides of the wire (see runtimestore.preHandlerStageKeys)."""
    assert "gpu_idle_before" in PRE_HANDLER_STAGES

    timer = StageTimer()
    timer.record_pre("gpu_idle_before", 12.5)
    timer.handler_open()
    with timer.stage("denoise"):
        time.sleep(0.02)
    timer.handler_close()

    out = stage_ms_for_metrics(timer, runtime_ms=timer.snapshot()["total.handler"])
    assert out["gpu_idle_before"] == 12500
    attributed, total = reconciliation(out)
    assert total < 12500, out
    assert abs(attributed - total) <= 2, out



def test_a_zero_bubble_still_reports_the_key() -> None:
    """The target state must be DISTINGUISHABLE from an unmetered worker.
    ``record_pre`` drops non-positive values, so a perfect handoff would
    otherwise emit no key at all and read exactly like a pre-pgw#1154 worker —
    the one reading the whole program is trying to achieve, rendered as
    'no data'."""
    timer = StageTimer()
    timer.record_pre("gpu_idle_before", 1e-9)
    timer.handler_open()
    with timer.stage("denoise"):
        time.sleep(0.01)
    timer.handler_close()

    out = stage_ms_for_metrics(timer, runtime_ms=timer.snapshot()["total.handler"])
    assert "gpu_idle_before" in out, out
    assert out["gpu_idle_before"] == 0, out
