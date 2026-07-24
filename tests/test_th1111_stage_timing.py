"""th#1111: `runtime_ms` splits into real stages, and the map reconciles.

The integration test drives a REAL worker (hub-double: real gRPC socket, real
transport/lifecycle/executor/registry) with an image-shaped handler that has
the structure a diffusion endpoint has — text encode, a stepped denoise loop
on the SHARED ``diffusers_step_callback``, an un-bracketed VAE-decode-shaped
gap, then the framework's own ``write_image`` encode. Everything asserted here
comes off the wire in ``JobResult.metrics.stage_ms``.

The unit tests cover what no CPU-only path can reach: the GPU-permit wait
(pre-handler), exclusive nesting, and the overlap case.
"""

from __future__ import annotations

import time

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.stage_timing import StageTimer, reconciliation, stage_ms_for_metrics

from harness.hub_double import hub_double, is_ready, is_result_for
from harness.stage_endpoints import DECODE_S, STEP_S, STEPS, TEXT_ENCODE_S

import msgspec


def _payload() -> bytes:
    return msgspec.msgpack.encode({"prompt": "a cat"})


def test_stage_map_reconciles_with_runtime_ms_on_the_real_serve_path() -> None:
    with hub_double(modules=("harness.stage_endpoints",)) as (scheduler, _h):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-stage", attempt=1, function_name="staged-generate",
            input_payload=_payload(), output_mode=pb.OUTPUT_MODE_INLINE))
        res = conn.wait_for(is_result_for("r-stage")).job_result

    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    stages = dict(res.metrics.stage_ms)
    assert stages, "stage_ms is empty — the instrument did not run"

    # (a) THE reconciliation invariant: every millisecond of runtime_ms is
    # either attributed to a stage or explicitly reported as unattributed.
    attributed, total = reconciliation(stages)
    assert total == res.metrics.runtime_ms
    assert abs(attributed - total) <= 5, (attributed, total, stages)

    # (b) the stages themselves, against known handler structure.
    assert stages["text_encode"] >= int(TEXT_ENCODE_S * 1000) - 5
    assert stages["denoise"] >= int(STEPS * STEP_S * 1000) - 15
    assert stages["image_encode"] > 0
    step_mean = stages["denoise.step_mean"]
    assert abs(step_mean - int(STEP_S * 1000)) <= 25, step_mean

    # (c) prep / tail — the two numbers pipelining is sized against.
    assert stages["total.prep"] >= int(TEXT_ENCODE_S * 1000) - 5
    assert stages["total.tail"] >= int(DECODE_S * 1000) - 5
    # The un-bracketed decode gap must surface as tail residual, not vanish.
    assert stages["resid.tail"] >= int(DECODE_S * 1000) - 15

    # (d) classification: denoise is device-busy, the encode tail is not.
    assert stages["class.gpu_busy"] >= int(STEPS * STEP_S * 1000) - 15
    assert stages["class.gpu_idle"] >= stages["image_encode"]
    # Denoise was derived from step marks, not an explicit bracket.
    assert stages.get("flag.denoise_estimated") == 1


def test_permit_wait_is_reported_and_excluded_from_the_handler_window() -> None:
    """The GPU-permit wait was in NO metric (audit FINDING 0). It is reported,
    and it never inflates the runtime reconciliation."""
    timer = StageTimer()
    timer.record_pre("gpu_permit_wait", 0.250)
    timer.handler_open()
    with timer.stage("denoise"):
        time.sleep(0.02)
    timer.handler_close()

    out = stage_ms_for_metrics(timer, runtime_ms=out_runtime(timer))
    assert out["gpu_permit_wait"] == 250
    attributed, total = reconciliation(out)
    assert total < 250  # the wait is outside the window
    assert abs(attributed - total) <= 2


def out_runtime(timer: StageTimer) -> int:
    return timer.snapshot()["total.handler"]


def test_nested_stages_are_charged_exclusively() -> None:
    timer = StageTimer()
    timer.handler_open()
    with timer.stage("upload"):
        time.sleep(0.02)
        with timer.stage("credential_stamp"):
            time.sleep(0.03)
    timer.handler_close()
    out = timer.snapshot()

    assert 15 <= out["upload"] <= 35, out
    assert out["credential_stamp"] >= 25
    attributed, total = reconciliation(out)
    assert abs(attributed - total) <= 2


def test_slot_prologue_closes_the_gap_to_runtime_ms() -> None:
    """runtime_ms starts a hair before the handler window (compile fence
    check, ref pins, adapter activation). That prologue is named, not smeared,
    so the map still sums to runtime_ms exactly."""
    timer = StageTimer()
    timer.handler_open()
    with timer.stage("denoise"):
        time.sleep(0.02)
    timer.handler_close()
    handler = timer.snapshot()["total.handler"]

    out = stage_ms_for_metrics(timer, runtime_ms=handler + 40)
    assert out["slot_prologue"] == 40
    attributed, total = reconciliation(out)
    assert total == handler + 40
    assert abs(attributed - total) <= 2


def test_concurrent_stages_report_overlap_instead_of_lying() -> None:
    import threading

    timer = StageTimer()
    timer.handler_open()

    def _upload() -> None:
        with timer.stage("upload"):
            time.sleep(0.05)

    threads = [threading.Thread(target=_upload) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    timer.handler_close()
    out = timer.snapshot()

    # Three 50ms uploads inside a ~50ms window: the sum exceeds wall clock,
    # and that is stated rather than clamped away.
    assert out["upload"] >= 140
    assert out.get("resid.overlap", 0) > 0
    assert out.get("resid.unattributed", 0) == 0
