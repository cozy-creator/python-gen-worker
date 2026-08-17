"""Serving telemetry: the stage map, the boot ladder, and the deferred tail.

One module per domain; each section keeps its incident id. Full incident
narratives live in the tracker, not here.
"""

from __future__ import annotations

import asyncio
import time
from io import BytesIO
from typing import Any, List, Optional

import msgspec
import pytest
from harness import deferred_endpoints as de
from harness.hub_double import hub_double, is_ready, is_result_for
from harness.shape_endpoints_pgw789 import ShapedIn
from harness.stage_endpoints import DECODE_S, STEP_S, STEPS, TEXT_ENCODE_S
from harness.upload_sink import DedupUploadSink, serve_upload_sink
from PIL import Image

from gen_worker import boot_phases, serving_mode
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.stage_timing import StageTimer, reconciliation, stage_ms_for_metrics

# ==========================================================================
# ---- pgw#789: Benchmark telemetry — the connections that make the hub-side records usable. ----
# ==========================================================================

@pytest.fixture(autouse=True)
def _reset() -> Any:
    boot_phases.reset_for_tests()
    serving_mode.detect_sm.cache_clear()
    yield
    boot_phases.reset_for_tests()
    serving_mode.detect_sm.cache_clear()


def test_job_metrics_carry_the_serving_dimensions_on_the_wire() -> None:
    """pgw#789: The dimensions reach the HUB, over the real stream."""
    with hub_double(modules=("harness.shape_endpoints_pgw789",)) as (scheduler, _h):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-dims", attempt=1, function_name="render",
            input_payload=msgspec.msgpack.encode(
                ShapedIn(prompt="x", num_inference_steps=28, width=1024)),
        ))
        res = conn.wait_for(is_result_for("r-dims")).job_result

    assert res.status == pb.JOB_STATUS_OK
    m = res.metrics
    # This endpoint declares no compile cell, so the honest answer is eager —
    # and it must be the WORD "eager", not an empty string. "" is
    # indistinguishable from "a worker too old to report", which is exactly the
    # ambiguity that made the aggregate unusable.
    assert m.serving_mode == serving_mode.MODE_EAGER
    assert m.served_cell_ref == ""
    assert m.served_eager_fallback is False
    # And neither may the REASON be an empty string. "" here was the
    # same ambiguity one level down — it could not tell "this release declares
    # no compile target, eager is the contract" from "the mint is still
    # running" from "the mint was declined for cause". `served_eager_fallback`
    # stays False because nothing fell back: there was nothing to fall back
    # FROM, which is precisely what this token says.
    assert m.fallback_reason == serving_mode.POSTURE_NO_COMPILE_DECLARED
    # The executed shape, defaults applied: steps+width came from the payload,
    # height from the struct default — because the DEFAULT is what executed.
    assert m.steps == 28
    assert m.width == 1024
    assert m.height == 768


def test_a_guard_missed_request_reports_the_fallback_not_the_tier() -> None:
    """pgw#789: A compiled lane that served ONE request eager must not be counted as a compiled latency sample."""
    served = serving_mode.resolve(
        active_compile_ref="root/family-sdxl#ck1",
        guard_missed=True,
        verdict=serving_mode.FALLBACK_VOLATILE,
        sm="sm_89",
    )
    assert served.served_eager_fallback is True
    assert served.fallback_reason == serving_mode.FALLBACK_VOLATILE
    assert served.sm == "89"


def test_router_verdict_sets_are_keyed_by_signature_not_target() -> None:
    """pgw#789: `hot_swap.Router.healing`/`.volatile` hold input SIGNATURES."""
    class _Router:
        volatile = {"sig-1024x768x28"}
        healing: set = set()

    assert serving_mode.fallback_of(_Router(), "sig-1024x768x28") == \
        serving_mode.FALLBACK_VOLATILE
    assert serving_mode.fallback_of(_Router(), "unet") == ""


def test_shape_reads_a_mapping_of_defaults_too() -> None:
    """pgw#789: The executor's fallback defaults are `_effective_config`'s dict, not an attribute holder; `shape..."""
    class _Payload:
        num_inference_steps = 40

    assert serving_mode.shape_of(_Payload(), {"width": 1280, "height": 720}) == \
        (40, 1280, 720)


def _row(rows: List[pb.BootPhase], phase: str) -> Optional[pb.BootPhase]:
    return next((r for r in rows if r.phase == phase and r.terminal), None)


def test_a_weights_fetch_span_attributes_bytes_and_their_source() -> None:
    """pgw#789: The ~230s question is "was it network or a warm volume"."""
    with boot_phases.span(boot_phases.PHASE_WEIGHTS_FETCH, ref="repo/sdxl") as f:
        f.bytes_moved(4_720_000_000, boot_phases.SOURCE_R2)
        f.note("ref=repo/sdxl net_bytes=4720000000 manifest_bytes=4720000000")
    row = _row(boot_phases.recorded_rows(), boot_phases.PHASE_WEIGHTS_FETCH)
    assert row is not None
    assert row.bytes == 4_720_000_000
    assert row.source == boot_phases.SOURCE_R2
    assert boot_phases.phase_class(row.phase) == boot_phases.CLASS_FETCH


def test_in_boot_closes_at_the_servable_milestone() -> None:
    """pgw#789: The gate that keeps steady-state materializations OUT of the boot ladder."""
    assert boot_phases.in_boot() is True
    # The close now has a PRECONDITION — `hello`. A worker the hub
    # cannot reach is not servable, and closing the boot before the stream
    # existed is what suppressed every span on 0.78.0.
    boot_phases.mark_once(boot_phases.PHASE_HELLO, since_process_start=True)
    boot_phases.mark(
        boot_phases.PHASE_FIRST_REQUEST_SERVABLE, since_process_start=True)
    assert boot_phases.in_boot() is False
    assert boot_phases.servable_ms() is not None


def test_the_boot_ladder_reconciles_with_the_new_phases() -> None:
    """th#1111's rule at boot scale: measured + residual == the boot window, and the cumulative milestone is the..."""
    boot_phases.mark_once(boot_phases.PHASE_HELLO, since_process_start=True)
    with boot_phases.span(boot_phases.PHASE_WEIGHTS_FETCH, ref="repo/sdxl") as f:
        f.bytes_moved(1_000, boot_phases.SOURCE_VOLUME)
    with boot_phases.span(boot_phases.PHASE_PIPELINE_LOAD, function="render"):
        pass
    boot_phases.mark(
        boot_phases.PHASE_FIRST_REQUEST_SERVABLE, since_process_start=True)
    rec = boot_phases.reconciliation()
    assert rec["measured_ms"] + rec["residual_ms"] == rec["total_ms"]
    # Both new phases are CLASSIFIED, so "this release's boots are
    # network-bound" is a query rather than a hunch.
    assert "class.fetch" in rec
    assert "class.load" in rec


def test_servable_is_deferred_until_awaited_functions_are_set_up() -> None:
    """pgw#789: The correctness bug behind every 4.8s "cold boot" recorded on chaos."""
    from gen_worker.lifecycle import Lifecycle

    class _Store:
        def __init__(self) -> None:
            self.present: set = set()

        def local_path(self, ref: str) -> Optional[str]:
            return "/cas/x" if ref in self.present else None

    class _Executor:
        def __init__(self, store: _Store) -> None:
            self.store = store
            self.specs: dict = {}
            self.unavailable: dict = {}

        async def ensure_setup(self, spec: Any) -> None:
            return None

    store = _Store()
    life = object.__new__(Lifecycle)
    life.executor = _Executor(store)  # type: ignore[attr-defined,assignment]
    life.draining = False  # type: ignore[attr-defined]

    async def _noop(*_a: Any, **_kw: Any) -> None:
        return None

    life.maybe_send_state_delta = _noop  # type: ignore[assignment]

    async def _drive() -> None:
        watch = asyncio.create_task(
            life._setup_awaiting_functions(
                {"render": ["repo/sdxl"]}))  # type: ignore[list-item]
        # The ref has not landed: the worker is READY but unservable, and the
        # boot must NOT have a closing milestone yet.
        for _ in range(50):
            await asyncio.sleep(0)
            if not watch.done():
                break
        assert boot_phases.servable_ms() is None
        store.present.add("repo/sdxl")
        await watch

    asyncio.run(_drive())
    # Snapshot delivered, setup finished — and the boot is STILL open, because
    # `maybe_send_state_delta` is stubbed here and nothing has advertised a
    # function. That is the corrected contract, not a regression: the close
    # follows the advertisement, never the setup call that enabled it.
    assert boot_phases.servable_ms() is None
    assert boot_phases.in_boot() is True
    assert not [
        r for r in boot_phases.recorded_rows()
        if r.phase == boot_phases.PHASE_FIRST_REQUEST_SERVABLE
    ], "_setup_awaiting_functions is no longer an owner of the boot close"


def test_a_draining_worker_never_claims_a_boot_number() -> None:
    """pgw#789: It never became servable."""
    from gen_worker.lifecycle import Lifecycle

    life = object.__new__(Lifecycle)
    life.draining = True  # type: ignore[attr-defined]

    class _Executor:
        specs: dict = {}
        unavailable: dict = {}

        class store:  # noqa: N801
            @staticmethod
            def local_path(_ref: str) -> None:
                return None

    life.executor = _Executor()  # type: ignore[attr-defined,assignment]
    asyncio.run(life._setup_awaiting_functions(
        {"render": ["repo/sdxl"]}))  # type: ignore[list-item]
    assert boot_phases.servable_ms() is None


# ==========================================================================
# ---- th#1111: th#1111: `runtime_ms` splits into real stages, and the map reconciles. ----
# ==========================================================================

def _payload() -> bytes:
    return msgspec.msgpack.encode({"prompt": "a cat"})


def test_stage_map_reconciles_with_runtime_ms_on_the_real_serve_path() -> None:
    # This test used to need no file API at all. Under
    # MEDIA_BYTES_INLINE the image AND the (~200 KiB) result envelope both took
    # the inline shortcut, and the envelope's shortcut was the defect — it
    # returned a ref for bytes that were never uploaded. With the envelope now
    # always really stored, the stage-timing path needs a real upload sink like
    # any other large-result dispatch. The test was green because of the bug.
    httpd, base_url = serve_upload_sink()
    try:
        with hub_double(modules=("harness.stage_endpoints",),
                        file_base_url=base_url) as (scheduler, _h):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id="r-stage", attempt=1, function_name="staged-generate",
                input_payload=_payload(), media_bytes=pb.MEDIA_BYTES_INLINE,
                org="00000000-0000-0000-0000-000000000001",
                capability_token="cap-token"))
            res = conn.wait_for(is_result_for("r-stage")).job_result
    finally:
        httpd.shutdown()
        DedupUploadSink.requests_seen = []

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
    """th#1111: The GPU-permit wait was in NO metric (audit FINDING 0)."""
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
    parent_ms, child_ms = 20, 30
    timer = StageTimer()
    timer.handler_open()
    with timer.stage("upload"):
        time.sleep(parent_ms / 1000)
        with timer.stage("credential_stamp"):
            time.sleep(child_ms / 1000)
    timer.handler_close()
    out = timer.snapshot()

    # bounds are DERIVED from the sleeps, never a literal band. A sleep
    # is a floor at any box load, so the child's floor is a real ceiling on an
    # exclusively-charged parent; the old `15 <= upload <= 35` was red at load
    # 90 and green at load 17 while nothing about the code had changed.
    assert out["credential_stamp"] >= child_ms, out
    assert out["upload"] >= parent_ms, out
    # THE claim: the child's time is not also charged to the parent. Inclusive
    # charging would push `upload` past this ceiling and oversubscribe the
    # handler window (surfacing as resid.overlap) — at any load.
    assert out["upload"] <= out["total.handler"] - child_ms, out
    assert abs(out["upload"] + out["credential_stamp"] - out["total.handler"]) <= 2, out
    assert "resid.overlap" not in out, out
    attributed, total = reconciliation(out)
    assert abs(attributed - total) <= 2


def test_slot_prologue_closes_the_gap_to_runtime_ms() -> None:
    """th#1111: runtime_ms starts a hair before the handler window (compile fence check, ref pins, adapter activ..."""
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


# ==========================================================================
# ---- th#1130: th#1130: the image encode+upload tail runs AFTER the GPU permit is released. ----
# ==========================================================================

MODULES = ("harness.deferred_endpoints",)


CUDA = pb.ResolvedCompute(accelerator="cuda", gpu_index=0)


def _run(conn: object, rid: str, fn: str) -> None:
    # Handler methods are advertised kebab-cased (``slow_encode`` ->
    # ``slow-encode``).
    conn.send(run_job=pb.RunJob(  # type: ignore[attr-defined]
        request_id=rid, attempt=1, function_name=fn.replace("_", "-"),
        input_payload=_payload(),
        media_bytes=pb.MEDIA_BYTES_INLINE, compute=CUDA))


def _image_bytes(res: pb.JobResult, field: str = "image") -> bytes:
    out = msgspec.msgpack.decode(res.inline)
    return out[field]["inline_bytes"]


def test_the_permit_is_released_before_the_deferred_encode_runs() -> None:
    """(a) The encode lands entirely inside the SLOTLESS window, and the th#1111 stage map still closes exactly."""
    de.reset()
    with hub_double(modules=MODULES) as (scheduler, _h):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _run(conn, "r-slow", "slow_encode")
        res = conn.wait_for(is_result_for("r-slow"), timeout=30).job_result

    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    stages = dict(res.metrics.stage_ms)
    encode_ms = stages["image_encode"]
    # This used to be a wall-clock claim about the runner's CPU, and
    # it has now failed a release THREE times under three different constants —
    # `>= 100` at 83ms (PR #397), `>= 0.5 * total.tail` at 76 vs 77, and
    # `>= 10` on the v0.78.0 publish rerun. Lowering the number each time is
    # the anti-pattern with a smaller constant: no value is safe, because the
    # quantity being asserted is the RUNNER's speed, not the code's behaviour.
    #
    # What the assertion was actually for is that the encode really happened,
    # so "it ran slotless" is not a statement about nothing. Its PRODUCT proves
    # that at any speed: a 1024^2 webp frame came back on the wire.
    encoded = _image_bytes(res)
    frame = Image.open(BytesIO(encoded))
    assert frame.size == (de.SLOW_PX, de.SLOW_PX), frame.size
    assert frame.format == "WEBP", frame.format
    assert "image_encode" in stages, stages  # ...and it is ATTRIBUTED as a stage

    # The whole encode fits inside the post-release window: the GPU was free
    # for every millisecond of it.
    #
    # pgw#1349: this reported `assert 87 >= 88` on master, and the cause was
    # NOT the runner. `finalize_wall_ms` truncated its seconds while every
    # `stage_ms` rounded, so a stage measured strictly INSIDE the finalize
    # interval could out-round its own container by one quantum. Both sides now
    # go through `stage_timing.ms_from_seconds`, and rounding is monotone — so
    # containment of the intervals (which holds by construction: `released_at`
    # is stamped before the drain and `handler_done` after it) now survives
    # quantization, and this row is exact rather than usually-true. No slop is
    # added here on purpose: a tolerance would have hidden the mixed quantizer
    # instead of removing it.
    assert res.metrics.finalize_wall_ms >= encode_ms, (
        res.metrics.finalize_wall_ms, encode_ms, stages)
    # ...and the permit was NOT held for it. The slack is the stage map's OWN
    # unattributed residual rather than a hard-coded 15ms: the residual is where
    # runner noise lands, so the tolerance grows with the noise it exists to
    # absorb instead of being a second constant waiting to be lowered.
    assert res.metrics.slot_held_ms + encode_ms <= (
        res.metrics.runtime_ms + stages["resid.unattributed"]), (
        res.metrics.slot_held_ms, res.metrics.runtime_ms, encode_ms, stages)

    # The tail work is attributed to the tail, and the map reconciles.
    assert stages["total.tail"] >= encode_ms
    attributed, total = reconciliation(stages)
    assert total == res.metrics.runtime_ms
    assert abs(attributed - total) <= 5, (attributed, total, stages)
    assert stages["class.gpu_idle"] >= encode_ms


def test_a_second_job_takes_the_gpu_while_the_first_is_still_encoding() -> None:
    """(b) The point of the whole lane: in-flight-N overlap."""
    de.reset()
    with hub_double(modules=MODULES) as (scheduler, _h):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _run(conn, "r-a", "slow_encode")
        _run(conn, "r-b", "fast_peer")
        a = conn.wait_for(is_result_for("r-a"), timeout=30).job_result
        a_done_at = time.monotonic()
        b = conn.wait_for(is_result_for("r-b"), timeout=30).job_result

    assert a.status == pb.JOB_STATUS_OK, a.safe_message
    assert b.status == pb.JOB_STATUS_OK, b.safe_message

    b_start = de.at("handler_start", "r-b")
    a_handler_end = de.at("handler_end", "r-a")
    # B waited for A's GPU phase (one slot)...
    assert b_start > a_handler_end, "B must not run inside A's GPU phase"
    # ...and started BEFORE A's tail finished: A's encode+upload and B's
    # compute were in flight at the same time.
    # The property is the OVERLAP EXISTING — B holding the GPU while
    # A is still encoding. How MUCH of A's tail B covers is a race between two
    # threads on a shared runner, so a `>= 0.5 * encode_ms` share was a claim
    # about the scheduler's mood; the overlap is reported, not asserted on.
    overlap_ms = int((a_done_at - b_start) * 1000)
    encode_ms = dict(a.metrics.stage_ms)["image_encode"]
    assert b_start < a_done_at, (
        f"B started {b_start - a_done_at:.3f}s after A's result — no overlap "
        f"(A's encode was {encode_ms}ms, overlap {overlap_ms}ms)")


def test_an_n_image_loop_never_releases_the_permit_early() -> None:
    """(c) The failure mode that ruled out copying th#1107's terminal release onto save_image: a handler that sa..."""
    de.reset()
    with hub_double(modules=MODULES) as (scheduler, _h):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _run(conn, "r-loop", "n_images")
        _run(conn, "r-peer", "fast_peer")
        loop = conn.wait_for(is_result_for("r-loop"), timeout=30).job_result
        peer = conn.wait_for(is_result_for("r-peer"), timeout=30).job_result

    assert loop.status == pb.JOB_STATUS_OK, loop.safe_message
    assert peer.status == pb.JOB_STATUS_OK, peer.safe_message
    assert len(msgspec.msgpack.decode(loop.inline)["images"]) == de.LOOP_IMAGES

    peer_start = de.at("handler_start", "r-peer")
    # Not one of the N saves handed the permit away: the peer ran only after
    # the whole handler returned.
    assert peer_start > de.at("handler_end", "r-loop"), (
        "a save released the GPU permit while the handler still had work")
    assert peer_start > de.at(f"saved_{de.LOOP_IMAGES - 1}", "r-loop")


def test_a_failing_deferred_encode_fails_the_request() -> None:
    """th#1130: (d) A tail that raises must fail the request, not report OK with a hollow asset."""
    de.reset()
    with hub_double(modules=MODULES) as (scheduler, _h):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _run(conn, "r-boom", "fails_in_tail")
        res = conn.wait_for(is_result_for("r-boom"), timeout=30).job_result

    assert res.status != pb.JOB_STATUS_OK, "a broken encode reported success"
    assert not res.inline, "no output may be shipped for a failed tail"
    assert res.safe_message, "the failure must carry a message"


def test_mutating_the_image_after_save_cannot_change_the_upload() -> None:
    """th#1130: (e) Copy-on-save: the handler paints blue over the same PIL object after saving a red frame; the..."""
    de.reset()
    with hub_double(modules=MODULES) as (scheduler, _h):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _run(conn, "r-mut", "mutates_after_save")
        res = conn.wait_for(is_result_for("r-mut"), timeout=30).job_result

    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    img = Image.open(BytesIO(_image_bytes(res))).convert("RGB")
    assert img.getpixel((5, 5)) == (255, 0, 0), (
        "the deferred encode picked up the handler's later mutation")


def test_reading_a_bytes_field_in_the_handler_forces_a_real_encode() -> None:
    """th#1130: Read-back is CORRECT, just not overlapped: the handle materializes inline rather than answering ..."""
    de.reset()
    with hub_double(modules=MODULES) as (scheduler, _h):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _run(conn, "r-read", "reads_back")
        res = conn.wait_for(is_result_for("r-read"), timeout=30).job_result

    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    out = msgspec.msgpack.decode(res.inline)
    assert out["size_bytes"] > 0, "the handler read a hollow size_bytes"
    assert out["ref"].endswith(".png")
