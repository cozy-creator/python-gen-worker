"""th#1130: the image encode+upload tail runs AFTER the GPU permit is released.

th#1107 put the slot-release + finalize-permit machinery on ``gw_io.write_image``
— which no deployed endpoint calls. All 19 live image call sites use
``ctx.save_image``, so the whole image fleet serialized a ~250ms-1.1s webp
encode plus the upload behind the GPU permit. The fix cannot be a blanket
terminal release inside ``save_image`` (endpoints save mid-pipeline and in
N-image loops); the terminality signal is the HANDLER'S RETURN, where the
executor already releases the permit. ``save_image`` therefore defers its
encode+upload and the executor drains the queue after the release.

Everything below drives the REAL executor over the hub-double: a real gRPC
socket, real transport/lifecycle/executor, real PIL encodes, real GPU-permit
semantics (``ResolvedCompute(accelerator="cuda")`` — no CUDA is touched).
"""

from __future__ import annotations

import time
from io import BytesIO

import msgspec
from PIL import Image

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.stage_timing import reconciliation

from harness import deferred_endpoints as de
from harness.hub_double import hub_double, is_ready, is_result_for

MODULES = ("harness.deferred_endpoints",)
CUDA = pb.ResolvedCompute(accelerator="cuda", gpu_index=0)


def _payload() -> bytes:
    return msgspec.msgpack.encode({"prompt": "a cat"})


def _run(conn: object, rid: str, fn: str) -> None:
    # Handler methods are advertised kebab-cased (``slow_encode`` ->
    # ``slow-encode``).
    conn.send(run_job=pb.RunJob(  # type: ignore[attr-defined]
        request_id=rid, attempt=1, function_name=fn.replace("_", "-"),
        input_payload=_payload(),
        output_mode=pb.OUTPUT_MODE_INLINE, compute=CUDA))


def _image_bytes(res: pb.JobResult, field: str = "image") -> bytes:
    out = msgspec.msgpack.decode(res.inline)
    return out[field]["inline_bytes"]


def test_the_permit_is_released_before_the_deferred_encode_runs() -> None:
    """(a) The encode lands entirely inside the SLOTLESS window, and the
    th#1111 stage map still closes exactly."""
    de.reset()
    with hub_double(modules=MODULES) as (scheduler, _h):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _run(conn, "r-slow", "slow_encode")
        res = conn.wait_for(is_result_for("r-slow"), timeout=30).job_result

    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    stages = dict(res.metrics.stage_ms)
    encode_ms = stages["image_encode"]
    # The encode must be big enough that "it ran slotless" means something —
    # but NOT a wall-clock claim about the runner's CPU. `>= 100` failed on a
    # GitHub runner at 83ms (pgw debt-sweep lane, PR #397), which said nothing
    # about the property under test. Measurable, and dominating the tail, is
    # the honest form of the same requirement.
    assert encode_ms >= 10, f"the 1024^2 webp encode should be visible: {stages}"
    assert encode_ms >= 0.5 * stages["total.tail"], (
        f"the deferred encode should dominate the tail: {stages}")

    # The whole encode fits inside the post-release window: the GPU was free
    # for every millisecond of it.
    assert res.metrics.finalize_wall_ms >= encode_ms, (
        res.metrics.finalize_wall_ms, encode_ms, stages)
    # ...and the permit was NOT held for it.
    assert res.metrics.slot_held_ms <= res.metrics.runtime_ms - encode_ms + 15, (
        res.metrics.slot_held_ms, res.metrics.runtime_ms, encode_ms)

    # th#1111: the tail work is attributed to the tail, and the map reconciles.
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
    assert b_start < a_done_at, (
        f"B started {b_start - a_done_at:.3f}s after A's result — no overlap")
    overlap_ms = int((a_done_at - b_start) * 1000)
    encode_ms = dict(a.metrics.stage_ms)["image_encode"]
    assert overlap_ms >= encode_ms * 0.5, (
        f"only {overlap_ms}ms of B ran inside A's {encode_ms}ms tail")


def test_an_n_image_loop_never_releases_the_permit_early() -> None:
    """(c) The failure mode that ruled out copying th#1107's terminal release
    onto save_image: a handler that saves, then does more GPU work."""
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
    """(d) A tail that raises must fail the request, not report OK with a
    hollow asset."""
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
    """(e) Copy-on-save: the handler paints blue over the same PIL object after
    saving a red frame; the uploaded bytes must still be red."""
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
    """Read-back is CORRECT, just not overlapped: the handle materializes
    inline rather than answering None."""
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
