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
        media_bytes=pb.MEDIA_BYTES_INLINE, compute=CUDA))


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
