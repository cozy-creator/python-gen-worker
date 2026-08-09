"""Dispatch & serving scenarios (design domain 3).

The canonical walk — dispatch -> load -> serve -> result upload — plus its
lifecycle edges (cancel, deadline, reconnect, drain, backpressure) and the
refusal matrix. Everything rides the hub double: a REAL Worker over a REAL
gRPC socket, real blob host for weights, real HTTP sink for result uploads.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import time

import msgspec
import pytest

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.stage_timing import reconciliation
from gen_worker.transport import SendQueue

from harness.hub_double import is_accept_for, is_model_event, is_ready, is_result_for

from tests_v2 import catalog

ORG = "00000000-0000-0000-0000-000000000042"


def _reply(res) -> str:
    return catalog.row("echo").decode(res.inline).response


def _progress_for(rid: str):
    return (lambda m: m.WhichOneof("msg") == "job_progress"
            and m.job_progress.request_id == rid)


def _hot_residency(
    snapshot: pb.Snapshot, *, generation: int = 7, file_base_url: str = "",
) -> pb.HelloAck:
    # file_base_url must ride EVERY HelloAck: the worker adopts the latest
    # ack's value, and an empty one strands later uploads.
    return pb.HelloAck(
        protocol_version=pb.PROTOCOL_VERSION_CURRENT,
        file_base_url=file_base_url,
        desired_residency=pb.DesiredResidency(
            generation=generation,
            disk_refs=[catalog.HOT_REF],
            snapshots={catalog.HOT_REF: snapshot},
            hot=[pb.DesiredInstance(
                function_name="hot-echo",
                models=[pb.ModelBinding(slot="model", ref=catalog.HOT_REF)],
            )],
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 1 — the canonical walk: residency -> load -> serve, streaming
# progress, stage timing, typed usage metrics, inline vs blob_ref upload,
# and at-most-once execution for a retransmitted attempt.
# ---------------------------------------------------------------------------


def test_dispatch_load_serve_and_upload_walk(hub, blob_host, upload_sink) -> None:
    weights = b"v2-hot-weights"
    snapshot = blob_host.one_file_snapshot("snap-v2", "hot", weights)
    with hub(file_base_url=upload_sink.base_url) as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        ready = conn.wait_for(is_ready).state_delta
        assert "hot-echo" not in ready.available_functions  # gated pre-residency

        # Residency: the hub's desired state drives a REAL download off the
        # blob host, digest-verified, and every model event is fenced by
        # snapshot digest + residency generation.
        conn.send(hello_ack=_hot_residency(
            snapshot, file_base_url=upload_sink.base_url))
        for state in (pb.MODEL_STATE_DOWNLOADING, pb.MODEL_STATE_ON_DISK,
                      pb.MODEL_STATE_IN_RAM):
            event = conn.wait_for(is_model_event(catalog.HOT_REF, state)).model_event
            assert event.snapshot_digest == "snap-v2"
            assert event.residency_generation == 7
        conn.wait_for(
            lambda m: m.WhichOneof("msg") == "state_delta"
            and "hot-echo" in m.state_delta.available_functions
            and m.state_delta.observed_residency_generation == 7
        )

        # Plain dispatch: accept precedes result; result is typed and carries
        # the attempt; metrics exist on the completion.
        conn.send(run_job=pb.RunJob(
            request_id="r-echo", attempt=1, function_name="echo",
            input_payload=catalog.row("echo").input_bytes(text="marco")))
        conn.wait_for(is_accept_for("r-echo"))
        res = conn.wait_for(is_result_for("r-echo")).job_result
        assert res.status == pb.JOB_STATUS_OK
        assert res.attempt == 1
        assert _reply(res) == "polo"
        assert res.metrics.runtime_ms >= 0

        # Model-bound serve: the response IS the hub-delivered bytes, so a
        # passing dispatch proves download -> verify -> load -> serve.
        conn.send(run_job=pb.RunJob(
            request_id="r-hot", attempt=1, function_name="hot-echo",
            input_payload=catalog.row("hot-echo").input_bytes()))
        res = conn.wait_for(is_result_for("r-hot")).job_result
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        assert _reply(res) == weights.decode()

        # Streaming: seq-ordered progress chunks, decodable, then the result.
        conn.send(run_job=pb.RunJob(
            request_id="r-stream", attempt=1, function_name="stream3",
            input_payload=catalog.row("stream3").input_bytes()))
        conn.wait_for(is_result_for("r-stream"))
        chunks = [m.job_progress for m in conn.received
                  if _progress_for("r-stream")(m)]
        assert [c.seq for c in chunks] == [1, 2, 3]
        assert msgspec.json.decode(chunks[0].data)["response"] == "chunk-0"

        # Stage timing on a stage-shaped handler: the map is real and it
        # reconciles against runtime_ms (attributed + residual == total).
        conn.send(run_job=pb.RunJob(
            request_id="r-stage", attempt=1, function_name="staged-generate",
            input_payload=catalog.row("staged-generate").input_bytes(prompt="a cat"),
            output_mode=pb.OUTPUT_MODE_INLINE,
            # pgw#767: the ~200 KiB result envelope is now always really
            # stored, so this dispatch needs the capability token every other
            # large-result dispatch needs. It passed without one only because
            # the inline shortcut skipped the upload and handed back a ref for
            # bytes that never left the process.
            org=ORG, capability_token="cap-token"))
        res = conn.wait_for(is_result_for("r-stage")).job_result
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        stages = dict(res.metrics.stage_ms)
        assert stages, "stage_ms is empty — the instrument did not run"
        attributed, total = reconciliation(stages)
        assert total == res.metrics.runtime_ms
        assert abs(attributed - total) <= 5, (attributed, total, stages)
        assert stages["text_encode"] >= int(catalog.TEXT_ENCODE_S * 1000) - 5
        assert stages["denoise"] >= int(catalog.STEPS * catalog.STEP_S * 1000) - 15

        # Small output: inline by size alone, typed usage intact.
        conn.send(run_job=pb.RunJob(
            request_id="r-small", attempt=1, function_name="small-usage",
            input_payload=catalog.row("small-usage").input_bytes()))
        res = conn.wait_for(is_result_for("r-small")).job_result
        assert res.status == pb.JOB_STATUS_OK
        assert res.inline and not res.blob_ref
        assert (res.metrics.input_tokens, res.metrics.input_cached_tokens,
                res.metrics.output_tokens) == (12, 2, 5)

        # Large output: blob_ref via a REAL upload round trip to the sink,
        # billed usage computed BEFORE the wire-form decision so it survives.
        conn.send(run_job=pb.RunJob(
            request_id="r-large", attempt=1, function_name="large-usage",
            input_payload=catalog.row("large-usage").input_bytes(),
            org=ORG, capability_token="cap-token"))
        res = conn.wait_for(is_result_for("r-large")).job_result
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        assert res.blob_ref and not res.inline
        assert (res.metrics.input_tokens, res.metrics.input_cached_tokens,
                res.metrics.output_tokens) == (4000, 100, 9000)
        assert upload_sink.requests, "the real upload sink was never hit"
        path, body = upload_sink.requests[-1]
        assert path.startswith(f"/api/v1/media/{ORG}/uploads")
        assert body["size_bytes"] > 64 * 1024

        # Retransmitted live attempt: re-acked, never re-executed — the
        # 2s bound is an ABSENCE probe (its expiry passes, never fails).
        conn.send(run_job=pb.RunJob(
            request_id="r-echo", attempt=1, function_name="echo",
            input_payload=catalog.row("echo").input_bytes(text="marco")))
        with pytest.raises(TimeoutError):
            conn.wait_for_count(is_result_for("r-echo"), 2, timeout=2.0)
        assert conn.count(is_result_for("r-echo")) == 1


# ---------------------------------------------------------------------------
# Scenario 2 — cancel unwinds clean: mid-stream, pre-output, cancelled setup
# residue, and a deadline that frees the worker for the next job.
# ---------------------------------------------------------------------------


def test_cancel_and_deadline_unwind_without_leak(hub) -> None:
    with hub() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)

        # Mid-stream cancel: typed CANCELED, and progress STOPS — the 20-chunk
        # generator would still be yielding for 4 more seconds otherwise.
        conn.send(run_job=pb.RunJob(
            request_id="r-slow-stream", attempt=1, function_name="slow-stream",
            input_payload=catalog.row("slow-stream").input_bytes()))
        conn.wait_for_count(_progress_for("r-slow-stream"), 1)
        conn.send(cancel_job=pb.CancelJob(request_id="r-slow-stream", attempt=1))
        res = conn.wait_for(is_result_for("r-slow-stream")).job_result
        assert res.status == pb.JOB_STATUS_CANCELED
        at_cancel = conn.count(_progress_for("r-slow-stream"))
        time.sleep(0.5)  # absence probe: more chunks would arrive if leaked
        assert conn.count(_progress_for("r-slow-stream")) == at_cancel

        # Pre-output cancel of a long await: typed CANCELED, nothing emitted.
        conn.send(run_job=pb.RunJob(
            request_id="r-slow", attempt=1, function_name="slow",
            input_payload=catalog.row("slow").input_bytes()))
        conn.wait_for(is_accept_for("r-slow"))
        conn.send(cancel_job=pb.CancelJob(request_id="r-slow", attempt=1))
        res = conn.wait_for(is_result_for("r-slow")).job_result
        assert res.status == pb.JOB_STATUS_CANCELED

        # pgw#904 part (d): the wall deadline is DEAD. A wire `timeout_ms`
        # kills nothing — no result appears when it expires — and the job
        # ends only through a real terminal edge (here, a cancel). Kill and
        # condemn come from liveness + progress-staleness, never a clock.
        conn.send(run_job=pb.RunJob(
            request_id="r-no-deadline", attempt=1, function_name="slow",
            input_payload=catalog.row("slow").input_bytes(), timeout_ms=300))
        conn.wait_for(is_accept_for("r-no-deadline"))
        time.sleep(1.0)  # 3x the old bound: a deadline kill would have landed
        assert conn.count(is_result_for("r-no-deadline")) == 0
        conn.send(cancel_job=pb.CancelJob(request_id="r-no-deadline", attempt=1))
        res = conn.wait_for(is_result_for("r-no-deadline")).job_result
        assert res.status == pb.JOB_STATUS_CANCELED
        conn.send(run_job=pb.RunJob(
            request_id="r-next", attempt=1, function_name="echo",
            input_payload=catalog.row("echo").input_bytes(text="marco")))
        assert conn.wait_for(is_result_for("r-next")).job_result.status == pb.JOB_STATUS_OK


def test_cancelled_setup_leaves_no_residue() -> None:
    """gw#624 live incident: 5 cancelled load retries climbed RAM 3%->97%.
    Real executor path — cancel ensure_setup mid-setup(), retry, and prove
    attempt 1's cycle-carrying buffer is GONE before attempt 2 allocates.
    gc is disabled so only the executor's own purge can pass."""
    from gen_worker.executor import Executor
    from gen_worker.registry import extract_specs

    sent: list = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    specs = extract_specs(catalog.HoldSetup)
    ex = Executor(specs, _send)
    probe = catalog.HoldSetupProbe
    probe.arm()

    async def run() -> None:
        task = asyncio.create_task(ex.ensure_setup(specs[0]))
        await asyncio.to_thread(probe.ENTERED.wait, 10)
        task.cancel()
        probe.RELEASE.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await ex.ensure_setup(specs[0])

    gc.disable()
    try:
        asyncio.run(run())
        verdict = list(probe.alive_at_second_attempt)
    finally:
        gc.enable()
        probe.reset()

    assert verdict == [False], (
        "the cancelled attempt's partial load survived into the retry — "
        "retries stack allocations until OOM (gw#624)"
    )


# ---------------------------------------------------------------------------
# Scenario 3 — durability & backpressure: exactly-once results across a
# stream kill, drain semantics, and the durable-vs-sheddable SendQueue.
# ---------------------------------------------------------------------------


def test_reconnect_drain_and_send_queue_durability(hub) -> None:
    with hub() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)

        # Kill the stream mid-job: the reconnected Hello names the in-flight
        # attempt, and the buffered result ships EXACTLY once across the
        # whole connection history — never zero, never twice.
        conn.send(run_job=pb.RunJob(
            request_id="r-mid", attempt=2, function_name="sleepy",
            input_payload=catalog.row("sleepy").input_bytes()))
        conn.wait_for(is_accept_for("r-mid"))
        conn.kill()
        conn2 = scheduler.wait_connection(1)
        assert conn2.hello is not None
        in_flight = {(j.request_id, j.attempt) for j in conn2.hello.in_flight}
        assert ("r-mid", 2) in in_flight
        res = conn2.wait_for(is_result_for("r-mid")).job_result
        assert res.status == pb.JOB_STATUS_OK and res.attempt == 2
        time.sleep(0.3)  # absence probe for a duplicate
        assert sum(c.count(is_result_for("r-mid"))
                   for c in scheduler.connections) == 1

        # Drain: in-flight work finishes, NEW work is refused typed (never
        # silently dropped), and the worker closes the stream.
        conn2.send(run_job=pb.RunJob(
            request_id="r-last", attempt=1, function_name="sleepy",
            input_payload=catalog.row("sleepy").input_bytes()))
        conn2.wait_for(is_accept_for("r-last"))
        conn2.send(drain=pb.Drain(deadline_ms=5000))
        conn2.send(run_job=pb.RunJob(
            request_id="r-after-drain", attempt=1, function_name="echo",
            input_payload=catalog.row("echo").input_bytes(text="marco")))
        rejected = conn2.wait_for(is_result_for("r-after-drain")).job_result
        assert rejected.status == pb.JOB_STATUS_RETRYABLE
        assert "draining" in rejected.safe_message
        assert conn2.count(is_accept_for("r-after-drain")) == 0
        assert conn2.wait_for(is_result_for("r-last")).job_result.status == pb.JOB_STATUS_OK
        assert conn2.client_done.wait(15.0), "worker must close the stream after drain"


def test_send_queue_sheds_progress_never_results() -> None:
    """The transport's actual queue: bounded progress (drop-oldest), exempt
    durable results that survive reconnect until marked shipped."""

    def _p(rid: str, seq: int) -> pb.WorkerMessage:
        return pb.WorkerMessage(job_progress=pb.JobProgress(
            request_id=rid, attempt=1, seq=seq))

    def _r(rid: str) -> pb.WorkerMessage:
        return pb.WorkerMessage(job_result=pb.JobResult(
            request_id=rid, attempt=1, status=pb.JOB_STATUS_OK))

    async def _run() -> None:
        q = SendQueue(maxsize=2)
        await q.put(_p("p", 1))
        await q.put(_p("p", 2))
        await q.put(_p("p", 3))  # overflow: seq=1 dropped, no producer block
        await q.put(_r("r1"))    # results are exempt from the bound
        drained = []
        while len(q):
            drained.append(await q.get())
        seqs = [m.job_progress.seq for k, m in drained if k == "progress"]
        assert seqs == [2, 3]
        assert any(k == "result" for k, _m in drained)

        # Results survive a reconnect until shipped; progress is shed.
        q2 = SendQueue(maxsize=4)
        await q2.put(_p("p", 1))
        await q2.put(_r("r1"))
        await q2.put(_r("r2"))
        while True:
            kind, msg = await q2.get()
            if kind == "result" and msg.job_result.request_id == "r1":
                await q2.mark_result_shipped(msg)
                break
        await q2.reset_for_reconnect()
        assert q2.pending_result_keys == [("r2", 1)]
        kind, msg = await q2.get()
        assert kind == "result" and msg.job_result.request_id == "r2"
        assert len(q2) == 0

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Scenario 4 — the dispatch refusal matrix. Every gate refuses TYPED, NAMES
# its cause, and the worker keeps serving afterwards. A silent failure or a
# log-only swallow here is the defect class this suite exists to prevent.
# ---------------------------------------------------------------------------


def test_dispatch_refusal_matrix_is_typed_and_named(hub, blob_host) -> None:
    rows = [
        # (request_id, RunJob kwargs, expected status, must-name fragments)
        ("r-unknown",
         dict(function_name="nope",
              input_payload=catalog.row("echo").input_bytes(text="marco")),
         pb.JOB_STATUS_INVALID, ["nope"]),
        ("r-bad-manifest",
         dict(function_name="echo", input_payload=b"\xc1not-msgpack"),
         pb.JOB_STATUS_INVALID, []),
        ("r-bad-input",
         dict(function_name="echo",
              input_payload=catalog.row("echo").input_bytes(text="not-marco")),
         pb.JOB_STATUS_INVALID, ["not-marco"]),
        # Ref-grammar/derailed pick: an adapter overlay naming a slot the
        # function never declared.
        ("r-bad-overlay",
         dict(function_name="echo",
              input_payload=catalog.row("echo").input_bytes(text="marco"),
              models=[pb.ModelBinding(
                  slot="nope",
                  loras=[pb.LoraOverlay(ref="catalog/some-lora", weight=1.0)])]),
         pb.JOB_STATUS_INVALID, ["nope"]),
    ]
    with hub() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        for rid, kwargs, want_status, fragments in rows:
            conn.send(run_job=pb.RunJob(request_id=rid, attempt=1, **kwargs))
            res = conn.wait_for(is_result_for(rid)).job_result
            assert res.status == want_status, (
                f"{rid}: status {res.status} != {want_status} "
                f"({res.safe_message!r})")
            assert res.safe_message, f"{rid}: refusal with NO named cause"
            for fragment in fragments:
                assert fragment in res.safe_message, (
                    f"{rid}: {fragment!r} not named in {res.safe_message!r}")
        # An unknown function is refused BEFORE acceptance — no accept race.
        assert conn.count(is_accept_for("r-unknown")) == 0

        # Missing model (th#763): a dispatch for a residency-gated function is
        # never a terminal fault pinned on the caller and never a silent hang
        # — the worker emits the typed missing_snapshot decline the hub
        # re-mints from, and once residency arrives the SAME parked job
        # completes with the delivered bytes.
        conn.send(run_job=pb.RunJob(
            request_id="r-missing-model", attempt=1, function_name="hot-echo",
            input_payload=catalog.row("hot-echo").input_bytes()))
        conn.wait_for(is_accept_for("r-missing-model"))
        missing = conn.wait_for(
            lambda m: m.WhichOneof("msg") == "model_event"
            and m.model_event.ref == catalog.HOT_REF
            and m.model_event.state == pb.MODEL_STATE_FAILED
        ).model_event
        assert "missing_snapshot" in missing.error
        weights = b"healed-after-remint"
        snapshot = blob_host.one_file_snapshot("snap-remint", "healed", weights)
        conn.send(hello_ack=_hot_residency(snapshot, generation=2))
        res = conn.wait_for(is_result_for("r-missing-model")).job_result
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        assert _reply(res) == weights.decode()

        # After the whole gauntlet the worker still serves: refusals fail the
        # REQUEST, never the process.
        conn.send(run_job=pb.RunJob(
            request_id="r-still-alive", attempt=1, function_name="echo",
            input_payload=catalog.row("echo").input_bytes(text="marco")))
        res = conn.wait_for(is_result_for("r-still-alive")).job_result
        assert res.status == pb.JOB_STATUS_OK
        assert _reply(res) == "polo"


def test_fixed_slot_wrong_repo_pick_refuses_by_name(hub, blob_host) -> None:
    """gw#583: a FIXED slot dispatched a DIFFERENT repo than declared refuses,
    naming the slot and BOTH refs — serving the wrong checkpoint under a
    pinned identity would be a wrong-output bug, worse than declining."""
    wrong_repo = "catalog/some-other-repo"
    snap = blob_host.one_file_snapshot("snap-wrong", "wrong", b"irrelevant")
    with hub() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-wrong-repo", attempt=1, function_name="pinned-echo",
            input_payload=catalog.row("pinned-echo").input_bytes(),
            models=[pb.ModelBinding(slot="pipeline", ref=wrong_repo)],
            snapshots={wrong_repo: snap}))
        res = conn.wait_for(is_result_for("r-wrong-repo")).job_result
        assert res.status != pb.JOB_STATUS_OK
        assert "pipeline" in res.safe_message
        assert catalog.PINNED_DEFAULT.path in res.safe_message
        assert "some-other-repo" in res.safe_message


def test_result_upload_refusal_is_typed_not_silent(hub) -> None:
    """Bad credentials at the upload boundary (the sink 403s): the job must
    land as a TYPED retryable naming the upload, never OK-with-lost-output
    and never a log-only swallow."""
    from tests_v2.conftest import UploadSink

    sink = UploadSink(status=403)
    try:
        with hub(file_base_url=sink.base_url) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id="r-upload-403", attempt=1, function_name="large-usage",
                input_payload=catalog.row("large-usage").input_bytes(),
                org=ORG, capability_token="bad-token"))
            res = conn.wait_for(is_result_for("r-upload-403")).job_result
            assert res.status == pb.JOB_STATUS_RETRYABLE, res.safe_message
            assert "upload" in res.safe_message
            assert not res.blob_ref and not res.inline
            assert sink.requests, "the refusal came without dialing the sink"
    finally:
        sink.shutdown()


def test_unknown_function_refusal_names_the_function(hub) -> None:
    """A refusal that does not name its cause is a log-only swallow on the
    wire: the hub records INVALID and no operator can tell which function was
    wrong. This is the contract the rest of the refusal matrix already holds
    to, and the reason this suite treats refusals as first-class behaviors."""
    with hub() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-unknown-named", attempt=1, function_name="nope",
            input_payload=catalog.row("echo").input_bytes(text="marco")))
        res = conn.wait_for(is_result_for("r-unknown-named")).job_result
        assert res.status == pb.JOB_STATUS_INVALID
        assert "nope" in res.safe_message, (
            f"refusal did not name the unknown function: {res.safe_message!r}")
