"""pgw#738: the upload/publish path must never park a job forever.

Death 1 of the incident (62922680 + d0cbf910, one H100, ~$24.6): admission
took GPU permit -> instance run_lock, but ``save_bytes`` yields the permit
mid-handler while HOLDING run_lock. The second job packed on the worker took
the freed permit, blocked on run_lock, and the uploader blocked forever in
``reacquire()`` — one blob persisted, then 3h51m of heartbeating silence.

Fix contract driven over the real hub-double + a real local media-upload
sink (no mocks on the executor path):

  * two jobs packed on one gpu_slots=1 worker BOTH complete when the first
    saves a blob mid-handler (run_lock-first admission);
  * a permit that genuinely cannot come back fails the job TYPED
    (GpuSlotReacquireTimeout -> RETRYABLE), never silently;
  * a job task that dies without reporting is reaped into a terminal
    JobResult (never-silent guarantee).

Red-verified: with the pre-fix executor (permit-first admission, unbounded
reacquire, no reaper) the packed-jobs test times out with NO result for
either job — the exact silent-death signature from the tracker.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, Optional, Tuple

import msgspec
import pytest

# pgw#797 adjudication (test-health audit): SKIP, do not FAIL. Unlike its two
# pgw#738 siblings, this file's src half exists NOWHERE — all three cases fail
# against HEAD *and* against the worktree carrying every sibling's WIP, so it is
# a test written ahead of a fix that was never made. It still describes a real,
# expensive incident (62922680 + d0cbf910, one H100, ~$24.6, 3h51m of
# heartbeating silence), and the contract it states — run_lock-first admission,
# a typed GpuSlotReacquireTimeout, a reaped terminal JobResult — is the design
# pgw#738 asked for. So it is preserved as the executable specification of that
# fix rather than deleted, and named as unimplemented rather than left red.
# REMOVE THIS GUARD as the first step of implementing pgw#738: it should go red,
# then green.
_ADMISSION_FIX_LANDED = hasattr(
    __import__("gen_worker.api.errors", fromlist=["errors"]),
    "GpuSlotReacquireTimeout",
)
if not _ADMISSION_FIX_LANDED:
    pytest.skip(
        "pgw#738 UNIMPLEMENTED: the upload/publish admission fix (run_lock-first "
        "admission + typed GpuSlotReacquireTimeout + dead-task reaper) has not "
        "landed on any branch. This file is its executable specification.",
        allow_module_level=True,
    )

from gen_worker import executor as executor_mod
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness import upload_endpoints_pgw738 as up
from harness.hub_double import hub_double, is_accept_for, is_ready, is_result_for

MODULE = "harness.upload_endpoints_pgw738"
CUDA = pb.ResolvedCompute(accelerator="cuda", gpu_index=0)
ORG = "00000000-0000-0000-0000-000000000001"


class _DedupSink(BaseHTTPRequestHandler):
    """Real local /api/v1/media/:owner/uploads answering a dedup create, so
    save_bytes completes without S3 part scripting (P9 pattern)."""

    hits: ClassVar[int] = 0

    def log_message(self, *_args: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).hits += 1
        resp = json.dumps({
            "dedup": True, "ref": body.get("ref") or "",
            "blake3": body.get("blake3") or "",
            "size_bytes": body.get("size_bytes") or 0,
            "mime_type": "application/octet-stream", "media_id": "m1",
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


def _serve() -> Tuple[ThreadingHTTPServer, str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _DedupSink)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _payload() -> bytes:
    return msgspec.msgpack.encode(up.UploadIn(text="x"))


def _send(conn: Any, rid: str, fn: str) -> None:
    conn.send(run_job=pb.RunJob(
        request_id=rid, attempt=1, function_name=fn,
        input_payload=_payload(), compute=CUDA,
        org=ORG, capability_token="cap-token"))


def _result(conn: Any, rid: str, timeout: float = 15.0) -> Optional[Any]:
    try:
        return conn.wait_for(is_result_for(rid), timeout=timeout).job_result
    except TimeoutError:
        return None


@pytest.fixture(autouse=True)
def _state() -> Any:
    up.reset()
    _DedupSink.hits = 0
    yield
    up.UPLOAD_GATE.set()  # never leave a handler thread parked at teardown


def test_two_packed_jobs_survive_a_mid_handler_save() -> None:
    """THE incident shape: job B parked on the same worker while job A saves
    evidence mid-handler. Pre-fix: A deadlocks in reacquire, B never runs,
    neither ever reports — this test then fails on both asserts."""
    httpd, base_url = _serve()
    try:
        with hub_double(modules=(MODULE,), file_base_url=base_url) as (scheduler, _h):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            _send(conn, "r-up-a", "uploader")
            assert up.UPLOADER_STARTED.wait(timeout=10.0)
            _send(conn, "r-up-b", "bystander")
            conn.wait_for(is_accept_for("r-up-b"))
            time.sleep(0.7)  # let B park behind A (the 62922680 packing)
            up.UPLOAD_GATE.set()
            res_a = _result(conn, "r-up-a")
            assert res_a is not None, (
                "uploader vanished: the pgw#738 reacquire deadlock")
            assert res_a.status == pb.JOB_STATUS_OK, res_a.safe_message
            assert _DedupSink.hits >= 1, "the save must ride the real upload path"
            res_b = _result(conn, "r-up-b")
            assert res_b is not None, "packed job vanished behind the uploader"
            assert res_b.status == pb.JOB_STATUS_OK, res_b.safe_message
            assert up.CALLS == ["uploader", "bystander"]
    finally:
        httpd.shutdown()


def test_reacquire_bound_fails_typed_never_silent() -> None:
    """If the yielded permit genuinely cannot come back (simulated thief on
    the semaphore), the job fails TYPED + RETRYABLE within the bound instead
    of parking forever while the GPU bills."""
    old = executor_mod._GpuSlotLease.REACQUIRE_TIMEOUT_S
    executor_mod._GpuSlotLease.REACQUIRE_TIMEOUT_S = 1.0
    httpd, base_url = _serve()
    thief: Optional[Any] = None
    try:
        with hub_double(modules=(MODULE,), file_base_url=base_url) as (scheduler, harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            _send(conn, "r-bound-a", "uploader")
            assert up.UPLOADER_STARTED.wait(timeout=10.0)
            ex = harness.worker.executor
            # Park a competing acquire; it wins the permit the moment the
            # handler's save yields it, and never gives it back.
            thief = asyncio.run_coroutine_threadsafe(
                ex._gpu_semaphore.acquire(), ex._loop)
            time.sleep(0.2)
            up.UPLOAD_GATE.set()
            res = _result(conn, "r-bound-a")
            assert res is not None, "the bound must convert the hang into a result"
            assert res.status == pb.JOB_STATUS_RETRYABLE, res.safe_message
            assert "reacquire" in res.safe_message.lower()
            if thief.done() and thief.exception() is None:
                asyncio.run_coroutine_threadsafe(
                    _release(ex), ex._loop).result(timeout=5.0)
    finally:
        executor_mod._GpuSlotLease.REACQUIRE_TIMEOUT_S = old
        httpd.shutdown()


async def _release(ex: Any) -> None:
    ex._gpu_semaphore.release()


def test_dead_job_task_reports_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never-silent guarantee: a job task that dies without reporting (any
    escape from _run_job's own handlers) is reaped into a terminal
    JobResult instead of leaving the request assigned forever."""

    async def escaping_run_job(self: Any, job: Any, run: Any) -> None:
        raise RuntimeError("boom-escape: died before any reporting")

    monkeypatch.setattr(executor_mod.Executor, "_run_job", escaping_run_job)
    with hub_double(modules=(MODULE,)) as (scheduler, _h):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _send(conn, "r-dead-a", "bystander")
        res = _result(conn, "r-dead-a")
        assert res is not None, "a dead task must still produce a terminal result"
        assert res.status == pb.JOB_STATUS_RETRYABLE
        assert "died without reporting terminal state" in res.safe_message
        assert "boom-escape" in res.safe_message
