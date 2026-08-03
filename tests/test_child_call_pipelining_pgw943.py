"""pgw#943: does a worker yield its request slot while awaiting a child call?

Paul named the first real consumer of ``ctx.call_endpoint``:

    *"an endpoint with cozy-eval that evaluates the results of like, model
    training and quantization, and then our quantization functions can call
    this and get the results back. It can process more requests (pipelining)
    while it's waiting for the results of the previous request."*

That last clause is a requirement, and ``async def`` proves nothing about it —
the question is whether the SCHEDULER releases the slot. So this is measured,
on the real executor, over the real ``@endpoint`` -> ``RequestContext`` ->
``callout.CalloutClient`` path, with a real HTTP server on the other end
speaking the documented callout wire contract (the platform API is external
to the worker; standing it up is not standing in for the worker runtime).

No model is loaded and no inference runs: the fixture is the two-function
endpoint the issue asks for, A calling B.

The measured answer, at ``gpu_slots=1`` (G == 1, every pod today):

* The EVENT LOOP is free. Sync handlers run in ``asyncio.to_thread``, so a
  handler parked in ``CalloutClient.wait``'s ``time.sleep``/``Event.wait``
  poll does not block intake. A second dispatch is accepted, ``JobAccepted``
  goes out, and its job task runs.
* The GPU PERMIT is NOT free. ``executor._run_job`` acquires the group permit
  BEFORE the handler and holds it until the handler returns
  (``executor.py:11741-11758``), and ``RequestContext.call_endpoint``
  (``request_context/__init__.py:719``) never enters ``_gpu_slot_yielded`` —
  the lease that ``save_bytes`` uses for exactly this reason (#382). So the
  second GPU request parks in ``WAIT_GPU_SLOT`` for the whole child call.

Both facts are pinned below, so a future change to either is a test failure
rather than a silent regression of a requirement nobody was measuring.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

import msgspec

from gen_worker.api import Resources, endpoint
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

CHILD_REQUEST_ID = "child-req-pgw943"


class _In(msgspec.Struct):
    tag: str = ""


class _Out(msgspec.Struct):
    tag: str


class _Platform:
    """The hub half of the callout contract, as a real local HTTP server.

    ``submit`` hands back a request id; the child stays ``in_progress`` until
    ``release()`` is called, then reports ``completed``. Every request the
    worker actually made is recorded, so "the await path was exercised" is an
    observation rather than an assumption.
    """

    def __init__(self) -> None:
        self.released = threading.Event()
        self.submitted = threading.Event()
        self.calls: List[Tuple[str, str]] = []
        self._lock = threading.Lock()
        platform = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                pass

            def _record(self) -> None:
                with platform._lock:
                    platform.calls.append((self.command, self.path))

            def _reply(self, code: int, doc: Dict[str, Any]) -> None:
                body = json.dumps(doc).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                self._record()
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                if self.path.startswith("/v1/requests/"):
                    self._reply(200, {})
                    return
                platform.submitted.set()
                self._reply(200, {"request_id": CHILD_REQUEST_ID})

            def do_GET(self) -> None:  # noqa: N802
                self._record()
                if platform.released.is_set():
                    self._reply(200, {"status": "completed", "output": ["child-ok"]})
                else:
                    self._reply(200, {"status": "in_progress"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_Platform":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def release(self) -> None:
        self.released.set()

    def polls(self) -> int:
        with self._lock:
            return sum(1 for method, _ in self.calls if method == "GET")


# --------------------------------------------------------------------------
# the fixture the issue asks for: two functions, A calls B
# --------------------------------------------------------------------------

_parent_entered = threading.Event()
_second_ran = threading.Event()
_parent_result: List[Any] = []


def _endpoints(*, gpu: bool) -> List[Any]:
    """The two-function fixture, decorated for the resource shape under test.

    Declared inside a factory only because ``EndpointSpec`` is frozen —
    ``resources=`` has to reach ``@endpoint`` itself. These are ordinary
    ``@endpoint`` functions and go through the ordinary
    ``extract_specs`` -> ``Executor`` path.
    """
    res = Resources(gpu=gpu)

    @endpoint(kind="inference", child_calls=True, resources=res, name="parent")
    def parent(ctx: Any, payload: _In) -> _Out:
        _parent_entered.set()
        out = ctx.call_endpoint(
            "harness/child-endpoint",
            "child",
            {"tag": payload.tag},
            poll_interval_s=0.02,
            timeout_s=60.0,
        )
        _parent_result.append(out)
        return _Out(tag="parent-done")

    @endpoint(kind="inference", resources=res, name="second")
    def second(ctx: Any, payload: _In) -> _Out:
        _second_ran.set()
        return _Out(tag="second-done")

    return [parent, second]


def _run_job(request_id: str, function_name: str) -> pb.RunJob:
    return pb.RunJob(
        request_id=request_id,
        attempt=1,
        function_name=function_name,
        input_payload=msgspec.msgpack.encode(_In(tag=request_id)),
        capability_token="pgw943-token",
    )


def _accepted(sent: List[pb.WorkerMessage]) -> List[str]:
    return [
        m.job_accepted.request_id
        for m in sent
        if m.WhichOneof("msg") == "job_accepted"
    ]


def _results(sent: List[pb.WorkerMessage]) -> Dict[str, int]:
    return {
        m.job_result.request_id: m.job_result.status
        for m in sent
        if m.WhichOneof("msg") == "job_result"
    }


class _Measurement(msgspec.Struct):
    """What the two-request run observed."""

    child_call_made: bool
    second_accepted: bool
    second_handler_ran_while_parent_waited: bool
    child_polls_while_waiting: int
    #: Had the parent still NOT produced a result when the observation window
    #: closed? Load-independent proof it really was suspended throughout.
    parent_still_in_flight: bool
    #: Was group 0's permit held for the whole observation window? This is
    #: what names the BLOCKING SITE: the semaphore, not intake, not the loop.
    gpu_permit_locked_throughout: bool
    #: Seconds from the child completing to the second handler entering. Near
    #: zero means the second request was queued on nothing but that permit.
    second_handler_latency_after_child_s: Optional[float]
    parent_status: Optional[int]
    second_status: Optional[int]


async def _measure(*, gpu: bool) -> _Measurement:
    _parent_entered.clear()
    _second_ran.clear()
    _parent_result.clear()

    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    fns = _endpoints(gpu=gpu)
    specs = [s for fn in fns for s in extract_specs(fn)]
    # gpu_slots=1: the G == 1 packing every pod runs today. Group 0's permit
    # IS the whole pool (executor.py:3100), which is the configuration the
    # cozy-eval pipelining claim is about.
    ex = Executor(specs, _send, gpu_slots=1)

    with _Platform() as platform:
        ex.file_base_url = platform.base_url

        await ex.handle_run_job(_run_job("req-parent", "parent"))
        # Wait until the parent is genuinely PARKED in the callout poll: the
        # platform has answered `in_progress` at least twice, so the handler
        # is inside CalloutClient.wait and not merely about to enter it.
        deadline = time.monotonic() + 30.0
        while platform.polls() < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert platform.submitted.is_set(), "the child call was never submitted"

        polls_at_dispatch = platform.polls()
        await ex.handle_run_job(_run_job("req-second", "second"))

        # Give the second request a generous window to make progress while the
        # parent is still suspended. The parent cannot finish during it: the
        # platform only reports `completed` after release(), below.
        window = time.monotonic() + 5.0
        permit_locked_throughout = True
        while time.monotonic() < window and not _second_ran.is_set():
            permit_locked_throughout &= ex._gpu_permits[0].locked()
            await asyncio.sleep(0.05)
        second_ran_while_waiting = _second_ran.is_set()
        polls_during_window = platform.polls() - polls_at_dispatch
        # The parent is still suspended: it has produced no result, and the
        # platform has not yet reported its child terminal. Asserted rather
        # than counting polls, which is a function of machine load.
        parent_in_flight = "req-parent" not in _results(sent)
        assert not platform.released.is_set()

        released_at = time.monotonic()
        platform.release()
        second_latency: Optional[float] = None
        if not second_ran_while_waiting:
            unblock = time.monotonic() + 30.0
            while time.monotonic() < unblock and not _second_ran.is_set():
                await asyncio.sleep(0.005)
            if _second_ran.is_set():
                second_latency = time.monotonic() - released_at

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if {"req-parent", "req-second"} <= set(_results(sent)):
                break
            await asyncio.sleep(0.05)
        for job in list(ex.jobs.values()):
            if job.task is not None and not job.task.done():
                await asyncio.wait_for(job.task, timeout=30)

    statuses = _results(sent)
    return _Measurement(
        child_call_made=bool(_parent_result),
        second_accepted="req-second" in _accepted(sent),
        second_handler_ran_while_parent_waited=second_ran_while_waiting,
        child_polls_while_waiting=polls_during_window,
        parent_still_in_flight=parent_in_flight,
        gpu_permit_locked_throughout=permit_locked_throughout,
        second_handler_latency_after_child_s=second_latency,
        parent_status=statuses.get("req-parent"),
        second_status=statuses.get("req-second"),
    )


# --------------------------------------------------------------------------
# 1. the await path is exercised at all — it never had been
# --------------------------------------------------------------------------


def test_ctx_call_endpoint_actually_reaches_the_platform() -> None:
    """`requests.parent_request_id` was non-empty on 0 of 800 rows and
    `max(call_depth)` was 0: no worker had ever invoked `ctx.call`. This is
    the first execution of the submit -> poll -> result path end to end."""
    m = asyncio.run(_measure(gpu=False))
    assert m.child_call_made, "ctx.call_endpoint returned no child output"
    assert m.parent_status == pb.JOB_STATUS_OK, m.parent_status
    assert _parent_result[0] == ["child-ok"]


# --------------------------------------------------------------------------
# 2. the answer, both halves
# --------------------------------------------------------------------------


def test_the_event_loop_keeps_accepting_and_running_during_a_child_call() -> None:
    """CPU function endpoints (no GPU permit) DO pipeline.

    Intake is not blocked by a parked handler — the sync handler is on a
    `asyncio.to_thread` worker, so `handle_run_job` accepts, `JobAccepted`
    goes out, and the second handler body runs to completion while the first
    is still polling its child.
    """
    m = asyncio.run(_measure(gpu=False))
    assert m.second_accepted
    assert m.second_handler_ran_while_parent_waited, (
        "a second request made no progress while a child call was in flight"
    )
    assert m.second_status == pb.JOB_STATUS_OK
    # The parent really was still suspended when the second one ran.
    assert m.parent_still_in_flight


def test_a_gpu_request_holds_its_permit_across_the_whole_child_call() -> None:
    """THE FINDING. At `gpu_slots=1` the second GPU request does NOT progress.

    `_run_job` takes the group permit before the handler and releases it after
    (`executor.py:11741-11758`); `call_endpoint` never yields the lease the way
    `save_bytes` does (`request_context/__init__.py:719` vs `_gpu_slot_yielded`).
    So the parent rents the accelerator for the duration of a network round
    trip, and every cozy-eval-shaped pipeline serializes while holding a GPU.

    This test pins the CURRENT behaviour so the fix flips it deliberately.
    """
    m = asyncio.run(_measure(gpu=True))
    assert m.second_accepted, "intake itself is not the blocking site"
    assert not m.second_handler_ran_while_parent_waited, (
        "the GPU permit is now yielded across child calls — pgw#943 is fixed; "
        "invert this assertion and delete the finding"
    )
    # The blocking site, named: group 0's permit was held for every sample of
    # the window, and the second handler entered only once the child completed.
    assert m.parent_still_in_flight
    assert m.gpu_permit_locked_throughout
    assert m.second_handler_latency_after_child_s is not None
    assert m.second_handler_latency_after_child_s < 2.0, (
        m.second_handler_latency_after_child_s
    )
    # It is only parked, not lost: both finish once the child completes.
    assert m.parent_status == pb.JOB_STATUS_OK
    assert m.second_status == pb.JOB_STATUS_OK


if __name__ == "__main__":  # pragma: no cover - manual measurement run
    for gpu in (False, True):
        print(f"gpu={gpu}: {asyncio.run(_measure(gpu=gpu))}")
