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

pgw#949: every wait here is progress-gated and every assertion is about a
property, not about the runner's speed. The observation window is denominated
in child-status polls the platform actually answered, so a slow machine
observes the SAME thing more slowly rather than observing less of it; the
promptness claim is a share of a baseline this same run measured. That is what
the finding always was — an ordering fact ("the second handler entered only
once the permit came free"), never a stopwatch reading.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

import msgspec

from gen_worker.api import Resources, endpoint
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs
from harness.progress_wait import Cadence, await_count, await_progress

CHILD_REQUEST_ID = "child-req-pgw943"

#: How much of the child call's OWN progress the observation window spans,
#: counted in status polls the platform answered. The parent provably cannot
#: finish during it (the platform reports `completed` only after `release()`),
#: so this bounds the observation by work the system under test performed
#: rather than by elapsed time: a slower runner takes longer to reach the same
#: count and observes exactly the same window of behaviour.
_OBSERVED_CHILD_POLLS = 100

#: The second handler's entry, once the permit comes free, must be a small
#: SHARE of how long the same permit had it blocked in this run. #455 measured
#: 10.7 ms against a multi-second block. Stating it as a ratio keeps it a claim
#: about WHAT GATED the handler: a slow runner lengthens both sides.
_PROMPT_ENTRY_SHARE = 0.25


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


# --------------------------------------------------------------------------
# progress-gated waiting, off the loop the executor is running on
# --------------------------------------------------------------------------


async def _progress(
    observe: Callable[[], Any],
    settled: Callable[[Any], bool],
    *,
    what: str,
    gone: Optional[Callable[[], Optional[str]]] = None,
    poll_s: float = 0.02,
) -> Any:
    """``harness.progress_wait.await_progress`` on a worker thread.

    The harness wait is synchronous and the executor's job tasks live on THIS
    event loop, so waiting on the loop would manufacture the very stall the
    wait exists to observe. Off-loading keeps the loop free, so what the wait
    sees is the worker's own scheduling and nothing else.
    """
    return await asyncio.to_thread(
        await_progress,
        observe,
        settled,
        what=what,
        cadence=Cadence(),
        gone=gone,
        poll_s=poll_s,
    )


async def _progress_count(
    observe: Callable[[], int],
    want: int,
    *,
    what: str,
    gone: Optional[Callable[[], Optional[str]]] = None,
) -> int:
    return int(
        await asyncio.to_thread(
            await_count, observe, want, what=what, cadence=Cadence(), gone=gone
        )
    )


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
    #: Seconds from the child completing to the second handler entering. Only
    #: ever compared against the baseline below, never against a constant.
    second_handler_latency_after_child_s: Optional[float]
    #: Seconds the second request spent blocked while the child was in flight —
    #: the baseline the field above is a share of. Both are measured in the
    #: same run on the same machine, so their RATIO is a property of what
    #: gated the handler rather than of how fast the runner is.
    blocked_while_child_in_flight_s: float
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

        def _parent_done_early() -> Optional[str]:
            """Nothing further can be learned once the parent has answered —
            it was supposed to be suspended on its child. Definitive, and no
            clock is involved."""
            if "req-parent" in _results(sent):
                return "the parent produced a result before its child was released"
            return None

        await ex.handle_run_job(_run_job("req-parent", "parent"))
        # The parent is genuinely PARKED in the callout poll once the platform
        # has answered `in_progress` twice — it is inside CalloutClient.wait,
        # not merely about to enter it. Gated on the platform's own answers, so
        # a parent that never gets there stalls the wait (and says what it saw)
        # instead of a clock deciding for it.
        await _progress_count(
            platform.polls, 2,
            what="child-status polls before the second dispatch",
            gone=_parent_done_early,
        )
        assert platform.submitted.is_set(), "the child call was never submitted"

        polls_at_dispatch = platform.polls()
        blocked_from = time.monotonic()
        await ex.handle_run_job(_run_job("req-second", "second"))

        # The observation window: _OBSERVED_CHILD_POLLS of the child call's own
        # progress, ending early the moment the second handler enters. The
        # parent cannot finish during it (the platform reports `completed` only
        # after release(), below), so the window is a quantity of work the
        # system performed rather than a duration the runner had to keep up
        # with. Group 0's permit is sampled on every observation.
        permit_samples: List[bool] = []

        def _observe() -> Tuple[int, bool]:
            permit_samples.append(ex._gpu_permits[0].locked())
            return platform.polls(), _second_ran.is_set()

        await _progress(
            _observe,
            lambda seen: (seen[1]
                          or seen[0] - polls_at_dispatch >= _OBSERVED_CHILD_POLLS),
            what=(
                f"{_OBSERVED_CHILD_POLLS} child-status polls with the parent "
                f"suspended, or the second handler entering first"
            ),
            gone=_parent_done_early,
        )
        second_ran_while_waiting = _second_ran.is_set()
        permit_locked_throughout = all(permit_samples)
        polls_during_window = platform.polls() - polls_at_dispatch
        # The parent is still suspended: it has produced no result, and the
        # platform has not yet reported its child terminal.
        parent_in_flight = "req-parent" not in _results(sent)
        assert not platform.released.is_set()

        blocked_s = time.monotonic() - blocked_from
        released_at = time.monotonic()
        platform.release()
        second_latency: Optional[float] = None
        if not second_ran_while_waiting:
            # It must now enter, and the wait ends on that happening — never on
            # a deadline. If the permit was NOT what gated it, nothing advances
            # and the wait fails saying so.
            await _progress(
                _second_ran.is_set, lambda entered: entered,
                what="the second handler entering once the permit is free",
                poll_s=0.005,
            )
            second_latency = time.monotonic() - released_at

        await _progress(
            lambda: frozenset(_results(sent)),
            lambda seen: {"req-parent", "req-second"} <= seen,
            what="both job results",
        )
        tasks = [j.task for j in ex.jobs.values() if j.task is not None]
        await _progress(
            lambda: sum(1 for t in tasks if t.done()),
            lambda done: done == len(tasks),
            what="every job task to finish",
        )
        for task in tasks:
            await task  # surfaces a handler exception the results hid

    statuses = _results(sent)
    return _Measurement(
        child_call_made=bool(_parent_result),
        second_accepted="req-second" in _accepted(sent),
        second_handler_ran_while_parent_waited=second_ran_while_waiting,
        child_polls_while_waiting=polls_during_window,
        parent_still_in_flight=parent_in_flight,
        gpu_permit_locked_throughout=permit_locked_throughout,
        second_handler_latency_after_child_s=second_latency,
        blocked_while_child_in_flight_s=blocked_s,
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
    # The window really did span the child call's own progress — the permit
    # samples above are not a handful taken while nothing was happening.
    assert m.child_polls_while_waiting >= _OBSERVED_CHILD_POLLS
    assert m.second_handler_latency_after_child_s is not None
    # THE ORDERING, as a share of a baseline measured in this same run: the
    # handler entered on the permit coming free, in a small fraction of the
    # time that same permit had it blocked (#455: 10.7 ms against a
    # multi-second block). So the permit was the ONLY thing gating it — and a
    # slow runner lengthens both sides, which is why this is a claim about the
    # code rather than about the machine.
    assert m.second_handler_latency_after_child_s < (
        m.blocked_while_child_in_flight_s * _PROMPT_ENTRY_SHARE
    ), (m.second_handler_latency_after_child_s, m.blocked_while_child_in_flight_s)
    # It is only parked, not lost: both finish once the child completes.
    assert m.parent_status == pb.JOB_STATUS_OK
    assert m.second_status == pb.JOB_STATUS_OK


if __name__ == "__main__":  # pragma: no cover - manual measurement run
    for gpu in (False, True):
        print(f"gpu={gpu}: {asyncio.run(_measure(gpu=gpu))}")
