"""pgw#943: a worker YIELDS its GPU permit while awaiting a child call.

Paul named the first real consumer of ``ctx.call_endpoint``:

    *"an endpoint with cozy-eval that evaluates the results of like, model
    training and quantization, and then our quantization functions can call
    this and get the results back. It can process more requests (pipelining)
    while it's waiting for the results of the previous request."*

PR #455 measured the defect: ``_run_job`` took the group permit before the
handler and held it until the handler returned, and ``call_endpoint`` never
entered ``_gpu_slot_yielded`` — so at ``gpu_slots=1`` every cozy-eval-shaped
pipeline serialized while renting an accelerator to wait on a network round
trip. The fix routes every child-call wait (``wait=True`` and the
``wait=False`` handle's ``.result()``) through ``ctx._child_call_wait``,
which yields the #382 GPU-slot lease for the park and re-acquires before
returning to tenant code.

The yield is SCOPED to jobs that hold no instance gate and no per-request
adapters, because both permit acquirers (``_run_job`` and ``_bg_turn``) take
the permit BEFORE the instance gate: a parked parent still holding
``run_lock`` would wedge its own re-acquire (the follower holds the permit
waiting on ``run_lock``; the parent waits on the permit — hold-and-wait
cycle), and a follower on a shared pipeline would clobber per-request
adapter state mid-handler. That scope is pinned below too.

Everything runs on the real executor, over the real ``@endpoint`` ->
``RequestContext`` -> ``callout.CalloutClient`` path, with a real HTTP server
on the other end speaking the documented callout wire contract. No model is
loaded and no inference runs.

pgw#949: every wait here is progress-gated and every assertion is about a
property — an ordering, or a share of a baseline measured in the same run —
never about the runner's speed.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import msgspec

from gen_worker.api import Resources, endpoint
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs
from harness.progress_wait import Cadence, await_count, await_progress

#: How much of the child call's OWN progress the observation window spans for
#: the gate-holding (non-yielding) case, counted in status polls the platform
#: answered. The parent provably cannot finish during it (the platform reports
#: `completed` only after `release()`), so this bounds the observation by work
#: the system under test performed rather than by elapsed time.
_OBSERVED_CHILD_POLLS = 100

#: The second handler's entry, once the permit comes free, must be a small
#: SHARE of how long the same permit had it blocked in this run. #455 measured
#: 10.7 ms against a multi-second block. A ratio keeps it a claim about WHAT
#: GATED the handler: a slow runner lengthens both sides.
_PROMPT_ENTRY_SHARE = 0.25


class _In(msgspec.Struct):
    tag: str = ""


class _Out(msgspec.Struct):
    tag: str


class _Platform:
    """The hub half of the callout contract, as a real local HTTP server.

    Each ``submit`` allocates ``child-req-N``; a child stays ``in_progress``
    until :meth:`release` (or :meth:`fail`) names it terminal. Every request
    the worker actually made is recorded, so "the await path was exercised"
    is an observation rather than an assumption.
    """

    def __init__(self) -> None:
        self.submitted = threading.Event()
        self.calls: List[Tuple[str, str]] = []
        self._lock = threading.Lock()
        self._children: List[str] = []
        self._terminal: Dict[str, Dict[str, Any]] = {}
        self._terminal_polls: Dict[str, int] = {}
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
                with platform._lock:
                    child_id = f"child-req-{len(platform._children) + 1}"
                    platform._children.append(child_id)
                platform.submitted.set()
                self._reply(200, {"request_id": child_id})

            def do_GET(self) -> None:  # noqa: N802
                self._record()
                child_id = self.path.rstrip("/").rsplit("/", 1)[-1]
                with platform._lock:
                    doc = platform._terminal.get(child_id)
                    if doc is not None:
                        platform._terminal_polls[child_id] = (
                            platform._terminal_polls.get(child_id, 0) + 1
                        )
                if doc is not None:
                    self._reply(200, doc)
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

    def children(self) -> List[str]:
        with self._lock:
            return list(self._children)

    def release(self, request_id: Optional[str] = None) -> None:
        """Mark one child (or every submitted child) completed."""
        with self._lock:
            targets = [request_id] if request_id else list(self._children)
            for rid in targets:
                self._terminal[rid] = {
                    "status": "completed",
                    "output": [f"child-ok:{rid}"],
                }

    def fail(self, request_id: Optional[str] = None) -> None:
        with self._lock:
            targets = [request_id] if request_id else list(self._children)
            for rid in targets:
                self._terminal[rid] = {
                    "status": "failed",
                    "error": {"type": "boom", "message": "child exploded"},
                }

    def polls(self, request_id: Optional[str] = None) -> int:
        with self._lock:
            if request_id is None:
                return sum(1 for method, _ in self.calls if method == "GET")
            return sum(
                1
                for method, path in self.calls
                if method == "GET" and path.rstrip("/").endswith(request_id)
            )

    def terminal_polls(self, request_id: str) -> int:
        with self._lock:
            return self._terminal_polls.get(request_id, 0)


# --------------------------------------------------------------------------
# fixtures: the two-function endpoint the issue asks for, plus variants
# --------------------------------------------------------------------------


class _Probe:
    """Per-run observation points, set from inside the handlers."""

    def __init__(self) -> None:
        self.parent_entered = threading.Event()
        #: Set the moment the (final) child-call wait returned to tenant code
        #: — i.e. after the slot re-acquire. Its ORDER against other events is
        #: what the re-acquire tests assert.
        self.parent_resumed = threading.Event()
        self.first_child_done = threading.Event()
        self.second_ran = threading.Event()
        self.blocker_entered = threading.Event()
        self.blocker_release = threading.Event()
        self.parent_result: List[Any] = []


_CHILD_CALL = dict(poll_interval_s=0.02, timeout_s=60.0)


def _endpoints(probe: _Probe, *, gpu: bool) -> List[Any]:
    """Plain-function fixtures (no instance gate — the yieldable shape)."""
    res = Resources(gpu=gpu)

    @endpoint(kind="inference", child_calls=True, resources=res, name="parent")
    def parent(ctx: Any, payload: _In) -> _Out:
        probe.parent_entered.set()
        out = ctx.call_endpoint(
            "harness/child-endpoint", "child", {"tag": payload.tag}, **_CHILD_CALL
        )
        probe.parent_resumed.set()
        probe.parent_result.append(out)
        return _Out(tag="parent-done")

    @endpoint(
        kind="inference", child_calls=True, resources=res, name="parent-deferred"
    )
    def parent_deferred(ctx: Any, payload: _In) -> _Out:
        probe.parent_entered.set()
        handle = ctx.call_endpoint(
            "harness/child-endpoint", "child", {"tag": payload.tag}, wait=False
        )
        out = handle.result(60.0, poll_interval_s=0.02)
        probe.parent_resumed.set()
        probe.parent_result.append(out)
        return _Out(tag="parent-done")

    @endpoint(kind="inference", child_calls=True, resources=res, name="parent-twice")
    def parent_twice(ctx: Any, payload: _In) -> _Out:
        probe.parent_entered.set()
        out1 = ctx.call_endpoint(
            "harness/child-endpoint", "child", {"tag": "one"}, **_CHILD_CALL
        )
        probe.parent_result.append(out1)
        probe.first_child_done.set()
        out2 = ctx.call_endpoint(
            "harness/child-endpoint", "child", {"tag": "two"}, **_CHILD_CALL
        )
        probe.parent_result.append(out2)
        probe.parent_resumed.set()
        return _Out(tag="parent-done")

    @endpoint(kind="inference", resources=res, name="second")
    def second(ctx: Any, payload: _In) -> _Out:
        probe.second_ran.set()
        return _Out(tag="second-done")

    @endpoint(kind="inference", resources=res, name="blocker")
    def blocker(ctx: Any, payload: _In) -> _Out:
        probe.blocker_entered.set()
        # No timeout: the tests release it in a ``finally``; a wedge here is
        # exactly what the surrounding progress-waits exist to report.
        probe.blocker_release.wait()
        return _Out(tag="blocker-done")

    return [parent, parent_deferred, parent_twice, second, blocker]


def _gated_endpoints(probe: _Probe) -> List[Any]:
    """A class-based (non-reentrant) parent: holds its instance gate for the
    whole handler, so its child calls must NOT yield the permit."""
    res = Resources(gpu=True)

    @endpoint(kind="inference", child_calls=True, resources=res)
    class GatedParent:
        def gated_parent(self, ctx: Any, payload: _In) -> _Out:
            if not payload.tag:
                # ensure_setup's warmup forward invokes the handler with a
                # synthetic default payload and a warmup ctx that has no
                # platform base URL — a child call there is meaningless.
                return _Out(tag="warmup")
            probe.parent_entered.set()
            out = ctx.call_endpoint(
                "harness/child-endpoint", "child", {"tag": payload.tag}, **_CHILD_CALL
            )
            probe.parent_resumed.set()
            probe.parent_result.append(out)
            return _Out(tag="parent-done")

    @endpoint(kind="inference", resources=res, name="second")
    def second(ctx: Any, payload: _In) -> _Out:
        probe.second_ran.set()
        return _Out(tag="second-done")

    return [GatedParent, second]


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


# --------------------------------------------------------------------------
# shared run scaffolding
# --------------------------------------------------------------------------


class _Run:
    def __init__(
        self, probe: _Probe, ex: Executor, platform: _Platform,
        sent: List[pb.WorkerMessage],
    ) -> None:
        self.probe = probe
        self.ex = ex
        self.platform = platform
        self.sent = sent

    def parent_done_early(self) -> Optional[str]:
        """Definitive failure: the parent answered before its child was
        released — it was supposed to be suspended on it. No clock involved."""
        if "req-parent" in _results(self.sent):
            return "the parent produced a result before its child was released"
        return None

    async def park_parent(self, function_name: str = "parent") -> None:
        """Dispatch the parent and wait until it is genuinely PARKED in the
        callout poll — gated on the platform answering `in_progress` twice."""
        await self.ex.handle_run_job(_run_job("req-parent", function_name))
        await _progress_count(
            self.platform.polls, 2,
            what="child-status polls proving the parent is parked",
            gone=self.parent_done_early,
        )
        assert self.platform.submitted.is_set(), "the child call was never submitted"

    async def finish(self) -> Dict[str, int]:
        await _progress(
            lambda: all(
                j.task.done() for j in self.ex.jobs.values() if j.task is not None
            ),
            lambda done: bool(done),
            what="every job task to finish",
        )
        for job in self.ex.jobs.values():
            if job.task is not None:
                await job.task  # surfaces a handler exception the results hid
        return _results(self.sent)


@asynccontextmanager
async def _running(
    specs_of: Callable[[_Probe], List[Any]],
) -> AsyncIterator[_Run]:
    probe = _Probe()
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    fns = specs_of(probe)
    specs = [s for fn in fns for s in extract_specs(fn)]
    # gpu_slots=1: the G == 1 packing every pod runs today. Group 0's permit
    # IS the whole pool, which is the configuration the cozy-eval pipelining
    # claim is about.
    ex = Executor(specs, _send, gpu_slots=1)
    with _Platform() as platform:
        ex.file_base_url = platform.base_url
        yield _Run(probe, ex, platform, sent)


# --------------------------------------------------------------------------
# 1. the await path is exercised at all — it never had been before #455
# --------------------------------------------------------------------------


def test_ctx_call_endpoint_actually_reaches_the_platform() -> None:
    """`requests.parent_request_id` was non-empty on 0 of 800 rows and
    `max(call_depth)` was 0: no worker had ever invoked `ctx.call`. This is
    the first execution of the submit -> poll -> result path end to end."""

    async def _measure() -> None:
        async with _running(
            lambda probe: _endpoints(probe, gpu=False)
        ) as run:
            await run.park_parent()
            run.platform.release()
            statuses = await run.finish()
            assert run.probe.parent_result, "ctx.call_endpoint returned no output"
            assert run.probe.parent_result[0] == ["child-ok:child-req-1"]
            assert statuses.get("req-parent") == pb.JOB_STATUS_OK

    asyncio.run(_measure())


# --------------------------------------------------------------------------
# 2. pipelining, both resource shapes
# --------------------------------------------------------------------------


async def _measure_pipelining(*, gpu: bool, parent_fn: str) -> None:
    """The fixed behaviour: while the parent is parked on its child, a second
    request is accepted, runs, and COMPLETES — then the parent resumes and
    completes once the child does. Pure ordering; no clocks."""
    async with _running(lambda probe: _endpoints(probe, gpu=gpu)) as run:
        await run.park_parent(parent_fn)
        await run.ex.handle_run_job(_run_job("req-second", "second"))
        await _progress(
            run.probe.second_ran.is_set, lambda entered: entered,
            what="the second handler entering while the parent is parked",
            gone=run.parent_done_early,
        )
        await _progress(
            lambda: "req-second" in _results(run.sent), lambda done: done,
            what="the second request completing while the parent is parked",
            gone=run.parent_done_early,
        )
        # The parent really was still suspended on its child throughout.
        assert not run.probe.parent_resumed.is_set()
        assert "req-parent" not in _results(run.sent)
        run.platform.release()
        statuses = await run.finish()
        assert "req-second" in _accepted(run.sent)
        assert statuses.get("req-parent") == pb.JOB_STATUS_OK, statuses
        assert statuses.get("req-second") == pb.JOB_STATUS_OK, statuses
        assert run.probe.parent_result[0] == ["child-ok:child-req-1"]


def test_the_event_loop_keeps_accepting_and_running_during_a_child_call() -> None:
    """CPU function endpoints (no GPU permit) pipeline — intake was never the
    blocking site. Sync handlers run in ``asyncio.to_thread``, so a handler
    parked in ``CalloutClient.wait`` does not block the loop."""
    asyncio.run(_measure_pipelining(gpu=False, parent_fn="parent"))


def test_a_gpu_request_yields_its_permit_across_the_whole_child_call() -> None:
    """THE pgw#943 FIX, pinned. At `gpu_slots=1` the second GPU request now
    runs TO COMPLETION while the parent is parked on its child: the child-call
    wait yields the #382 lease instead of renting the accelerator for a
    network round trip, and the parent re-acquires and completes once the
    child does. (#455 pinned the opposite — the permit held across the whole
    call — and its assertion was inverted into this one, per its own
    instruction.)"""
    asyncio.run(_measure_pipelining(gpu=True, parent_fn="parent"))


def test_wait_false_handle_result_also_yields() -> None:
    """The yield must not depend on which waiting style tenant code picked:
    ``wait=False`` + ``handle.result()`` parks in the same guard."""
    asyncio.run(_measure_pipelining(gpu=True, parent_fn="parent-deferred"))


# --------------------------------------------------------------------------
# 3. re-acquisition: contention, composition, failure, cancellation
# --------------------------------------------------------------------------


def test_reacquire_waits_out_a_contender_then_the_parent_resumes() -> None:
    """The re-acquire path under contention, made deterministic: a follower
    HOLDS the permit (its handler blocks on an event) while the parent's
    child completes. The parent cannot resume until the follower releases —
    ``parent_resumed`` staying unset is gated on the semaphore itself, not on
    timing — and then it does resume and completes. No wedge, no leak."""

    async def _measure() -> None:
        async with _running(
            lambda probe: _endpoints(probe, gpu=True)
        ) as run:
            try:
                await run.park_parent()
                await run.ex.handle_run_job(_run_job("req-blocker", "blocker"))
                # The follower could only enter because the parked parent
                # YIELDED the permit — this is the yield, observed.
                await _progress(
                    run.probe.blocker_entered.is_set, lambda e: e,
                    what="the blocker entering on the yielded permit",
                    gone=run.parent_done_early,
                )
                run.platform.release()
                # The parent has SEEN its child terminal...
                await _progress_count(
                    lambda: run.platform.terminal_polls("child-req-1"), 1,
                    what="the parent observing its child's terminal status",
                )
                # ...but cannot have resumed: resuming requires re-acquiring
                # the permit the blocker still holds. This is a property of
                # the semaphore, not of how long we waited.
                assert not run.probe.parent_resumed.is_set()
                assert "req-parent" not in _results(run.sent)
                assert run.ex._gpu_permits[0].locked()
            finally:
                run.probe.blocker_release.set()
            statuses = await run.finish()
            assert run.probe.parent_resumed.is_set()
            assert statuses.get("req-parent") == pb.JOB_STATUS_OK, statuses
            assert statuses.get("req-blocker") == pb.JOB_STATUS_OK, statuses
            assert not run.ex._gpu_permits[0].locked(), "permit leaked"

    asyncio.run(_measure())


def test_sequential_child_calls_compose_yield_reacquire_yield() -> None:
    """Depth composition on one worker: two sequential child calls in one
    handler yield, re-acquire, and yield again — a follower pipelines inside
    EACH wait, and the balance stays exact."""

    async def _measure() -> None:
        async with _running(
            lambda probe: _endpoints(probe, gpu=True)
        ) as run:
            await run.park_parent("parent-twice")
            # Yield #1: a follower completes inside the first wait.
            await run.ex.handle_run_job(_run_job("req-second", "second"))
            await _progress(
                lambda: "req-second" in _results(run.sent), lambda done: done,
                what="a follower completing inside the FIRST child wait",
                gone=run.parent_done_early,
            )
            run.platform.release("child-req-1")
            # Re-acquire #1 worked: the parent moved on to its second call...
            await _progress(
                run.probe.first_child_done.is_set, lambda done: done,
                what="the parent resuming after its first child",
                gone=run.parent_done_early,
            )
            # ...and is parked on child 2 (the platform answered for it twice).
            await _progress_count(
                lambda: run.platform.polls("child-req-2"), 2,
                what="child-status polls proving the parent is parked again",
                gone=run.parent_done_early,
            )
            # Yield #2: another follower completes inside the second wait.
            await run.ex.handle_run_job(_run_job("req-third", "second"))
            await _progress(
                lambda: "req-third" in _results(run.sent), lambda done: done,
                what="a follower completing inside the SECOND child wait",
                gone=run.parent_done_early,
            )
            assert not run.probe.parent_resumed.is_set()
            run.platform.release("child-req-2")
            statuses = await run.finish()
            assert statuses.get("req-parent") == pb.JOB_STATUS_OK, statuses
            assert run.probe.parent_result == [
                ["child-ok:child-req-1"], ["child-ok:child-req-2"],
            ]
            assert not run.ex._gpu_permits[0].locked(), "permit leaked"

    asyncio.run(_measure())


def test_child_failure_mid_yield_reacquires_and_fails_cleanly() -> None:
    """A child that FAILS while the parent's slot is yielded: the wait raises
    the typed error only after re-acquiring, the job fails cleanly, and the
    permit balance stays exact — a subsequent GPU request runs fine."""

    async def _measure() -> None:
        async with _running(
            lambda probe: _endpoints(probe, gpu=True)
        ) as run:
            await run.park_parent()
            run.platform.fail("child-req-1")
            await _progress(
                lambda: "req-parent" in _results(run.sent), lambda done: done,
                what="the parent failing on its child's failure",
            )
            statuses = _results(run.sent)
            assert statuses["req-parent"] != pb.JOB_STATUS_OK, statuses
            assert statuses["req-parent"] != pb.JOB_STATUS_CANCELED, statuses
            assert not run.probe.parent_resumed.is_set()
            # The permit came back on the failure path: a follower serves.
            assert not run.ex._gpu_permits[0].locked(), "permit leaked"
            await run.ex.handle_run_job(_run_job("req-second", "second"))
            statuses = await run.finish()
            assert statuses.get("req-second") == pb.JOB_STATUS_OK, statuses

    asyncio.run(_measure())


def test_cancel_mid_yield_leaves_the_permit_free() -> None:
    """A parent cancelled while parked: the callout wait raises promptly (the
    cancel event, not a poll deadline), the re-acquired slot is released
    again for the dying job (#382's cancel discipline), and the permit is
    free for the next request."""

    async def _measure() -> None:
        async with _running(
            lambda probe: _endpoints(probe, gpu=True)
        ) as run:
            await run.park_parent()
            run.ex.handle_cancel(pb.CancelJob(request_id="req-parent", attempt=1))
            await _progress(
                lambda: "req-parent" in _results(run.sent), lambda done: done,
                what="the cancelled parent reaching a terminal result",
            )
            statuses = _results(run.sent)
            assert statuses["req-parent"] == pb.JOB_STATUS_CANCELED, statuses
            assert not run.ex._gpu_permits[0].locked(), "permit leaked"
            await run.ex.handle_run_job(_run_job("req-second", "second"))
            statuses = await run.finish()
            assert statuses.get("req-second") == pb.JOB_STATUS_OK, statuses

    asyncio.run(_measure())


# --------------------------------------------------------------------------
# 4. the scope: a gate-holding parent must KEEP its permit
# --------------------------------------------------------------------------


def test_gate_holding_parent_keeps_the_permit_across_child_calls() -> None:
    """A non-reentrant CLASS endpoint holds its instance gate (``run_lock``)
    for the whole handler. Yielding its permit would wedge: both `_run_job`
    and `_bg_turn` take the permit BEFORE the gate, so a same-instance
    follower (or mint turn) would hold the permit while waiting on the
    parked parent's `run_lock`, and the parent's re-acquire would wait on
    that permit forever — hold-and-wait cycle. So for this shape the child
    call KEEPS the permit (pre-#943 behaviour), pinned here so widening the
    yield is a deliberate act (it requires inverting the permit/gate lock
    order first — see the pgw#943 tracker entry)."""

    async def _measure() -> None:
        async with _running(_gated_endpoints) as run:
            gated_name = next(
                s.name for fn in _gated_endpoints(_Probe())
                for s in extract_specs(fn) if s.cls is not None
            )
            await run.park_parent(gated_name)
            polls_at_dispatch = run.platform.polls()
            blocked_from = time.monotonic()
            await run.ex.handle_run_job(_run_job("req-second", "second"))
            # The observation window: _OBSERVED_CHILD_POLLS of the child
            # call's own progress, ending early if the second handler enters
            # (which would be the fix's scope leaking). Group 0's permit is
            # sampled on every observation.
            permit_samples: List[bool] = []

            def _observe() -> Tuple[int, bool]:
                permit_samples.append(run.ex._gpu_permits[0].locked())
                return run.platform.polls(), run.probe.second_ran.is_set()

            await _progress(
                _observe,
                lambda seen: (
                    seen[1]
                    or seen[0] - polls_at_dispatch >= _OBSERVED_CHILD_POLLS
                ),
                what=(
                    f"{_OBSERVED_CHILD_POLLS} child-status polls with the "
                    f"gate-holding parent suspended"
                ),
                gone=run.parent_done_early,
            )
            assert not run.probe.second_ran.is_set(), (
                "a gate-holding parent yielded its permit — that shape wedges "
                "on re-acquire (see docstring); the scope stamp regressed"
            )
            assert all(permit_samples)
            assert run.platform.polls() - polls_at_dispatch >= _OBSERVED_CHILD_POLLS
            blocked_s = time.monotonic() - blocked_from
            released_at = time.monotonic()
            run.platform.release()
            await _progress(
                run.probe.second_ran.is_set, lambda entered: entered,
                what="the second handler entering once the permit is free",
                poll_s=0.005,
            )
            second_latency = time.monotonic() - released_at
            # The ordering, as a share of a baseline measured in this same
            # run: the handler entered on the permit coming free, in a small
            # fraction of the time that same permit had it blocked. A slow
            # runner lengthens both sides.
            assert second_latency < blocked_s * _PROMPT_ENTRY_SHARE, (
                second_latency, blocked_s,
            )
            statuses = await run.finish()
            assert statuses.get("req-parent") == pb.JOB_STATUS_OK, statuses
            assert statuses.get("req-second") == pb.JOB_STATUS_OK, statuses

    asyncio.run(_measure())
