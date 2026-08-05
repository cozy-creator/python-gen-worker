"""The hub-double: an in-process ``grpc.server`` playing the orchestrator,
driving a REAL ``gen_worker.worker.Worker`` over a REAL TCP gRPC socket.

Extracted from ``tests/test_worker_grpc_e2e.py``'s ``FakeScheduler`` (#365) per
th#960/pgw#609 — this is the ONLY double anywhere in the pgw suite: the true
process boundary the worker does not own. Everything downstream of the
socket (transport, lifecycle, executor, registry) is the real worker.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from concurrent import futures
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import grpc

from gen_worker import config as gw_config
from gen_worker.config import load_settings
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
from gen_worker.worker import Worker

from harness.progress_wait import Cadence, StalledError

DEFAULT_TIMEOUT_S = 15.0

# pgw#795: how often a progress-gated wait re-evaluates its staleness window
# while nothing is arriving. Not a deadline — the condition variable wakes it
# the instant a message lands; this only bounds how stale the window estimate
# may get when the peer is silent.
_REEVALUATE_S = 0.25

#: What a compute child pays before it can say anything: a fresh interpreter,
#: the Worker import (torch rides in on it) and the endpoint modules the child
#: was told to load. Nothing observes that gap from the hub side, so it is the
#: one silence a healthy boot really produces — and it is what gets measured.
_BOOT_PROBE_SRC = """
import importlib, os
from gen_worker.worker import Worker  # noqa: F401
for m in os.environ.get("PGW763_CHILD_MODULES", "harness.procsplit_endpoints").split(","):
    if m:
        importlib.import_module(m)
"""


def measure_child_boot_cost_s(env: Optional[Mapping[str, str]] = None) -> float:
    """Spawn-plus-import, measured on THIS runner at THIS moment (pgw#960).

    Boot cost is a property of the loaded box, not of the code under test, so a
    wait that must cover it asks rather than assumes. Called only when a wait is
    already about to give up, so the common path pays nothing.
    """
    child_env: Dict[str, str] = dict(os.environ)
    child_env.update(env or {})
    started = time.monotonic()
    subprocess.run(
        [sys.executable, "-c", _BOOT_PROBE_SRC],
        env=child_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    )
    return time.monotonic() - started


def _extended(cadence: Cadence, boot_cost: Optional[Callable[[], float]]) -> bool:
    """Re-measure the runner; keep waiting only if it says the box got slower.

    Self-limiting: on a steady box the second measurement matches the first, the
    window does not move, and the wait ends. Only evidence of a genuinely slower
    runner buys more patience — never a repeat of a literal.
    """
    if boot_cost is None:
        return False
    before = cadence.window_s
    cadence.record(boot_cost())
    return cadence.window_s > before


def _label(m: pb.WorkerMessage) -> str:
    which = m.WhichOneof("msg")
    if which == "job_result":
        return f"job_result({m.job_result.request_id})"
    if which == "job_accepted":
        return f"job_accepted({m.job_accepted.request_id})"
    return str(which)


class Conn:
    """One live worker connection as seen by the fake scheduler."""

    def __init__(self) -> None:
        self.hello: Optional[pb.Hello] = None
        self.received: List[pb.WorkerMessage] = []
        self._recv_cond = threading.Condition()
        self._out: "queue.Queue[Any]" = queue.Queue()
        self.client_done = threading.Event()
        # pgw#960: set by the scheduler that made this conn. A wait that spans a
        # child boot (Ready needs every group advertised) has no in-band signal
        # to gate on, so its window is measured instead of assumed.
        self.boot_cost: Optional[Callable[[], float]] = None

    def send(self, **oneof: Any) -> None:
        self._out.put(pb.SchedulerMessage(**oneof))

    def kill(self) -> None:
        """Abruptly fail the stream (server-side error) — simulates a dead hub."""
        self._out.put(RuntimeError("killed"))

    def close(self) -> None:
        """End the response stream cleanly."""
        self._out.put(None)

    def _record(self, msg: pb.WorkerMessage) -> None:
        with self._recv_cond:
            self.received.append(msg)
            self._recv_cond.notify_all()

    def _wait(
        self,
        take: Callable[[], Optional[pb.WorkerMessage]],
        describe: Callable[[], str],
        timeout: Optional[float],
    ) -> pb.WorkerMessage:
        """The one waiting loop (pgw#795).

        ``timeout=None`` — the default for "this MUST happen" waits — is
        progress-gated: it ends when the worker delivers, when the worker is
        provably gone (stream ended: definitive, no clock), or when the wait has
        gone a staleness window without the message it asked for. That window is
        derived from the advances this run has actually measured, so it widens
        with the machine. Only the awaited message counts as progress: unrelated
        traffic on the connection does not reset the window, because a peer
        chattering about other things is not progress toward what you asked for
        — and a window that resets on it never closes, so a broken test would
        hang instead of failing (measured while authoring this).

        Every wait that passes today does so inside the 15s TOTAL budget this
        replaced, and the floor alone is twice that, so this is strictly more
        patient than what it replaces — it just stops being patient for the
        wrong reason.

        An explicit ``timeout`` is preserved verbatim for the callers that
        probe for ABSENCE ("no result within 2s") — there the bound IS the
        assertion, and its expiry makes the test pass rather than flake.
        """
        # pgw#795 round 4: a FRESH cadence per wait. It used to be per-
        # connection and, before that, session-wide — and a shared slowest
        # sample let one slow advance widen every later wait until a
        # zero-progress wait hung for 13 minutes. A wait that observes no
        # advance of its own is now bounded by the floor, always.
        cadence = Cadence()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._recv_cond:
            last_advance = time.monotonic()
            while True:
                got = take()
                if got is not None:
                    return got
                now = time.monotonic()
                if deadline is not None:
                    remaining = deadline - now
                    if remaining <= 0:
                        raise TimeoutError(f"{describe()} within {timeout}s")
                    self._recv_cond.wait(remaining)
                    continue
                if self.client_done.is_set():
                    raise StalledError(
                        f"{describe()}: the worker ended the stream, so no "
                        f"further message can arrive"
                    )
                silent = now - last_advance
                if silent >= cadence.window_s:
                    # The probe spawns a process; recording must not be blocked
                    # behind it, or a message arriving mid-probe reads as silence.
                    self._recv_cond.release()
                    try:
                        widened = _extended(cadence, self.boot_cost)
                    finally:
                        self._recv_cond.acquire()
                    if not widened:
                        raise StalledError(
                            f"{describe()}: no such message in {silent:.1f}s "
                            f"(staleness window {cadence.describe()})"
                        )
                    continue
                self._recv_cond.wait(min(_REEVALUATE_S, cadence.window_s - silent))

    def wait_for(
        self,
        pred: Callable[[pb.WorkerMessage], bool],
        timeout: Optional[float] = None,
    ) -> pb.WorkerMessage:
        checked = 0

        def _take() -> Optional[pb.WorkerMessage]:
            nonlocal checked
            for msg in self.received[checked:]:
                checked += 1
                if pred(msg):
                    return msg
            return None

        return self._wait(
            _take,
            lambda: f"no matching message; got {[_label(m) for m in self.received]}",
            timeout,
        )

    def count(self, pred: Callable[[pb.WorkerMessage], bool]) -> int:
        with self._recv_cond:
            return sum(1 for m in self.received if pred(m))

    def wait_for_count(
        self,
        pred: Callable[[pb.WorkerMessage], bool],
        count: int,
        timeout: Optional[float] = None,
    ) -> pb.WorkerMessage:
        def _take() -> Optional[pb.WorkerMessage]:
            matches = [m for m in self.received if pred(m)]
            return matches[count - 1] if len(matches) >= count else None

        return self._wait(
            _take,
            lambda: (
                f"only {sum(1 for m in self.received if pred(m))} matching "
                f"messages; wanted {count}"
            ),
            timeout,
        )


class FakeScheduler(pb_grpc.WorkerSchedulerServicer):
    """The hub double. Plays HelloAck-before-anything-else, records every
    inbound WorkerMessage per connection, and can reject the handshake
    outright (auth-rejection test rows)."""

    def __init__(
        self, *, reject_unauthenticated: bool = False,
        file_base_url: str = "http://127.0.0.1:1/files",
    ) -> None:
        self.connections: List[Conn] = []
        self._conn_cond = threading.Condition()
        self.reject_unauthenticated = reject_unauthenticated
        self.file_base_url = file_base_url
        # pgw#795: set by hub_double() once the worker exists. A worker that
        # has EXITED can never dial in — that is the definitive give-up for
        # wait_connection, and it needs no clock.
        self.worker_alive: Optional[Callable[[], bool]] = None
        # pgw#960: set by harnesses whose worker boots in a SUBPROCESS, where
        # the silence a wait must tolerate is the child's spawn-plus-import.
        self.boot_cost: Optional[Callable[[], float]] = None

    def Connect(self, request_iterator: Any, context: grpc.ServicerContext) -> Any:
        if self.reject_unauthenticated:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "bad worker jwt")

        first = next(request_iterator)
        assert first.WhichOneof("msg") == "hello", "first message must be Hello"
        conn = Conn()
        conn.hello = first.hello
        conn.boot_cost = self.boot_cost
        # Queue the HelloAck BEFORE exposing the connection: the contract says
        # HelloAck precedes all other scheduler->worker traffic.
        conn.send(hello_ack=pb.HelloAck(
            protocol_version=pb.PROTOCOL_VERSION_CURRENT,
            file_base_url=self.file_base_url,
        ))
        with self._conn_cond:
            self.connections.append(conn)
            self._conn_cond.notify_all()

        def _reader() -> None:
            try:
                for msg in request_iterator:
                    conn._record(msg)
            except Exception:
                pass
            finally:
                conn.client_done.set()
                conn._out.put(None)  # end the response stream too

        threading.Thread(target=_reader, daemon=True).start()
        while True:
            item = conn._out.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def wait_connection(self, index: int, timeout: Optional[float] = None) -> Conn:
        """Wait for the worker to dial in (pgw#795: progress-gated by default).

        A boot on a loaded runner is slow, not broken — the honest give-up is
        "the worker process is gone", which is definitive, plus a staleness
        window calibrated from the advances this run has measured.

        pgw#960: a dial-in has no intermediate signal to advance on — the boot
        is silent until it lands — so the window is calibrated from what a boot
        COSTS on this runner, measured only when the wait is already at its
        floor. That is the difference between a bound the box earns and the
        180 s literal callers used to pass in to escape this branch entirely.
        """
        cadence = Cadence()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._conn_cond:
            started = time.monotonic()
            while len(self.connections) <= index:
                now = time.monotonic()
                if deadline is not None:
                    remaining = deadline - now
                    if remaining <= 0:
                        raise TimeoutError(
                            f"connection #{index} never arrived within "
                            f"{timeout}s ({len(self.connections)} so far)"
                        )
                    self._conn_cond.wait(remaining)
                    continue
                if self.worker_alive is not None and not self.worker_alive():
                    raise StalledError(
                        f"connection #{index} never arrived: the worker exited "
                        f"({len(self.connections)} connections so far)"
                    )
                waited = now - started
                if waited >= cadence.window_s:
                    # Probe outside the lock — Connect() appends under it.
                    self._conn_cond.release()
                    try:
                        widened = _extended(cadence, self.boot_cost)
                    finally:
                        self._conn_cond.acquire()
                    if not widened:
                        raise StalledError(
                            f"connection #{index} never arrived in {waited:.1f}s of "
                            f"silence (staleness window {cadence.describe()}); "
                            f"{len(self.connections)} connections so far"
                        )
                    continue
                self._conn_cond.wait(min(_REEVALUATE_S, cadence.window_s - waited))
            return self.connections[index]


class WorkerHarness:
    """Runs a REAL ``Worker`` against a hub-double connection in a background
    thread. ``modules`` is the endpoint-module list handed to ``Worker`` —
    callers pick which toy endpoints to expose per test.

    ``cache_dir`` is REQUIRED to be test-scoped (never the process default):
    the CAS store persists real bytes to disk keyed by wire ref, so two
    tests sharing a cache dir (or the host's default) silently see each
    other's "hub-delivered" state — a real bug this harness hit once
    (th#960 P3 authoring notes) and now refuses to repeat.
    """

    def __init__(
        self,
        scheduler: FakeScheduler,
        port: int,
        cache_dir: Path,
        *,
        modules: Sequence[str] = ("harness.toy_endpoints",),
        worker_id: str = "hub-double-worker",
        gpu_slots: int = 1,
        backoff_base_s: float = 0.05,
        backoff_cap_s: float = 0.2,
    ) -> None:
        self.scheduler = scheduler
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id=worker_id,
            worker_jwt="",
            tensorhub_cache_dir=str(cache_dir),
        )
        self.worker = Worker(
            settings,
            list(modules),
            gpu_slots=gpu_slots,
            backoff_base_s=backoff_base_s,
            backoff_cap_s=backoff_cap_s,
        )
        self.exit_code: Optional[int] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.exit_code = self.worker.run()

    @property
    def alive(self) -> bool:
        """Whether the worker is still running — the liveness half of every
        progress-gated wait against it (pgw#795)."""
        return self._thread.is_alive()

    def reconcile_marker(self) -> tuple:
        """A MARKER of residency-reconcile work, for progress-gated waits.

        Deliberately not a "reconcile in flight?" boolean: a task that is stuck
        is also not done, so a wait that refreshed on that predicate would
        never end (pgw#795 red-verify measured exactly that hang). What counts
        as evidence is CHANGE — a new pass, a pass finishing, a different work
        item — so this returns the identity of both.
        """
        lifecycle = self.worker.lifecycle
        task = getattr(lifecycle, "_residency_task", None)
        return (
            id(task) if task is not None else None,
            task.done() if task is not None else None,
            repr(getattr(lifecycle, "_reconcile_active", None)),
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[int]:
        """Ask the worker to stop, and REQUIRE that it did.

        pgw#795: this used to `join(15.0)` and throw the result away, so a
        worker that never exited passed teardown in silence — a wedged shutdown
        is exactly the defect this harness exists to catch, and it was the one
        outcome nothing asserted. The join is now progress-gated (a thread that
        is still alive is not progress) and a survivor is a loud failure.
        """
        self.worker.stop()
        self._join_or_fail(timeout, "stop()")
        return self.exit_code

    def join(self, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[int]:
        self._join_or_fail(timeout, "join()")
        return self.exit_code

    def _join_or_fail(self, timeout: float, who: str) -> None:
        # ``timeout`` is a silence FLOOR: a caller's existing number can only
        # make this more patient, never less.
        cadence = Cadence(floor_s=max(timeout, Cadence().floor_s))
        started = time.monotonic()
        while self._thread.is_alive():
            waited = time.monotonic() - started
            if waited >= cadence.window_s:
                raise StalledError(
                    f"the worker thread was still alive {waited:.1f}s after "
                    f"{who} (staleness window {cadence.describe()}); a wedged "
                    f"shutdown must never pass teardown quietly"
                )
            self._thread.join(min(0.25, cadence.window_s - waited))


@contextmanager
def hub_double(
    modules: Sequence[str] = ("harness.toy_endpoints",),
    *,
    reject_unauthenticated: bool = False,
    worker_id: str = "hub-double-worker",
    gpu_slots: int = 1,
    backoff_base_s: float = 0.05,
    backoff_cap_s: float = 0.2,
    max_workers: int = 16,
    cache_dir: Optional[Path] = None,
    file_base_url: str = "http://127.0.0.1:1/files",
) -> Iterator[Tuple[FakeScheduler, WorkerHarness]]:
    """Stand up one hub-double gRPC server + one real Worker against it.
    Tears both down on exit even if the body raises. ``cache_dir`` defaults
    to a fresh temp dir PER CALL (never a shared/default cache) so real
    downloaded bytes from one test can never leak into another's "boot saw
    nothing on disk yet" assumptions.

    ``TENSORHUB_CACHE_DIR`` is the ONLY thing that actually steers the CAS
    root (``gen_worker.models.cache_paths.tensorhub_cache_dir`` reads the
    process-wide cached ``gw_config.current()``, not the per-worker ``Settings``
    instance) — passing ``tensorhub_cache_dir=`` to ``load_settings()``
    alone does NOT redirect it. Found the hard way authoring P3's
    boot-precedence test: without this, every hub-double test on a dev box
    shares (and pollutes) ``/tmp/tensorhub-cache``."""
    prior_env = os.environ.get("TENSORHUB_CACHE_DIR")
    scheduler = FakeScheduler(
        reject_unauthenticated=reject_unauthenticated, file_base_url=file_base_url,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    with tempfile.TemporaryDirectory(prefix="pgw-hub-double-cache-") as tmp:
        resolved_cache_dir = cache_dir or Path(tmp)
        os.environ["TENSORHUB_CACHE_DIR"] = str(resolved_cache_dir)
        gw_config.reload_for_test()
        harness = WorkerHarness(
            scheduler, port, cache_dir=resolved_cache_dir,
            modules=modules, worker_id=worker_id, gpu_slots=gpu_slots,
            backoff_base_s=backoff_base_s, backoff_cap_s=backoff_cap_s,
        )
        scheduler.worker_alive = lambda: harness.alive
        harness.start()
        try:
            yield scheduler, harness
        finally:
            harness.stop()
            server.stop(grace=0)
            if prior_env is None:
                os.environ.pop("TENSORHUB_CACHE_DIR", None)
            else:
                os.environ["TENSORHUB_CACHE_DIR"] = prior_env
            gw_config.reload_for_test()


@contextmanager
def custom_scheduler_server(
    servicer_factory: Callable[[], Any],
    *,
    modules: Sequence[str] = ("harness.toy_endpoints",),
    worker_id: str = "hub-double-worker",
    backoff_base_s: float = 0.05,
    backoff_cap_s: float = 0.2,
    max_workers: int = 8,
    port: Optional[int] = None,
) -> Iterator[Tuple[Any, WorkerHarness, int]]:
    """Like ``hub_double()`` but for a BESPOKE ``WorkerSchedulerServicer``
    (auth-reject/precondition/redirect/stall scenarios) instead of the
    ordinary ``FakeScheduler``. ``port`` lets a caller rebind a second
    server onto the SAME address a worker already dialed (redirect tests).
    Callers own the servicer's own connection-tracking; only cache-dir
    isolation and worker lifecycle are handled here."""
    prior_env = os.environ.get("TENSORHUB_CACHE_DIR")
    servicer = servicer_factory()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb_grpc.add_WorkerSchedulerServicer_to_server(servicer, server)
    bound_port = server.add_insecure_port(f"127.0.0.1:{port or 0}")
    server.start()
    with tempfile.TemporaryDirectory(prefix="pgw-hub-double-cache-") as tmp:
        os.environ["TENSORHUB_CACHE_DIR"] = tmp
        gw_config.reload_for_test()
        harness = WorkerHarness(
            servicer, bound_port, cache_dir=Path(tmp), modules=modules, worker_id=worker_id,
            backoff_base_s=backoff_base_s, backoff_cap_s=backoff_cap_s,
        )
        harness.start()
        try:
            yield servicer, harness, bound_port
        finally:
            harness.stop()
            server.stop(grace=0)
            if prior_env is None:
                os.environ.pop("TENSORHUB_CACHE_DIR", None)
            else:
                os.environ["TENSORHUB_CACHE_DIR"] = prior_env
            gw_config.reload_for_test()


# ---------------------------------------------------------------------------
# Predicate helpers shared across P1/P2/P3/P6/P9.
# ---------------------------------------------------------------------------


def is_result_for(rid: str) -> Callable[[pb.WorkerMessage], bool]:
    return lambda m: m.WhichOneof("msg") == "job_result" and m.job_result.request_id == rid


def is_accept_for(rid: str) -> Callable[[pb.WorkerMessage], bool]:
    return lambda m: m.WhichOneof("msg") == "job_accepted" and m.job_accepted.request_id == rid


def is_ready(m: pb.WorkerMessage) -> bool:
    return m.WhichOneof("msg") == "state_delta" and m.state_delta.phase == pb.WORKER_PHASE_READY


def is_model_event(ref: str, state: int) -> Callable[[pb.WorkerMessage], bool]:
    return lambda m: (
        m.WhichOneof("msg") == "model_event"
        and m.model_event.ref == ref
        and m.model_event.state == state
    )


def is_exact_model_event(
    ref: str, state: int, digest: str, generation: int,
) -> Callable[[pb.WorkerMessage], bool]:
    return lambda m: (
        is_model_event(ref, state)(m)
        and m.model_event.snapshot_digest == digest
        and m.model_event.residency_generation == generation
    )


def is_fn_unavailable(function_name: str) -> Callable[[pb.WorkerMessage], bool]:
    return lambda m: (
        m.WhichOneof("msg") == "fn_unavailable"
        and m.fn_unavailable.function_name == function_name
    )
