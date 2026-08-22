"""The hub-double: an in-process ``grpc.server`` playing the orchestrator, driving a REAL ``gen_worker.worker.Worker`` over a REAL TCP gRPC socket."""

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

_REEVALUATE_S = 0.25

_BOOT_PROBE_SRC = """
import importlib, os
from gen_worker.worker import Worker  # noqa: F401
for m in os.environ.get("PGW763_CHILD_MODULES", "harness.procsplit_endpoints").split(","):
    if m:
        importlib.import_module(m)
"""


def measure_child_boot_cost_s(env: Optional[Mapping[str, str]] = None) -> float:
    child_env: Dict[str, str] = dict(os.environ)
    child_env.update(env or {})
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-c", _BOOT_PROBE_SRC],
        env=child_env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        check=False, text=True,
    )
    cost = time.monotonic() - started
    assert done.returncode == 0, (
        f"the child-boot probe could not run ({done.returncode}) — it is not "
        f"measuring what a child boot costs:\n{done.stderr[-2000:]}"
    )
    return cost


def _extended(cadence: Cadence, boot_cost: Optional[Callable[[], float]]) -> bool:
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
        self.diagnostic_reports: List[pb.HardwareUnsuitable] = []
        self._recv_cond = threading.Condition()
        self._out: "queue.Queue[Any]" = queue.Queue()
        self.client_done = threading.Event()
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
            lambda: (
                f"no matching message; got {[_label(m) for m in self.received]}; "
                f"diagnostics={[(r.reason_class, r.detail) for r in self.diagnostic_reports]}"
            ),
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
    """The hub double."""

    def __init__(
        self, *, reject_unauthenticated: bool = False,
        file_base_url: str = "http://127.0.0.1:1/files",
        hello_ack: Optional[Callable[[pb.Hello], pb.HelloAck]] = None,
    ) -> None:
        self.connections: List[Conn] = []
        self.diagnostic_reports: List[pb.HardwareUnsuitable] = []
        self._conn_cond = threading.Condition()
        self.reject_unauthenticated = reject_unauthenticated
        self.file_base_url = file_base_url
        #: Lets a test author the HelloAck the way the real hub would — the
        #: DesiredStateCommand the worker owes a GoalReceipt for lives there.
        self.hello_ack = hello_ack
        self.worker_alive: Optional[Callable[[], bool]] = None
        self.boot_cost: Optional[Callable[[], float]] = None

    def Connect(self, request_iterator: Any, context: grpc.ServicerContext) -> Any:
        if self.reject_unauthenticated:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "bad worker jwt")

        first = next(request_iterator)
        first_kind = first.WhichOneof("msg")
        if first_kind == "hardware_unsuitable":
            with self._conn_cond:
                self.diagnostic_reports.append(first.hardware_unsuitable)
                self._conn_cond.notify_all()
            for _ in request_iterator:
                pass
            return
        assert first_kind == "hello", (
            f"first message must be Hello, got {first_kind!r}")
        conn = Conn()
        conn.hello = first.hello
        conn.boot_cost = self.boot_cost
        conn.diagnostic_reports = self.diagnostic_reports
        if self.hello_ack is not None:
            ack = self.hello_ack(first.hello)
        else:
            ack = pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                file_base_url=self.file_base_url,
            )
        conn.send(hello_ack=ack)
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
                conn._out.put(None)

        threading.Thread(target=_reader, daemon=True).start()
        while True:
            item = conn._out.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def wait_connection(self, index: int, timeout: Optional[float] = None) -> Conn:
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
    """Runs a REAL ``Worker`` against a hub-double connection in a background thread."""

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
        release_id: str = "",
    ) -> None:
        self.scheduler = scheduler
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id=worker_id,
            worker_jwt="",
            worker_release_id=release_id,
            tensorhub_cache_dir=str(cache_dir),
        )
        self.worker = Worker(
            settings,
            list(modules),
            backoff_base_s=backoff_base_s,
            backoff_cap_s=backoff_cap_s,
        )
        self.exit_code: Optional[int] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.exit_code = self.worker.run()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def reconcile_marker(self) -> tuple:
        raise NotImplementedError(
            "no residency-reconcile loop on the v2 worker (pgw#1373): residency "
            "is per-request admission, so there is no pass to wait on"
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[int]:
        """Ask the worker to stop, and REQUIRE that it did."""
        self.worker.stop()
        self._join_or_fail(timeout, "stop()")
        return self.exit_code

    def join(self, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[int]:
        self._join_or_fail(timeout, "join()")
        return self.exit_code

    def _join_or_fail(self, timeout: float, who: str) -> None:
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
    release_id: str = "",
    hello_ack: Optional[Callable[[pb.Hello], pb.HelloAck]] = None,
) -> Iterator[Tuple[FakeScheduler, WorkerHarness]]:
    """Stand up one hub-double gRPC server + one real Worker against it."""
    prior_env = os.environ.get("TENSORHUB_CACHE_DIR")
    prior_config_path = os.environ.get("GEN_WORKER_CONFIG_SNAPSHOT_PATH")
    scheduler = FakeScheduler(
        reject_unauthenticated=reject_unauthenticated, file_base_url=file_base_url,
        hello_ack=hello_ack,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    with tempfile.TemporaryDirectory(prefix="pgw-hub-double-cache-") as tmp:
        resolved_cache_dir = cache_dir or Path(tmp)
        os.environ["TENSORHUB_CACHE_DIR"] = str(resolved_cache_dir)
        # th#1087's durable config snapshot: production injects this at pod
        # launch and privdrop grants its directory; a rig that leaves it unset
        # writes to /app and every config-generation push fails.
        os.environ["GEN_WORKER_CONFIG_SNAPSHOT_PATH"] = str(
            resolved_cache_dir / "runtime_config.msgpack"
        )
        gw_config.reload_for_test()
        harness = WorkerHarness(
            scheduler, port, cache_dir=resolved_cache_dir,
            modules=modules, worker_id=worker_id, gpu_slots=gpu_slots,
            backoff_base_s=backoff_base_s, backoff_cap_s=backoff_cap_s,
            release_id=release_id,
        )
        scheduler.worker_alive = lambda: harness.alive
        harness.start()
        try:
            yield scheduler, harness
        finally:
            harness.stop()
            server.stop(grace=0)
            for name, prior in (
                ("TENSORHUB_CACHE_DIR", prior_env),
                ("GEN_WORKER_CONFIG_SNAPSHOT_PATH", prior_config_path),
            ):
                if prior is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = prior
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
    """Like ``hub_double()`` but for a BESPOKE ``WorkerSchedulerServicer`` (auth-reject/precondition/redirect/stall scenarios) instead of the ordinary ``FakeScheduler``."""
    prior_env = os.environ.get("TENSORHUB_CACHE_DIR")
    prior_config_path = os.environ.get("GEN_WORKER_CONFIG_SNAPSHOT_PATH")
    servicer = servicer_factory()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb_grpc.add_WorkerSchedulerServicer_to_server(servicer, server)
    bound_port = server.add_insecure_port(f"127.0.0.1:{port or 0}")
    server.start()
    with tempfile.TemporaryDirectory(prefix="pgw-hub-double-cache-") as tmp:
        os.environ["TENSORHUB_CACHE_DIR"] = tmp
        os.environ["GEN_WORKER_CONFIG_SNAPSHOT_PATH"] = str(
            Path(tmp) / "runtime_config.msgpack"
        )
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
            for name, prior in (
                ("TENSORHUB_CACHE_DIR", prior_env),
                ("GEN_WORKER_CONFIG_SNAPSHOT_PATH", prior_config_path),
            ):
                if prior is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = prior
            gw_config.reload_for_test()


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
