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
import tempfile
import threading
import time
from concurrent import futures
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import grpc

from gen_worker.config import get_settings, load_settings
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
from gen_worker.worker import Worker

DEFAULT_TIMEOUT_S = 15.0


class Conn:
    """One live worker connection as seen by the fake scheduler."""

    def __init__(self) -> None:
        self.hello: Optional[pb.Hello] = None
        self.received: List[pb.WorkerMessage] = []
        self._recv_cond = threading.Condition()
        self._out: "queue.Queue[Any]" = queue.Queue()
        self.client_done = threading.Event()

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

    def wait_for(
        self, pred: Callable[[pb.WorkerMessage], bool], timeout: float = DEFAULT_TIMEOUT_S,
    ) -> pb.WorkerMessage:
        deadline = time.monotonic() + timeout
        with self._recv_cond:
            checked = 0
            while True:
                for msg in self.received[checked:]:
                    checked += 1
                    if pred(msg):
                        return msg
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    def _label(m: pb.WorkerMessage) -> str:
                        which = m.WhichOneof("msg")
                        if which == "job_result":
                            return f"job_result({m.job_result.request_id})"
                        if which == "job_accepted":
                            return f"job_accepted({m.job_accepted.request_id})"
                        return str(which)
                    raise TimeoutError(
                        f"no matching message within {timeout}s; got "
                        f"{[_label(m) for m in self.received]}"
                    )
                self._recv_cond.wait(remaining)

    def count(self, pred: Callable[[pb.WorkerMessage], bool]) -> int:
        with self._recv_cond:
            return sum(1 for m in self.received if pred(m))

    def wait_for_count(
        self,
        pred: Callable[[pb.WorkerMessage], bool],
        count: int,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> pb.WorkerMessage:
        deadline = time.monotonic() + timeout
        with self._recv_cond:
            while True:
                matches = [m for m in self.received if pred(m)]
                if len(matches) >= count:
                    return matches[count - 1]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"only {len(matches)} matching messages within "
                        f"{timeout}s; wanted {count}"
                    )
                self._recv_cond.wait(remaining)


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

    def Connect(self, request_iterator: Any, context: grpc.ServicerContext) -> Any:
        if self.reject_unauthenticated:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "bad worker jwt")

        first = next(request_iterator)
        assert first.WhichOneof("msg") == "hello", "first message must be Hello"
        conn = Conn()
        conn.hello = first.hello
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

    def wait_connection(self, index: int, timeout: float = DEFAULT_TIMEOUT_S) -> Conn:
        deadline = time.monotonic() + timeout
        with self._conn_cond:
            while len(self.connections) <= index:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"connection #{index} never arrived "
                        f"({len(self.connections)} so far)"
                    )
                self._conn_cond.wait(remaining)
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

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[int]:
        self.worker.stop()
        self._thread.join(timeout)
        return self.exit_code

    def join(self, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[int]:
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "worker did not exit"
        return self.exit_code


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
    process-wide cached ``get_settings()``, not the per-worker ``Settings``
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
        get_settings.cache_clear()
        harness = WorkerHarness(
            scheduler, port, cache_dir=resolved_cache_dir,
            modules=modules, worker_id=worker_id, gpu_slots=gpu_slots,
            backoff_base_s=backoff_base_s, backoff_cap_s=backoff_cap_s,
        )
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
            get_settings.cache_clear()


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
