"""DESIGN-RULINGS §1.27(b),(g): the wire-protocol MAJOR is the proto package,
so it is in the gRPC service path.

Two halves, and the second is the safety one:

* the client dials ``/cozy.scheduler.v1.WorkerScheduler/Connect`` — a package
  rename that got silently reverted would be invisible otherwise, because every
  in-repo test uses generated stubs on both ends and so agrees with itself no
  matter what the package says;
* a hub that does NOT serve this major answers ``UNIMPLEMENTED`` at the routing
  layer, and the worker must treat that as FATAL and exit. That exit is
  load-bearing hub-side: it is what makes the pod die before Hello, leaving
  ``everConnected`` false, which is how the hub's death taxonomy marks the
  release ``boot_crashing`` and fails its queued requests. A worker that retried
  instead would reconnect-loop forever with no durable mark and no operator
  signal — strictly worse than dying.

The negative case is driven by a REAL gRPC server registered under the
pre-v1 unversioned service name, which is exactly the production shape of "new
worker, hub not yet cut over".
"""

from __future__ import annotations

import threading
from concurrent import futures
from typing import Optional

import grpc

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.transport import _CONNECT_METHOD

from harness.hub_double import hub_double, is_ready

_TIMEOUT = 15.0

#: The service path this repo's proto no longer speaks. A hub still serving it
#: is a hub that predates the v1 cut.
_PRE_V1_SERVICE = "cozy.scheduler.WorkerScheduler"


def test_connect_path_carries_the_major() -> None:
    """§1.27(b): the major lives in the package, therefore in the path."""
    assert pb.DESCRIPTOR.package == "cozy.scheduler.v1"
    assert _CONNECT_METHOD == "/cozy.scheduler.v1.WorkerScheduler/Connect"
    # §1.27(g): the first package is v1, and nothing pre-launch claims a
    # history it does not have.
    assert pb.PROTOCOL_VERSION_CURRENT == 1


def test_v1_worker_handshakes_with_a_v1_hub() -> None:
    """The positive control the refusal test below is only meaningful against:
    on the matching major the same worker connects and reports ready."""
    with hub_double(worker_id="th1597-right-major") as (scheduler, harness):
        conn = scheduler.wait_connection(0)
        assert conn.hello is not None
        assert conn.hello.protocol_version == pb.PROTOCOL_VERSION_CURRENT
        conn.wait_for(is_ready)
        assert harness.alive


def _serve_only_the_pre_v1_path() -> tuple[grpc.Server, int]:
    """A hub that serves the OLD unversioned path and nothing else. Any dial of
    the v1 path is answered UNIMPLEMENTED by gRPC itself, before a byte of the
    message body is interpreted — which is the whole point of putting the major
    in the routing key."""

    def _unreached(request_iterator, context):  # pragma: no cover - never dialed
        raise AssertionError("a v1 client must not reach the pre-v1 handler")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                _PRE_V1_SERVICE,
                {
                    "Connect": grpc.stream_stream_rpc_method_handler(
                        _unreached,
                        request_deserializer=pb.WorkerMessage.FromString,
                        response_serializer=pb.SchedulerMessage.SerializeToString,
                    )
                },
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


def test_a_hub_that_does_not_serve_this_major_is_fatal_not_a_reconnect_loop(
    caplog,
) -> None:
    import logging

    from gen_worker.config import load_settings
    from gen_worker.worker import Worker

    server, port = _serve_only_the_pre_v1_path()
    try:
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id="th1597-wrong-major",
            worker_jwt="",
        )
        worker = Worker(
            settings, ["harness.toy_endpoints"], backoff_base_s=0.05, backoff_cap_s=0.2
        )
        exit_code: Optional[int] = None

        def _run() -> None:
            nonlocal exit_code
            exit_code = worker.run()

        with caplog.at_level(logging.ERROR):
            thread = threading.Thread(target=_run, name="th1597-worker", daemon=True)
            thread.start()
            thread.join(timeout=_TIMEOUT)

        assert not thread.is_alive(), (
            "the worker is still running against a hub that does not serve its "
            "wire-protocol major — it is reconnect-looping, so no pod ever dies "
            "pre-Hello and th#874 can never mark the release boot_crashing"
        )
        assert exit_code == 1, f"expected a fatal exit, got {exit_code!r}"

        # It exited for the RIGHT reason, and the reason is typed. Asserting
        # only the exit code would pass for any fatal at all — including the
        # unreachable-hub and auth families, which are the ones this code path
        # must NOT be confused with. Deliberately NOT asserted: that zero
        # reconnect delays were recorded. A channel that is not ready yet
        # raises before any status arrives, and retrying THAT is correct; only
        # a delivered UNIMPLEMENTED is unretryable.
        fatal = "\n".join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR
        )
        assert "does not serve this wire-protocol major" in fatal, fatal
        assert _CONNECT_METHOD in fatal, fatal
    finally:
        server.stop(grace=0)
