"""th#1087 stage D: worker reconcile of config-generation pushes.

  1. A generation bump updates memory AND atomically rewrites the local
     snapshot file; a REAL subprocess (the run_process primitive, explicit
     env) reads the new values via ``read_snapshot`` — Paul's watched-file
     pattern for per-invoke subprocesses. Stale/duplicate generations are
     ignored.
  2. Over the hub-double wire: after a config push lands on the live
     worker's store, the next dispatched job's ``ctx.config`` serves the
     pushed value (read-at-dispatch), and a mid-stream HelloAck re-send —
     the hub's config-write propagation transport — is pod-churn-free: the
     same worker keeps serving.
"""

from __future__ import annotations

import sys
from pathlib import Path

import msgspec

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.runtime_config import (
    SNAPSHOT_PATH_ENV,
    ConfigStore,
    read_snapshot,
)
from gen_worker.subproc import run_process

from harness.hub_double import hub_double, is_ready, is_result_for
from harness.toy_endpoints import EchoIn


def test_gen_bump_rewrites_snapshot_and_subprocess_sees_it(
    tmp_path: Path, monkeypatch,
) -> None:
    snap_path = tmp_path / "cfg" / "runtime_config.json"
    # ConfigStore exports the path into os.environ; route that through
    # monkeypatch so the export never leaks across tests.
    monkeypatch.setenv(SNAPSHOT_PATH_ENV, str(snap_path))
    store = ConfigStore(str(snap_path))

    assert store.apply(
        generation=3,
        parameters={"config-echo": {"default_steps": 60}},
        release_id="rel-1",
    )
    on_disk = read_snapshot(str(snap_path))
    assert on_disk.config_generation == 3
    assert on_disk.parameters["config-echo"]["default_steps"] == 60

    # Stale and duplicate generations are ignored — file untouched.
    before = snap_path.read_bytes()
    assert not store.apply(generation=2, parameters={"config-echo": {"default_steps": 1}})
    assert not store.apply(generation=3, parameters={"config-echo": {"default_steps": 1}})
    assert snap_path.read_bytes() == before

    # A real subprocess with an EXPLICIT env mapping still finds the
    # snapshot (run_process injects the known-path env var) and reads the
    # post-bump value.
    store.apply(generation=4, parameters={"config-echo": {"default_steps": 75}})
    lines: list[str] = []
    code = run_process(
        [
            sys.executable, "-c",
            "from gen_worker.runtime_config import read_snapshot; "
            "s = read_snapshot(); "
            "print(s.config_generation, s.parameters['config-echo']['default_steps'])",
        ],
        env={},
        on_line=lines.append,
    )
    assert code == 0, lines
    assert lines == ["4 75"]


def _run_config_echo(conn, request_id: str) -> str:
    conn.send(run_job=pb.RunJob(
        request_id=request_id, attempt=1, function_name="config-echo",
        input_payload=msgspec.msgpack.encode(EchoIn(text="x"))))
    res = conn.wait_for(is_result_for(request_id)).job_result
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    return msgspec.msgpack.decode(res.inline)["response"]


def test_config_push_serves_next_request_pod_churn_free(
    tmp_path: Path, monkeypatch,
) -> None:
    snap_path = tmp_path / "runtime_config.json"
    monkeypatch.setenv(SNAPSHOT_PATH_ENV, str(snap_path))
    with hub_double() as (scheduler, harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        assert _run_config_echo(conn, "r-before") == "ddim:30"

        # The hub's config-write push: gen bump with new parameter values
        # (transport adapter is proto-gated; the store IS the landing
        # point) + the full-replace HelloAck re-send that carries it.
        store = harness.worker.executor.runtime_config
        assert store.apply(
            generation=2,
            parameters={"config-echo": {"default_steps": 90, "scheduler": "euler_a"}},
        )
        conn.send(hello_ack=pb.HelloAck(
            protocol_version=pb.PROTOCOL_VERSION_CURRENT,
            file_base_url=scheduler.file_base_url,
        ))

        # Same worker, same connection, no pod churn: the NEXT request
        # serves the pushed values, and the snapshot file tracked the push.
        assert _run_config_echo(conn, "r-after") == "euler_a:90"
        assert read_snapshot(str(snap_path)).config_generation == 2

        # Undeclared parameter names never leak into ctx.config.
        store.apply(
            generation=3,
            parameters={"config-echo": {"default_steps": 90, "scheduler": "euler_a",
                                        "bogus": True}},
        )
        assert _run_config_echo(conn, "r-undeclared") == "euler_a:90"
