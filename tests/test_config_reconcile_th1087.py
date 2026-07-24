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

import os
import sys
from pathlib import Path

import msgspec
import pytest

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.request_context import RequestContext
from gen_worker.runtime_config import (
    SNAPSHOT_PATH_ENV,
    ConfigSnapshotWriteError,
    ConfigStore,
    read_snapshot,
)
from gen_worker.subproc import run_process

from harness.hub_double import hub_double, is_ready, is_result_for
from harness.toy_endpoints import EchoIn


def test_gen_bump_rewrites_snapshot_and_subprocess_sees_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snap_path = tmp_path / "cfg" / "runtime_config.msgpack"
    # ConfigStore exports the path into os.environ; route that through
    # monkeypatch so the export never leaks across tests.
    monkeypatch.setenv(SNAPSHOT_PATH_ENV, str(snap_path))
    store = ConfigStore(str(snap_path))

    # A desired-state push advertises gen 3; a dispatch stamps the values.
    assert store.observe(3, release_id="rel-1")
    assert store.stamp_function("config-echo", {"default_steps": 60}, 3)
    on_disk = read_snapshot(str(snap_path))
    assert on_disk.config_generation == 3
    assert on_disk.release_id == "rel-1"
    assert on_disk.parameters["config-echo"]["default_steps"] == 60

    # Stale/duplicate generations and unchanged stamps are ignored — file
    # untouched.
    before = snap_path.read_bytes()
    assert not store.observe(2)
    assert not store.observe(3)
    assert not store.stamp_function("config-echo", {"default_steps": 1}, 2)
    assert not store.stamp_function("config-echo", {"default_steps": 60}, 3)
    assert snap_path.read_bytes() == before

    # A real subprocess with an EXPLICIT env mapping still finds the
    # snapshot (run_process injects the known-path env var) and reads the
    # post-bump value.
    assert store.stamp_function("config-echo", {"default_steps": 75}, 4)
    lines: list[str] = []
    code = run_process(
        [
            sys.executable,
            "-c",
            "from gen_worker.runtime_config import read_snapshot; "
            "s = read_snapshot(); "
            "print(s.config_generation, s.parameters['config-echo']['default_steps'])",
        ],
        env={},
        on_line=lines.append,
    )
    assert code == 0, lines
    assert lines == ["4 75"]

    # A job already stamped at an older generation gets an immutable
    # per-invocation subprocess snapshot. A newer global push cannot change
    # the bytes that run_process(ctx=...) exposes to that child.
    ctx = RequestContext("old-gen")
    old_values = {"default_steps": 40}
    ctx._set_config(
        old_values,
        snapshot=store.invocation_snapshot(
            "config-echo",
            old_values,
            4,
        ),
    )
    assert store.stamp_function("config-echo", {"default_steps": 100}, 5)
    lines = []
    code = run_process(
        [
            sys.executable,
            "-c",
            "from gen_worker.runtime_config import read_snapshot; "
            "s = read_snapshot(); "
            "print(s.config_generation, s.parameters['config-echo']['default_steps'])",
        ],
        ctx=ctx,
        env={},
        on_line=lines.append,
    )
    assert code == 0, lines
    assert lines == ["4 40"]
    assert read_snapshot(str(snap_path)).config_generation == 5


def test_failed_snapshot_write_does_not_advance_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snap_path = tmp_path / "runtime_config.msgpack"
    store = ConfigStore(str(snap_path))
    assert store.observe(1, release_id="rel-1")
    before = snap_path.read_bytes()

    def fail_replace(_source: str, _target: str) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(ConfigSnapshotWriteError):
        store.observe(2, release_id="rel-1")

    assert store.generation == 1
    assert snap_path.read_bytes() == before
    assert read_snapshot(str(snap_path)).config_generation == 1


def test_full_parameter_snapshot_is_atomic_and_release_fenced(
    tmp_path: Path,
) -> None:
    snap_path = tmp_path / "runtime_config.msgpack"
    store = ConfigStore(str(snap_path))
    raw = msgspec.msgpack.encode(
        {
            "config-echo": {"default_steps": 42, "scheduler": "euler"},
        }
    )
    assert store.apply_parameter_snapshot(
        raw,
        4,
        release_id="release-1",
    )
    assert read_snapshot(str(snap_path)).parameters == {
        "config-echo": {"default_steps": 42, "scheduler": "euler"},
    }
    before = snap_path.read_bytes()

    with pytest.raises(ConfigSnapshotWriteError, match="release_id mismatch"):
        store.apply_parameter_snapshot(
            raw,
            5,
            release_id="release-2",
        )
    assert store.generation == 4
    assert snap_path.read_bytes() == before


def _run_config_echo(
    conn,
    request_id: str,
    *,
    generation: int = 0,
    params: dict[str, object] | None = None,
) -> str:
    conn.send(
        run_job=pb.RunJob(
            request_id=request_id,
            attempt=1,
            function_name="config-echo",
            input_payload=msgspec.msgpack.encode(EchoIn(text="x")),
            config_generation=generation,
            config_params=msgspec.msgpack.encode(params) if params is not None else b"",
        )
    )
    res = conn.wait_for(is_result_for(request_id)).job_result
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    return msgspec.msgpack.decode(res.inline)["response"]


def test_config_push_serves_next_request_pod_churn_free(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snap_path = tmp_path / "runtime_config.msgpack"
    monkeypatch.setenv(SNAPSHOT_PATH_ENV, str(snap_path))
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        assert _run_config_echo(conn, "r-before") == "ddim:30"

        # The hub's config-write push: a full-replace HelloAck advertises
        # the desired generation; the next RunJob carries this function's
        # effective values as msgpack.
        conn.send(
            hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                file_base_url=scheduler.file_base_url,
                desired_residency=pb.DesiredResidency(
                    release_id="rel-1",
                    config_generation=2,
                ),
            )
        )
        conn.wait_for(
            lambda m: (
                m.WhichOneof("msg") == "state_delta"
                and m.state_delta.observed_config_generation == 2
            )
        )

        # Same worker, same connection, no pod churn: the NEXT request
        # serves the pushed values, and the snapshot file tracked the push.
        values = {"default_steps": 90, "scheduler": "euler_a"}
        assert (
            _run_config_echo(
                conn,
                "r-after",
                generation=2,
                params=values,
            )
            == "euler_a:90"
        )
        snapshot = read_snapshot(str(snap_path))
        assert snapshot.config_generation == 2
        assert snapshot.release_id == "rel-1"
        assert snapshot.parameters["config-echo"] == values

        # Undeclared parameter names never leak into ctx.config OR the
        # subprocess snapshot.
        conn.send(
            hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                file_base_url=scheduler.file_base_url,
                desired_residency=pb.DesiredResidency(
                    release_id="rel-1",
                    config_generation=3,
                ),
            )
        )
        assert (
            _run_config_echo(
                conn,
                "r-undeclared",
                generation=3,
                params={**values, "bogus": True},
            )
            == "euler_a:90"
        )
        assert "bogus" not in read_snapshot(str(snap_path)).parameters["config-echo"]

        # A job already stamped at the older generation keeps its own
        # values, but cannot roll the worker's latest snapshot backward.
        assert (
            _run_config_echo(
                conn,
                "r-in-flight-old-gen",
                generation=2,
                params={"default_steps": 40, "scheduler": "ddim"},
            )
            == "ddim:40"
        )
        assert read_snapshot(str(snap_path)).config_generation == 3
