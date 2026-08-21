from __future__ import annotations

import os
import sys
import threading
from concurrent import futures
from pathlib import Path
from typing import Optional

import grpc
import msgspec
import pytest

from gen_worker import worker_fatal
from gen_worker.config import load_settings
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
from gen_worker.procsplit.parent import DEATH_LABEL, ParentControl
from gen_worker.topology import ExecutionTopology

from harness.hub_double import (
    FakeScheduler,
    is_accept_for,
    is_ready,
    is_result_for,
    measure_child_boot_cost_s,
)
from harness.progress_wait import Cadence, await_count, await_progress

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
CHILD_MAIN = TESTS_DIR / "harness" / "procsplit_child_main.py"


class _In(msgspec.Struct):
    text: str = ""


def _payload(text: str = "") -> bytes:
    return msgspec.msgpack.encode(_In(text=text))


def _postmortem_dir(tmp: Path) -> Path:
    d = tmp / "postmortem"
    d.mkdir(parents=True, exist_ok=True)
    return d


class G2Harness:
    """One hub-double + one ParentControl with a 2-group topology + two real child subprocesses."""

    def __init__(self, tmp: Path) -> None:
        self.scheduler = FakeScheduler()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
        pb_grpc.add_WorkerSchedulerServicer_to_server(self.scheduler, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id="split-parent-g2",
            worker_jwt="",
        )
        child_env = {
            "PYTHONPATH": os.pathsep.join(
                [str(TESTS_DIR), str(SRC_DIR), os.environ.get("PYTHONPATH", "")]
            ),
            "TENSORHUB_CACHE_DIR": str(tmp / "cache"),
            "GEN_WORKER_CHILD_WATCHDOG_PING_S": "0.5",
            "GEN_WORKER_BOOT_RECORD": str(_postmortem_dir(tmp) / "boot-record.json"),
        }
        self.pc = ParentControl(
            settings,
            child_cmd=[sys.executable, str(CHILD_MAIN)],
            child_env=child_env,
            socket_path=str(tmp / "ctl.sock"),
            topology=ExecutionTopology(gpu_count=2, gpus_per_execution_group=1),
            respawn_backoff_base_s=0.1,
            respawn_backoff_cap_s=0.5,
            transport_backoff_base_s=0.05,
            transport_backoff_cap_s=0.2,
            watchdog_budget_s=60.0,
        )
        self.exit_code: Optional[int] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.scheduler.worker_alive = lambda: self.alive
        self.scheduler.boot_cost = lambda: measure_child_boot_cost_s(child_env)
        self._thread.start()

    def _run(self) -> None:
        self.exit_code = self.pc.run()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def close(self) -> None:
        self.pc.stop()
        self._thread.join(20.0)
        self.server.stop(grace=0)


@pytest.fixture(autouse=True)
def _isolated_postmortem(tmp_path, monkeypatch):
    from gen_worker import postmortem

    d = _postmortem_dir(tmp_path)
    monkeypatch.setattr(
        postmortem, "INFLIGHT_PATH", d / "gen-worker-inflight.json"
    )
    monkeypatch.setattr(
        postmortem, "CRASH_REGISTRY_PATH", d / "gen-worker-crash-streaks.json"
    )
    monkeypatch.setattr(
        postmortem, "FAULT_DUMP_PATH", d / "gen-worker-fault-dump.txt"
    )
    return d


@pytest.fixture()
def _dials(monkeypatch):
    got = []
    monkeypatch.setattr(
        worker_fatal, "report_worker_detail",
        lambda settings, detail: (got.append(detail), True)[1],
    )
    return got


@pytest.fixture()
def g2(tmp_path, _dials):
    h = G2Harness(tmp_path)
    try:
        yield h
    finally:
        h.close()


def test_two_children_boot_and_each_group_serves_its_own_dispatch(g2):
    """The whole worker is READY only once BOTH groups are (the fan-in merges phase to the least-ready), and a dispatch to each group's rank-0 device is served by THAT group's child — proven by the ordina..."""
    conn = g2.scheduler.wait_connection(0)
    conn.wait_for(is_ready)

    assert g2.pc.execution_groups == 2
    assert all(slot.proc is not None for slot in g2.pc._slots)
    assert g2.pc._slots[0].socket_path != g2.pc._slots[1].socket_path

    conn.send(run_job=pb.RunJob(
        request_id="r-g0", attempt=1, function_name="whoami",
        input_payload=_payload(), compute=pb.ResolvedCompute(gpu_index=0)))
    r0 = conn.wait_for(is_result_for("r-g0"), timeout=60.0)
    assert r0.job_result.status == pb.JOB_STATUS_OK
    assert b"g=0" in r0.job_result.inline
    assert b"cvd=0" in r0.job_result.inline
    assert b"sib=2" in r0.job_result.inline

    conn.send(run_job=pb.RunJob(
        request_id="r-g1", attempt=1, function_name="whoami",
        input_payload=_payload(), compute=pb.ResolvedCompute(gpu_index=1)))
    r1 = conn.wait_for(is_result_for("r-g1"), timeout=60.0)
    assert r1.job_result.status == pb.JOB_STATUS_OK
    assert b"g=1" in r1.job_result.inline
    assert b"cvd=1" in r1.job_result.inline

    assert g2.alive and g2.exit_code is None


def test_one_childs_death_is_attributed_to_its_request_siblings_keep_serving(
    g2, _dials, tmp_path,
):
    conn = g2.scheduler.wait_connection(0)
    conn.wait_for(is_ready)

    slot1_pid_before = g2.pc._slots[1].proc.pid

    conn.send(run_job=pb.RunJob(
        request_id="r-sleep-g1", attempt=1, function_name="sleepy",
        input_payload=_payload(), compute=pb.ResolvedCompute(gpu_index=1)))
    conn.wait_for(is_accept_for("r-sleep-g1"), timeout=30.0)
    g1_marker = _postmortem_dir(tmp_path) / "g1" / "gen-worker-inflight.json"
    await_progress(
        g1_marker.exists,
        lambda exists: exists,
        what="g1 per-group in-flight marker",
        cadence=Cadence(),
        gone=lambda: None if g2.alive else f"parent exited {g2.exit_code}",
    )
    assert g1_marker.exists(), "g1 never published its per-group in-flight marker"

    conn.send(run_job=pb.RunJob(
        request_id="r-die-g0", attempt=1, function_name="segfault",
        input_payload=_payload(), compute=pb.ResolvedCompute(gpu_index=0)))
    died = conn.wait_for(is_result_for("r-die-g0"), timeout=60.0)
    assert died.job_result.status == pb.JOB_STATUS_FATAL
    assert DEATH_LABEL in died.job_result.safe_message
    assert "function=segfault" in died.job_result.safe_message

    await_count(
        lambda: sum('"group": 0' in d for d in _dials),
        1,
        what="group-0 post-mortem dials",
        cadence=Cadence(),
        gone=lambda: None if g2.alive else f"parent exited {g2.exit_code}",
    )
    g0_dials = [d for d in _dials if '"group": 0' in d]
    assert g0_dials, "group-0 death produced no typed post-mortem dial"
    assert any(
        '"function": "segfault"' in d
        and "r-die-g0" in d
        and "fault_dump_tail" in d
        and "procsplit_endpoints.py" in d
        for d in g0_dials
    )
    assert all("r-sleep-g1" not in d for d in g0_dials), (
        "group 0 was attributed to group 1's live request"
    )
    assert g1_marker.exists() and "r-sleep-g1" in g1_marker.read_text(), (
        "reaping group 0 destroyed group 1's live marker"
    )

    assert g2.pc._slots[1].proc is not None
    assert g2.pc._slots[1].proc.pid == slot1_pid_before, "sibling group was disturbed"
    conn.send(cancel_job=pb.CancelJob(request_id="r-sleep-g1", attempt=1))
    slept = conn.wait_for(is_result_for("r-sleep-g1"), timeout=30.0)
    assert slept.job_result.status == pb.JOB_STATUS_CANCELED
    conn.send(run_job=pb.RunJob(
        request_id="r-g1-after", attempt=1, function_name="whoami",
        input_payload=_payload(), compute=pb.ResolvedCompute(gpu_index=1)))
    r = conn.wait_for(is_result_for("r-g1-after"), timeout=60.0)
    assert r.job_result.status == pb.JOB_STATUS_OK
    assert b"g=1" in r.job_result.inline

    assert g2.alive and g2.exit_code is None
