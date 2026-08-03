"""pgw#783: the N-child RUNTIME proof — a real parent spawning TWO real compute
children (one per execution group) and routing dispatches between them.

The unit rows in ``test_group_processes_pgw783`` prove the parent's routing and
fan-in LOGIC on a real ParentControl; this proves the whole thing at RUNTIME:
two child subprocesses, one unix server per group, the per-group env delta
applied at spawn, a dispatch to each group's rank-0 device served by THAT group,
and one child's death attributed to ITS request while the sibling keeps serving.

Real everything on the worker side: a real ParentControl (real Transport,
real supervision, two real ``_ChildSlot``s) speaking real gRPC to the hub-double,
spawning two real compute children. The hub is the only double. The toy child is
CPU-only, so it boots identically with or without CUDA visibility.
"""

from __future__ import annotations

import os
import sys
import threading
import time
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

from harness.hub_double import FakeScheduler, is_accept_for, is_ready, is_result_for

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
CHILD_MAIN = TESTS_DIR / "harness" / "procsplit_child_main.py"
BOOT_TIMEOUT_S = 180.0  # two children import the worker; generous but real


class _In(msgspec.Struct):
    text: str = ""


def _payload(text: str = "") -> bytes:
    return msgspec.msgpack.encode(_In(text=text))


def _postmortem_dir(tmp: Path) -> Path:
    d = tmp / "postmortem"
    d.mkdir(parents=True, exist_ok=True)
    return d


class G2Harness:
    """One hub-double + one ParentControl with a 2-group topology + two real
    child subprocesses."""

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
            # TWO execution groups, one device each — the 4.00x shape at width 2.
            topology=ExecutionTopology(gpu_count=2, gpus_per_execution_group=1),
            respawn_backoff_base_s=0.1,
            respawn_backoff_cap_s=0.5,
            transport_backoff_base_s=0.05,
            transport_backoff_cap_s=0.2,
            watchdog_budget_s=60.0,
        )
        self.exit_code: Optional[int] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
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
    """The whole worker is READY only once BOTH groups are (the fan-in merges
    phase to the least-ready), and a dispatch to each group's rank-0 device is
    served by THAT group's child — proven by the ordinal it reports."""
    conn = g2.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
    # READY requires the parent to have merged two children's state to READY.
    conn.wait_for(is_ready, timeout=BOOT_TIMEOUT_S)

    # Two real children exist, one per group.
    assert g2.pc.execution_groups == 2
    assert all(slot.proc is not None for slot in g2.pc._slots)
    assert g2.pc._slots[0].socket_path != g2.pc._slots[1].socket_path

    # A dispatch to group 0's rank-0 device (gpu_index 0) is served by g0.
    conn.send(run_job=pb.RunJob(
        request_id="r-g0", attempt=1, function_name="whoami",
        input_payload=_payload(), compute=pb.ResolvedCompute(gpu_index=0)))
    r0 = conn.wait_for(is_result_for("r-g0"), timeout=60.0)
    assert r0.job_result.status == pb.JOB_STATUS_OK
    assert b"g=0" in r0.job_result.inline
    assert b"cvd=0" in r0.job_result.inline   # scoped to card 0
    assert b"sib=2" in r0.job_result.inline    # knows it shares the pod

    # A dispatch to group 1's rank-0 device (gpu_index 1) is served by g1.
    conn.send(run_job=pb.RunJob(
        request_id="r-g1", attempt=1, function_name="whoami",
        input_payload=_payload(), compute=pb.ResolvedCompute(gpu_index=1)))
    r1 = conn.wait_for(is_result_for("r-g1"), timeout=60.0)
    assert r1.job_result.status == pb.JOB_STATUS_OK
    assert b"g=1" in r1.job_result.inline
    assert b"cvd=1" in r1.job_result.inline   # scoped to card 1

    assert g2.alive and g2.exit_code is None


def test_one_childs_death_is_attributed_to_its_request_siblings_keep_serving(
    g2, _dials, tmp_path,
):
    """A group where one of the children dies is not a dead group: only the dead
    child's in-flight job is attributed, only its group respawns, and the sibling
    group serves throughout — the pgw#783 failure model on real processes."""
    conn = g2.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
    conn.wait_for(is_ready, timeout=BOOT_TIMEOUT_S)

    slot1_pid_before = g2.pc._slots[1].proc.pid

    # Keep a real request open in g1 while g0 dies.  Its marker is the exact
    # cross-process evidence pgw#938 found the shared path could unlink.
    conn.send(run_job=pb.RunJob(
        request_id="r-sleep-g1", attempt=1, function_name="sleepy",
        input_payload=_payload(), compute=pb.ResolvedCompute(gpu_index=1)))
    conn.wait_for(is_accept_for("r-sleep-g1"), timeout=30.0)
    g1_marker = _postmortem_dir(tmp_path) / "g1" / "gen-worker-inflight.json"
    deadline = time.monotonic() + 10.0
    while not g1_marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert g1_marker.exists(), "g1 never published its per-group in-flight marker"

    # Kill group 0 below Python while group 1's request and fault-dump file are
    # live.  The parent must consume only g0's one-writer evidence.
    conn.send(run_job=pb.RunJob(
        request_id="r-die-g0", attempt=1, function_name="segfault",
        input_payload=_payload(), compute=pb.ResolvedCompute(gpu_index=0)))
    died = conn.wait_for(is_result_for("r-die-g0"), timeout=60.0)
    assert died.job_result.status == pb.JOB_STATUS_FATAL
    assert DEATH_LABEL in died.job_result.safe_message
    assert "function=segfault" in died.job_result.safe_message

    deadline = time.monotonic() + 10.0
    while not any('"group": 0' in d for d in _dials) and time.monotonic() < deadline:
        time.sleep(0.02)
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

    # The sibling group (g1) never died — same process — and still serves.
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
