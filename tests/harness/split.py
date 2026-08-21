from __future__ import annotations

import os
import sys
import threading
from concurrent import futures
from pathlib import Path
from typing import List, Optional

import grpc
import msgspec
import pytest

from gen_worker import postmortem, worker_fatal
from gen_worker.config import load_settings
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc  # noqa: F401
from gen_worker.procsplit.parent import ParentControl
from harness.hub_double import FakeScheduler, measure_child_boot_cost_s

_BOOT_RECORD_NAME = "gen-worker-boot-record.json"

_INFLIGHT_NAME = "gen-worker-inflight.json"

_CRASH_REGISTRY_NAME = "gen-worker-crash-streaks.json"

_FAULT_DUMP_NAME = "gen-worker-fault-dump.txt"

def postmortem_dir(tmp: Path) -> Path:
    d = tmp / "postmortem"
    d.mkdir(parents=True, exist_ok=True)
    return d


TESTS_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = TESTS_DIR.parent / "src"

class _In(msgspec.Struct):
    text: str = ""

CHILD_MAIN = TESTS_DIR / "harness" / "procsplit_child_main.py"

def _payload(text: str = "") -> bytes:
    return msgspec.msgpack.encode(_In(text=text))

class SplitHarness:
    """One hub-double + one in-process ParentControl + real child subprocesses."""

    def __init__(
        self,
        tmp: Path,
        *,
        child_cmd: Optional[List[str]] = None,
        watchdog_budget_s: float = 60.0,
        # pgw#1630: the flatness FLOOR. Separate from the budget above, which is
        # now only the /proc sampling cadence. A harness that wants to observe a
        # kill has to say how long "flat" means HERE, because the production
        # default is a derived 120 s and the ladder needs four of them.
        # 0 = the production default, which is what most rows want: they are
        # asserting that nothing is killed.
        liveness_floor_s: float = 0.0,
        start_limit_burst: int = 3,
        start_limit_interval_s: float = 600.0,
        stop_timeout_s: float = 120.0,
        stop_flush_timeout_s: float = 30.0,
        beat_interval_s: float = 0.0,
        extra_child_env: Optional[dict] = None,
    ) -> None:
        self.scheduler = FakeScheduler()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
        pb_grpc.add_WorkerSchedulerServicer_to_server(self.scheduler, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id="split-parent",
            worker_jwt="",
        )
        child_env = {
            "PYTHONPATH": os.pathsep.join(
                [str(TESTS_DIR), str(SRC_DIR), os.environ.get("PYTHONPATH", "")]
            ),
            "TENSORHUB_CACHE_DIR": str(tmp / "cache"),
            "GEN_WORKER_CHILD_WATCHDOG_PING_S": "0.5",
            "GEN_WORKER_BOOT_RECORD": str(postmortem_dir(tmp) / _BOOT_RECORD_NAME),
        }
        child_env.update(extra_child_env or {})
        self.child_env = dict(child_env)
        self.pc = ParentControl(
            settings,
            child_cmd=child_cmd or [sys.executable, str(CHILD_MAIN)],
            child_env=child_env,
            socket_path=str(tmp / "ctl.sock"),
            respawn_backoff_base_s=0.1,
            respawn_backoff_cap_s=0.5,
            transport_backoff_base_s=0.05,
            transport_backoff_cap_s=0.2,
            watchdog_budget_s=watchdog_budget_s,
            liveness_floor_s=liveness_floor_s,
            start_limit_burst=start_limit_burst,
            start_limit_interval_s=start_limit_interval_s,
            stop_timeout_s=stop_timeout_s,
            stop_flush_timeout_s=stop_flush_timeout_s,
            beat_interval_s=beat_interval_s,
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

    def signal(self, signum: int) -> None:
        loop = self.pc._loop
        assert loop is not None
        loop.call_soon_threadsafe(self.pc._forward_signal, signum)

    def wait_exit(self, timeout: float) -> Optional[int]:
        self._thread.join(timeout)
        return self.exit_code

    def close(self) -> None:
        self.pc.stop()
        self._thread.join(20.0)
        self.server.stop(grace=0)

@pytest.fixture(autouse=True)
def isolated_postmortem(tmp_path, monkeypatch):

    d = postmortem_dir(tmp_path)
    monkeypatch.setattr(postmortem, "INFLIGHT_PATH", d / _INFLIGHT_NAME)
    monkeypatch.setattr(postmortem, "CRASH_REGISTRY_PATH", d / _CRASH_REGISTRY_NAME)
    monkeypatch.setattr(postmortem, "FAULT_DUMP_PATH", d / _FAULT_DUMP_NAME)
    return d

@pytest.fixture()
def captured_dials(monkeypatch):
    """Keep post-mortem dials in-process (no real HTTP) and observable."""
    dials: List[str] = []

    def _capture(settings, detail):
        dials.append(detail)
        return True

    monkeypatch.setattr(worker_fatal, "report_worker_detail", _capture)
    return dials

@pytest.fixture()
def captured_reports(monkeypatch):
    """Keep the parent's HardwareUnsuitable relay in-process and observable."""
    from gen_worker import hardware_report

    reports: List[object] = []

    def _capture(settings, report):
        reports.append(report)
        return True

    monkeypatch.setattr(hardware_report, "deliver_hardware_report", _capture)
    return reports
