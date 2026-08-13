"""pgw#676: a native crash (SIGSEGV in a CUDA/C extension) must be NAMED and
must not crash-loop the pod.

The live shape: gen-worker 0.66.0 on RTX A4500 (sm_86) segfaulted
(``exit_code=139``) on every 28-step CFG-on ``generate`` — six times across
two pods — while 4-step ``generate-turbo`` completed on the same workers.
The hub saw ``phase=worker_process_exit exit_code=139`` and NOTHING else; the
process restarted in the pod, took the same shape, and died again until
th#878's wedge terminate killed the pod (~31 min of billing), with every
request burned 5 attempts deep first.

Three closures, tested here with real forks and a real native fault:

  * the dying process's faulthandler dump file gives exit 139 Python frames;
  * the in-flight marker names WHAT was executing (function, kind, request);
  * the per-pod crash registry makes the NEXT boot's gate refuse a function
    with ``NATIVE_CRASH_REFUSE_STREAK`` signal deaths — siblings keep
    serving, the refusal is loud and typed (degrade-never-die across
    process death; the pgw#673/pgw#672 posture extended below Python).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List

import msgspec
import pytest

from gen_worker.api.binding import Hub
from gen_worker.api.decorators import Resources
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec
from gen_worker import postmortem

# ---------------------------------------------------------------------------
# Real fork + real native fault: the supervisor names the death.
# ---------------------------------------------------------------------------

_SEGV_SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from gen_worker.supervisor import supervise

    supervise(Path(sys.argv[1]))
    # only the child gets here
    from gen_worker import postmortem
    postmortem.enable_fault_dump()
    postmortem.note_inflight("request", "generate", request_id="req-676")
    import ctypes
    ctypes.string_at(0)  # real native fault: NULL deref -> SIGSEGV
    """
)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
def test_segv_death_is_attributed_and_carries_frames(tmp_path: Path) -> None:
    script = tmp_path / "boot.py"
    script.write_text(_SEGV_SCRIPT)
    record = tmp_path / "record.json"
    sink = tmp_path / "postmortem.txt"
    env = dict(os.environ)
    env["GEN_WORKER_POSTMORTEM_FILE"] = str(sink)
    env["GEN_WORKER_BOOT_RECORD"] = str(record)  # siblings land in tmp_path
    env.pop("GEN_WORKER_SUPERVISED", None)
    env.pop("ORCHESTRATOR_PUBLIC_ADDR", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [sys.executable, str(script), str(record)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 139, proc.stderr
    detail = sink.read_text()
    assert "KILLED BY SIGNAL SIGSEGV" in detail
    # the in-flight marker attributes the death to the executing function
    assert '"function": "generate"' in detail
    assert "req-676" in detail
    # the faulthandler dump gives exit 139 actual Python frames
    assert "fault_dump_tail" in detail
    assert "boot.py" in detail  # the dying frame is visible
    # the crash registry on the "pod" fs recorded the streak
    streaks = json.loads((tmp_path / "gen-worker-crash-streaks.json").read_text())
    assert streaks["generate"]["count"] == 1
    assert streaks["generate"]["last_signal"] == "SIGSEGV"
    # marker consumed: the next death cannot re-attribute this one
    assert not (tmp_path / "gen-worker-inflight.json").exists()


# ---------------------------------------------------------------------------
# The gate: streak >= NATIVE_CRASH_REFUSE_STREAK refuses the function,
# siblings keep serving.
# ---------------------------------------------------------------------------


class _In(msgspec.Struct):
    prompt: str = ""


class _Fake:
    def setup(self, pipeline) -> None:  # pragma: no cover
        self.pipeline = pipeline

    def generate(self, ctx, payload: _In) -> dict:  # pragma: no cover
        return {}

    def generate_turbo(self, ctx, payload: _In) -> dict:  # pragma: no cover
        return {}


def _specs() -> List[EndpointSpec]:
    return [
        EndpointSpec(
            name="generate", method=_Fake.generate, kind="inference",
            payload_type=_In, output_mode="single", cls=_Fake,
            models={"pipeline": Hub("acme/sdxl")},
            resources=Resources(gpu=True),
        ),
        EndpointSpec(
            name="generate-turbo", method=_Fake.generate_turbo,
            kind="inference", payload_type=_In, output_mode="single",
            cls=_Fake, models={"pipeline": Hub("acme/sdxl")},
            resources=Resources(gpu=True),
        ),
    ]


def _executor() -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:  # pragma: no cover
        pass

    return Executor(_specs(), _send)


_GPU = {"gpu_total_mem": 20 * 1024**3, "gpu_free_mem": 20 * 1024**3,
        "gpu_sm": "86", "installed_libs": []}


def test_crash_streak_gates_only_the_crashing_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "streaks.json"
    monkeypatch.setattr(postmortem, "CRASH_REGISTRY_PATH", registry)

    ex = _executor()
    # One crash = one free retry; nothing is gated yet.
    postmortem.record_native_crash(
        "generate", kind="request", signal_name="SIGSEGV")
    ex.gate_functions(_GPU)
    assert "generate" not in ex.unavailable

    # The second signal death trips the refusal — for that function only.
    postmortem.record_native_crash(
        "generate", kind="request", signal_name="SIGSEGV")
    ex.gate_functions(_GPU)
    code, detail, axes = ex.unavailable["generate"]
    assert code == "native_crash_streak"
    assert "SIGSEGV" in detail and "siblings keep serving" in detail
    assert axes["streak"] == "2"
    assert "generate-turbo" not in ex.unavailable, (
        "the turbo sibling completed on the same A4500 workers live — it "
        "must keep serving"
    )
    # Idempotent re-gate: the mark is gate-owned, re-derived every pass.
    ex.gate_functions(_GPU)
    assert ex.unavailable["generate"][0] == "native_crash_streak"


# ---------------------------------------------------------------------------
# Marker mechanics: overlapping executions, boot/exit hygiene.
# ---------------------------------------------------------------------------


def test_inflight_markers_stack_and_clear_by_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "inflight.json"
    monkeypatch.setattr(postmortem, "INFLIGHT_PATH", path)
    postmortem.clear_inflight()  # process-global hygiene for the test rig

    t_req = postmortem.note_inflight("request", "generate", request_id="r1")
    t_warm = postmortem.note_inflight("warmup", "generate-turbo")
    active = json.loads(path.read_text())["active"]
    assert {r["function"] for r in active} == {"generate", "generate-turbo"}

    postmortem.clear_inflight(t_warm)
    active = json.loads(path.read_text())["active"]
    assert [r["function"] for r in active] == ["generate"]

    postmortem.clear_inflight(t_req)
    assert not path.exists()

    # take_inflight consumes whatever a dead process left
    postmortem.note_inflight("request", "generate", request_id="r2")
    left = postmortem.take_inflight(path)
    assert [r["request_id"] for r in left] == ["r2"]
    assert not path.exists()
    postmortem.clear_inflight()


def test_streak_counts_distinct_requests_not_attempts_of_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#763 stage 4, measured on a live pod: the hub's blame ladder re-ran
    ONE deterministically fatal payload on the same pod, each attempt killed a
    child, the streak hit the refuse threshold, and a healthy pod was condemned
    `worker_native_crash_loop` — a verdict manufactured entirely by retries of a
    single request. The gate exists for a function that keeps killing this pod
    across DIFFERENT work (the A4500 case above), and that must stay armed.
    """
    registry = tmp_path / "streaks.json"
    monkeypatch.setattr(postmortem, "CRASH_REGISTRY_PATH", registry)
    ex = _executor()

    # The live shape: attempts 1, 2 and 3 of request r-oom, same pod.
    for _ in range(3):
        postmortem.record_native_crash(
            "generate", kind="request", signal_name="SIGKILL", request_id="r-oom")
    ex.gate_functions(_GPU)
    assert "generate" not in ex.unavailable, (
        "one request's retry ladder refused the function and would condemn the "
        "pod: " + json.dumps(postmortem.native_crash_streaks(registry))
    )
    assert postmortem.native_crash_streaks(registry)["generate"]["count"] == 1

    # The guard rail: a DIFFERENT request crashing the same function is the
    # real signal, and it still trips exactly as before.
    postmortem.record_native_crash(
        "generate", kind="request", signal_name="SIGSEGV", request_id="r-other")
    ex.gate_functions(_GPU)
    code, _detail, axes = ex.unavailable["generate"]
    assert code == "native_crash_streak" and axes["streak"] == "2"


def test_compile_deaths_still_count_every_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A background compile carries no request id, so distinct-request
    counting must not silently disarm it — every death still counts."""
    registry = tmp_path / "streaks.json"
    monkeypatch.setattr(postmortem, "CRASH_REGISTRY_PATH", registry)
    marker = postmortem.compile_marker("unet")
    for expected in (1, 2, 3):
        got = postmortem.record_native_crash(
            marker, kind=postmortem.COMPILE_KIND, signal_name="SIGSEGV")
        assert got == expected
