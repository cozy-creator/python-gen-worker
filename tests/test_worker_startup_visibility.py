from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from gen_worker.worker import Worker


def test_registration_watchdog_emits_timeout_and_sets_stop() -> None:
    w = Worker.__new__(Worker)
    w.worker_id = "w-1"
    w.scheduler_addr = "127.0.0.1:8080"
    w._process_started_monotonic = time.monotonic() - 0.5
    w._registered_event = threading.Event()
    w._running = True
    w._stop_event = threading.Event()
    w._startup_timeout_triggered = False

    startup_events: list[tuple[str, dict[str, Any]]] = []
    worker_events: list[tuple[str, dict[str, Any]]] = []
    w._emit_startup_phase = lambda phase, **kw: startup_events.append((phase, kw))  # type: ignore[method-assign]
    w._emit_worker_event_bytes = (  # type: ignore[method-assign]
        lambda run_id, event_type, payload_json: worker_events.append(
            (event_type, json.loads(payload_json.decode("utf-8")))
        )
    )
    w._close_connection = lambda: None  # type: ignore[method-assign]

    w._registration_watchdog_loop(timeout_s=0.02)

    assert w._startup_timeout_triggered is True
    assert w._stop_event.is_set() is True
    assert any(name == "startup_timeout_unregistered" for name, _ in startup_events)
    assert any(name == "worker.startup_timeout_unregistered" for name, _ in worker_events)


def test_task_phase_watchdog_emits_stuck_event() -> None:
    w = Worker.__new__(Worker)
    seen: list[tuple[str, dict[str, Any]]] = []
    w._emit_worker_event_bytes = (  # type: ignore[method-assign]
        lambda run_id, event_type, payload_json: seen.append((event_type, json.loads(payload_json.decode("utf-8"))))
    )

    timer = w._start_task_phase_watchdog(
        run_id="run-1",
        phase="inference",
        warn_after_s=0.02,
        payload={"function_name": "generate"},
    )
    time.sleep(0.06)
    if timer is not None:
        timer.cancel()

    assert any(name == "task.inference.stuck" for name, _ in seen)
    ev_payload = next(p for name, p in seen if name == "task.inference.stuck")
    assert ev_payload["function_name"] == "generate"
    assert ev_payload["elapsed_ms"] >= 20


def test_emit_worker_fatal_includes_traceback_metadata() -> None:
    w = Worker.__new__(Worker)
    w.worker_id = "w-2"
    w.scheduler_addr = "scheduler:8080"
    w._process_started_monotonic = time.monotonic() - 1.0
    seen: list[tuple[str, dict[str, Any]]] = []
    w._emit_worker_event_bytes = (  # type: ignore[method-assign]
        lambda run_id, event_type, payload_json: seen.append((event_type, json.loads(payload_json.decode("utf-8"))))
    )

    try:
        raise RuntimeError("boom")
    except Exception as exc:
        w._emit_worker_fatal("startup", exc, exit_code=7)

    assert seen
    event_type, payload = seen[-1]
    assert event_type == "worker.fatal"
    assert payload["phase"] == "startup"
    assert payload["exception_class"] == "RuntimeError"
    assert "boom" in payload["exception_message"]
    assert "RuntimeError" in payload["traceback"]
    assert payload["exit_code"] == 7


def test_run_raises_when_registration_timeout_reached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod_dir = tmp_path / "mod"
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "tiny_mod.py").write_text(
        """
from __future__ import annotations
import msgspec
from gen_worker.decorators import worker_function
from gen_worker.worker import ActionContext

class Input(msgspec.Struct):
    name: str

class Output(msgspec.Struct):
    ok: bool

@worker_function()
def tiny(ctx: ActionContext, payload: Input) -> Output:
    return Output(ok=True)
""".lstrip(),
        encoding="utf-8",
    )
    sys.path.insert(0, str(mod_dir))

    w = Worker(
        scheduler_addr="127.0.0.1:1",
        user_module_names=["tiny_mod"],
        worker_jwt="jwt-test",
        reconnect_delay=0,
        max_reconnect_attempts=0,
    )
    monkeypatch.setattr(w, "connect", types.MethodType(lambda self: False, w))
    w._register_timeout_s = 1
    w._reconnect_delay_base = 0
    w._reconnect_delay_max = 0
    w._reconnect_jitter = 0

    with pytest.raises(RuntimeError, match="startup_timeout_unregistered"):
        w.run()

