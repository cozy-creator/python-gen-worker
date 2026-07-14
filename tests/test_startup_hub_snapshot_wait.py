"""ie#455: a tensorhub-bound function whose snapshot never arrived was dropped
from the advertised set in SILENCE — available_functions=[] with no logged
reason (z-image fns=[] on RunPod, 2026-07-10).

startup() must (a) keep such functions in loading_functions (they are waiting
on the hub, not broken), (b) never call ensure_setup for them, and (c) LOG
which functions wait on which refs so a pod log names the deadlock instead of
saying nothing.
"""

from __future__ import annotations

import asyncio
import logging

import msgspec

from gen_worker.api.binding import HF, Hub
from gen_worker.config.settings import Settings
from gen_worker.executor import Executor
from gen_worker.lifecycle import Lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _In(msgspec.Struct):
    x: str


class _Out(msgspec.Struct):
    y: str


def _spec(name: str, binding) -> EndpointSpec:
    class Endpoint:
        def setup(self, model: str) -> None:  # pragma: no cover
            raise AssertionError("setup must not run for hub-waiting specs")

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out(y=payload.x)

    return EndpointSpec(
        name=name, method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="run", models={"model": binding},
    )


async def _noop_send(msg: pb.WorkerMessage) -> None:
    pass


def test_startup_logs_functions_waiting_on_hub_snapshots(caplog, monkeypatch) -> None:
    spec = _spec("generate", Hub("tensorhub/z-image"))
    ex = Executor([spec], _noop_send)
    lc = Lifecycle(Settings(orchestrator_public_addr="localhost:1"), ex)
    # No GPU on the test host: keep needs_gpu functions out of `unavailable`
    # so the hub-wait branch (not the cuda gate) is what's under test.
    lc.hardware = {"gpu_count": 1, "gpu_total_mem": 32 * 1024**3,
                   "gpu_free_mem": 30 * 1024**3, "gpu_sm": "90", "installed_libs": []}

    with caplog.at_level(logging.WARNING, logger="gen_worker.lifecycle"):
        asyncio.run(lc.startup())

    # (a) still advertised as loading, not silently dropped or unavailable
    assert ex.available_functions() == []
    assert ex.loading_functions() == ["generate"]
    assert "generate" not in ex.unavailable

    # (c) the wait is LOGGED, naming the function and the hub ref
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "generate" in m and "tensorhub/z-image" in m and "DesiredResidency" in m
        for m in msgs
    ), msgs


def test_startup_does_not_log_when_nothing_waits_on_hub(caplog, tmp_path, monkeypatch) -> None:
    # A non-tensorhub binding self-prefetches; stub the fetch and confirm the
    # hub-wait warning does NOT fire.
    spec = _spec("run-hf", HF("acme/model"))
    ex = Executor([spec], _noop_send)

    async def _fake_ensure_local(ref: str, **kwargs):  # noqa: ANN003
        return tmp_path

    monkeypatch.setattr(ex.store, "ensure_local", _fake_ensure_local)
    monkeypatch.setattr(ex.store, "local_path", lambda ref: tmp_path)

    async def _fake_setup(spec, snapshots=None):  # noqa: ANN001
        return None

    monkeypatch.setattr(ex, "ensure_setup", _fake_setup)

    lc = Lifecycle(Settings(orchestrator_public_addr="localhost:1"), ex)
    lc.hardware = {"gpu_count": 1, "gpu_total_mem": 32 * 1024**3,
                   "gpu_free_mem": 30 * 1024**3, "gpu_sm": "90", "installed_libs": []}

    with caplog.at_level(logging.WARNING, logger="gen_worker.lifecycle"):
        asyncio.run(lc.startup())

    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("waiting on hub-supplied snapshots" in m for m in msgs), msgs
