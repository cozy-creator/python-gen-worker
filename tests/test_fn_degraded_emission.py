"""th#683 P3: structured degradation events (FnDegraded) reach the orchestrator.

A function that gates as serveable-but-degraded (fp8 storage / emergency nf4 /
offload / cpu) must be reported STRUCTURALLY over the worker stream — once per
connection, re-emitted on reconnect — so placement learns "this release
degraded on this card". With no orchestrator (cozy-local) nothing is emitted:
the honest-guidance advisory on the terminal is the surface there.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import List

import msgspec

from gen_worker.api.binding import HF
from gen_worker.api.decorators import Resources
from gen_worker.executor import Executor
from gen_worker.lifecycle import Lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _In(msgspec.Struct):
    x: str


class _Out(msgspec.Struct):
    y: str


def _spec(name: str, vram_gb: float) -> EndpointSpec:
    class Endpoint:
        def setup(self, model: str) -> None:  # pragma: no cover
            pass

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out(y=payload.x)

    return EndpointSpec(
        name=name, method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="run", models={"model": HF("acme/tiny")},
        resources=Resources(vram_gb=vram_gb),
    )


class _StubTransport:
    def __init__(self) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self.connected = True
        self.queue = SimpleNamespace(pending_result_keys=set())

    async def send(self, msg: pb.WorkerMessage) -> None:
        self.sent.append(msg)


async def _noop_send(msg) -> None:  # pragma: no cover
    pass


_SMALL_CARD = {  # 8 GB card, SM86 — big models degrade here
    "gpu_count": 1,
    "gpu_total_mem": 8 << 30,
    "gpu_free_mem": 8 << 30,
    "gpu_sm": "86",
}


def _settings() -> SimpleNamespace:
    return SimpleNamespace(worker_jwt="", worker_id="w-test", runpod_pod_id="")


def _degraded_msgs(sent: List[pb.WorkerMessage]) -> List[pb.FnDegraded]:
    return [m.fn_degraded for m in sent if m.WhichOneof("msg") == "fn_degraded"]


def test_degraded_function_emits_structured_event() -> None:
    async def _go() -> None:
        # 12 GB model on the 8 GB card -> runtime fp8-storage rung (degraded).
        ex = Executor([_spec("gen", 12.0), _spec("small", 4.0)], _noop_send)
        ex.gate_functions(_SMALL_CARD)
        lc = Lifecycle(_settings(), ex)
        tp = _StubTransport()
        lc.transport = tp

        await lc.on_hello_ack(pb.HelloAck())
        events = _degraded_msgs(tp.sent)
        assert [e.function_name for e in events] == ["gen"]
        e = events[0]
        assert e.wanted == "bf16"
        assert e.ran == "fp8_storage"
        assert e.est_latency_multiplier > 1.0
        assert e.recommended_vram_gb == 12.0
        assert "#fp8" in e.reason  # better-flavor hint rides the reason

        # Dedupe: further state deltas do not re-emit.
        await lc.maybe_send_state_delta()
        assert len(_degraded_msgs(tp.sent)) == 1

        # Reconnect (new HelloAck): per-connection state resets -> re-emit.
        await lc.on_disconnect()
        await lc.on_hello_ack(pb.HelloAck())
        assert len(_degraded_msgs(tp.sent)) == 2

    asyncio.run(_go())


def test_unavailable_functions_do_not_emit_degraded(monkeypatch) -> None:
    async def _go() -> None:
        monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "1")
        # 100 GB model: even 4-bit doesn't fit the 8 GB card -> offload-only,
        # which this box forbids -> unavailable, NOT degraded.
        ex = Executor([_spec("huge", 100.0)], _noop_send)
        ex.gate_functions(_SMALL_CARD)
        assert "huge" in ex.unavailable
        lc = Lifecycle(_settings(), ex)
        tp = _StubTransport()
        lc.transport = tp
        await lc.on_hello_ack(pb.HelloAck())
        assert _degraded_msgs(tp.sent) == []
        # The refusal went out on the FnUnavailable channel instead.
        assert any(m.WhichOneof("msg") == "fn_unavailable" for m in tp.sent)

    asyncio.run(_go())


def test_no_orchestrator_means_no_emit() -> None:
    async def _go() -> None:
        ex = Executor([_spec("gen", 12.0)], _noop_send)
        ex.gate_functions(_SMALL_CARD)
        lc = Lifecycle(_settings(), ex)
        lc.transport = None  # cozy-local: no orchestrator present
        await lc._emit_degraded()  # must be a clean no-op

    asyncio.run(_go())
