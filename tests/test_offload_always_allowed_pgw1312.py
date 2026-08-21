from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from gen_worker import activity as activity_mod
from gen_worker.models import memory
from gen_worker.pb import worker_scheduler_pb2 as pb

FORBID_ENV = "GEN_WORKER_FORBID_CPU_OFFLOAD"


class _Events:

    def __init__(self) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self.loop = asyncio.new_event_loop()

    def __enter__(self) -> "_Events":
        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        activity_mod.bind_sink(_send, self.loop)
        return self

    def __exit__(self, *exc: object) -> None:
        self.loop.run_until_complete(asyncio.sleep(0.02))
        activity_mod.reset_for_tests()
        self.loop.close()

    def offload_engaged(self) -> List[pb.ActivityUpdate]:
        return [
            m.activity_update for m in self.sent
            if m.WhichOneof("msg") == "activity_update"
            and m.activity_update.kind == activity_mod.KIND_SERVE_DEGRADE
            and m.activity_update.phase == memory.OFFLOAD_ENGAGED_PHASE
        ]


class _StubPipeline:

    def __init__(self, *, oom_on_cuda: int = 0) -> None:
        self.calls: List[str] = []
        self.components: Dict[str, Any] = {}
        self._oom_left = oom_on_cuda

    def to(self, device: str) -> "_StubPipeline":
        self.calls.append(f"to:{device}")
        if device == "cuda" and self._oom_left > 0:
            self._oom_left -= 1
            raise RuntimeError("CUDA error: out of memory")
        return self

    def enable_model_cpu_offload(self, gpu_id: int = 0) -> None:
        self.calls.append("model_offload")

    def enable_sequential_cpu_offload(self, gpu_id: int = 0) -> None:
        self.calls.append("sequential")

    def enable_group_offload(self, **kwargs: Any) -> None:
        self.calls.append("group_offload")

    def enable_vae_slicing(self) -> None:
        self.calls.append("vae_slicing")

    def enable_vae_tiling(self) -> None:
        self.calls.append("vae_tiling")

    def enable_attention_slicing(self, *args: Any) -> None:
        self.calls.append("attention_slicing")


@pytest.fixture
def carded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A card is present and host RAM is roomy — the two facts the applier reads about the machine, neither of which is the code under test."""
    monkeypatch.setattr(memory, "cuda_ready", lambda: True)
    monkeypatch.setattr(memory, "_should_auto_disk_offload", lambda: False)
    monkeypatch.setattr(memory, "flush_memory", lambda: None)
    monkeypatch.setattr(memory, "repair_device_placement", lambda *_a: [])


@pytest.mark.parametrize("rung", ["model_offload", "group_offload", "sequential"])
def test_offload_proceeds_with_the_dead_env_set_and_confesses(
    monkeypatch: pytest.MonkeyPatch, carded: None, rung: str,
) -> None:
    monkeypatch.setenv(FORBID_ENV, "1")
    pipe = _StubPipeline()
    with _Events() as events:
        applied = memory.apply_low_vram_config(pipe, mode=rung)

    assert applied["mode"] == rung
    assert rung in pipe.calls, f"{rung} hooks were never attached: {pipe.calls}"
    assert memory.low_vram_mode(pipe) == rung

    engaged = events.offload_engaged()
    assert len(engaged) == 1, f"expected ONE confession, got {len(engaged)}"
    assert rung in engaged[0].detail
    assert "DEGRADED" in engaged[0].detail


def test_disarming_the_emitter_leaves_the_offload_silent(
    monkeypatch: pytest.MonkeyPatch, carded: None,
) -> None:
    """Prove the instrument can go red."""
    monkeypatch.setattr(memory, "_report_offload_engaged", lambda *a, **k: None)
    pipe = _StubPipeline()
    with _Events() as events:
        memory.apply_low_vram_config(pipe, mode="model_offload")

    assert "model_offload" in pipe.calls, "the offload itself must be unaffected"
    assert events.offload_engaged() == [], (
        "the emitter was disarmed yet an event arrived — some OTHER site is "
        "reporting offload, so this is no longer one home")


def test_a_resident_rung_confesses_nothing(
    monkeypatch: pytest.MonkeyPatch, carded: None,
) -> None:
    """The negative control."""
    pipe = _StubPipeline()
    with _Events() as events:
        applied = memory.apply_low_vram_config(pipe, mode="vae_only")

    assert applied["mode"] == "vae_only"
    assert events.offload_engaged() == []


def test_cuda_oom_still_descends_and_serves(
    monkeypatch: pytest.MonkeyPatch, carded: None,
) -> None:
    """A CUDA OOM during a resident placement is a ladder transition, not a failure (gw#463) — unchanged by this issue, and now audible."""
    monkeypatch.setenv(FORBID_ENV, "1")
    pipe = _StubPipeline(oom_on_cuda=1)
    with _Events() as events:
        placed = memory.place_pipeline(pipe, mode="off")

    assert placed["mode"] == "model_offload", "the descent did not run"
    assert placed["oom_demotions"] == 1
    assert placed["requested_mode"] == "off"
    assert "model_offload" in pipe.calls
    assert len(events.offload_engaged()) == 1
