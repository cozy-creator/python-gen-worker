"""pgw#1312: CPU offload is ALWAYS allowed, and every activation confesses.

Paul, 2026-08-17, deleting `GEN_WORKER_FORBID_CPU_OFFLOAD`: *"There is no
FORBID_CPU_OFFLOAD. Envs are only for configs + secrets, they are not
logic-gates. That is a logic gate. We ALWAYS allow CPU-offload, and encourage
it — but when it happens we warn LOUDLY so the error can be caught (we don't
want to serve degraded in production)."*

Two halves, and the second is the one that had a hole. pgw#929's veto arm is
gone, so the env is inert. In its place every route into a CPU-touching rung
emits ONE typed `serve_degrade` event (`phase=cpu_offload_engaged`) — and
before this issue only the OOM-triggered descent said anything off the pod at
all: a plan-time rung selected against free VRAM applied the same diffusers
hooks and logged `low_vram: model_offload applied` at INFO, which a
hub-spawned worker's operator never sees.

Seam: the REAL `apply_low_vram_config` / `place_pipeline` against a stub
pipeline that owns the offload entry points and nothing else. No torch
modules, no weights, no card — the code under test is the activation report,
not diffusers' hooks.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from gen_worker import activity as activity_mod
from gen_worker.models import memory
from gen_worker.pb import worker_scheduler_pb2 as pb

FORBID_ENV = "GEN_WORKER_FORBID_CPU_OFFLOAD"


class _Events:
    """The REAL activity sink the worker transport installs, drained after the
    placement — so these assertions read exactly the ActivityUpdates a hub
    would bank, not an in-process spy."""

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
    """A diffusers pipeline reduced to its offload surface."""

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
    """A card is present and host RAM is roomy — the two facts the applier
    reads about the machine, neither of which is the code under test."""
    monkeypatch.setattr(memory, "cuda_ready", lambda: True)
    monkeypatch.setattr(memory, "_should_auto_disk_offload", lambda: False)
    monkeypatch.setattr(memory, "flush_memory", lambda: None)
    monkeypatch.setattr(memory, "repair_device_placement", lambda *_a: [])


# ---------------------------------------------------------------------------
# 1. The veto is gone: the env this box exports no longer decides anything.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rung", ["model_offload", "group_offload", "sequential"])
def test_offload_proceeds_with_the_dead_env_set_and_confesses(
    monkeypatch: pytest.MonkeyPatch, carded: None, rung: str,
) -> None:
    """RED before pgw#1312: `CpuOffloadForbidden` at the model/sequential
    rungs. The env is now inert on every rung, and each activation banks one
    typed event."""
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


def test_the_env_name_is_gone_from_the_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """The literal, the reader and the refusal all die together — a dead env
    read that still exists is the pgw#929 C3 shape with the polarity flipped."""
    for gone in ("cpu_offload_forbidden", "_refuse_cpu_offload",
                 "CpuOffloadForbidden", "_FORBID_CPU_OFFLOAD_ENV"):
        assert not hasattr(memory, gone), f"{gone} survived the cut"
    src = memory.__file__
    with open(src, encoding="utf-8") as fh:
        assert FORBID_ENV not in fh.read(), f"{FORBID_ENV} still literal in {src}"


# ---------------------------------------------------------------------------
# 2. The confession is load-bearing, not decorative.
# ---------------------------------------------------------------------------


def test_disarming_the_emitter_leaves_the_offload_silent(
    monkeypatch: pytest.MonkeyPatch, carded: None,
) -> None:
    """Prove the instrument can go red. With the ONE emitter neutered the same
    activation still happens and the hub hears NOTHING — which is exactly the
    state this issue found the plan-time rung in, and exactly what the test
    above would fail on."""
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
    """The negative control. `vae_only` keeps every weight on the card, so a
    confession there would train operators to ignore the loud ones."""
    pipe = _StubPipeline()
    with _Events() as events:
        applied = memory.apply_low_vram_config(pipe, mode="vae_only")

    assert applied["mode"] == "vae_only"
    assert events.offload_engaged() == []


# ---------------------------------------------------------------------------
# 3. Works-always: the OOM rung still degrades rather than dying.
# ---------------------------------------------------------------------------


def test_cuda_oom_still_descends_and_serves(
    monkeypatch: pytest.MonkeyPatch, carded: None,
) -> None:
    """A CUDA OOM during a resident placement is a ladder transition, not a
    failure (gw#463) — unchanged by this issue, and now audible."""
    monkeypatch.setenv(FORBID_ENV, "1")
    pipe = _StubPipeline(oom_on_cuda=1)
    with _Events() as events:
        placed = memory.place_pipeline(pipe, mode="off")

    assert placed["mode"] == "model_offload", "the descent did not run"
    assert placed["oom_demotions"] == 1
    assert placed["requested_mode"] == "off"
    assert "model_offload" in pipe.calls
    assert len(events.offload_engaged()) == 1
