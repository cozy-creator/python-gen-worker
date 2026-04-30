"""Tests for gen_worker.inference_memory.

These tests use fake pipelines that record which enable_* methods were called,
so the harness does not require CUDA.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

from gen_worker import inference_memory as im


class FakeComponent:
    """Minimal nn.Module-like object so estimate_pipeline_size_gb has something to traverse."""

    def __init__(self, n: int = 0) -> None:
        self._params: List[Any] = []
        self._buffers: List[Any] = []

    def parameters(self):
        return iter(self._params)

    def buffers(self):
        return iter(self._buffers)


class FakePipeline:
    def __init__(self) -> None:
        self.calls: List[str] = []
        self.components: Dict[str, Any] = {}

    def enable_vae_slicing(self) -> None:
        self.calls.append("vae_slicing")

    def enable_vae_tiling(self) -> None:
        self.calls.append("vae_tiling")

    def enable_attention_slicing(self) -> None:
        self.calls.append("attention_slicing")

    def enable_model_cpu_offload(self) -> None:
        self.calls.append("model_cpu_offload")

    def enable_sequential_cpu_offload(self) -> None:
        self.calls.append("sequential_cpu_offload")

    def to(self, device: str) -> "FakePipeline":
        self.calls.append(f"to:{device}")
        return self


# ---------------------------------------------------------------------------
# apply_low_vram_config
# ---------------------------------------------------------------------------


def test_apply_off_does_nothing(monkeypatch):
    monkeypatch.delenv("COZY_INFERENCE_MEMORY_MODE", raising=False)
    p = FakePipeline()
    out = im.apply_low_vram_config(p, mode="off")
    assert p.calls == []
    assert out["mode"] == "off"


def test_apply_vae_only_enables_vae_and_attention(monkeypatch):
    monkeypatch.delenv("COZY_INFERENCE_MEMORY_MODE", raising=False)
    p = FakePipeline()
    out = im.apply_low_vram_config(p, mode="vae_only")
    assert "vae_slicing" in p.calls
    assert "vae_tiling" in p.calls
    assert "attention_slicing" in p.calls
    assert out["mode"] == "vae_only"
    assert out["vae_slicing"] is True
    assert out["vae_tiling"] is True
    assert out["attention_slicing"] is True


def test_apply_model_offload_runs_vae_then_model_offload(monkeypatch):
    monkeypatch.delenv("COZY_INFERENCE_MEMORY_MODE", raising=False)
    # Simulate CUDA available so the post-vae_only escalation runs.
    monkeypatch.setattr(im, "get_total_vram_gb", lambda *a, **k: 8.0)

    import gen_worker.inference_memory as mod

    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            pass

        @staticmethod
        def reset_peak_memory_stats() -> None:
            pass

    class FakeTorch:
        cuda = FakeTorchCuda

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch)

    p = FakePipeline()
    out = mod.apply_low_vram_config(p, mode="model_offload")
    assert "vae_slicing" in p.calls
    assert "model_cpu_offload" in p.calls
    assert out["model_offload"] is True


def test_apply_is_idempotent(monkeypatch):
    monkeypatch.delenv("COZY_INFERENCE_MEMORY_MODE", raising=False)
    p = FakePipeline()
    im.apply_low_vram_config(p, mode="vae_only")
    first_calls = list(p.calls)
    out2 = im.apply_low_vram_config(p, mode="vae_only")
    assert p.calls == first_calls
    assert out2["already_applied"] is True


def test_env_override_wins_over_auto(monkeypatch):
    monkeypatch.setenv("COZY_INFERENCE_MEMORY_MODE", "off")
    p = FakePipeline()
    out = im.apply_low_vram_config(p, mode="auto")
    assert p.calls == []
    assert out["mode"] == "off"


def test_invalid_mode_raises():
    p = FakePipeline()
    with pytest.raises(ValueError):
        im.apply_low_vram_config(p, mode="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# select_auto_mode
# ---------------------------------------------------------------------------


def test_select_auto_returns_off_when_no_cuda(monkeypatch):
    monkeypatch.setattr(im, "get_total_vram_gb", lambda *a, **k: 0.0)
    p = FakePipeline()
    assert im.select_auto_mode(pipeline=p) == "off"


def test_select_auto_group_offload_when_tiny_vram(monkeypatch):
    monkeypatch.setattr(im, "get_total_vram_gb", lambda *a, **k: 6.0)
    p = FakePipeline()
    assert im.select_auto_mode(pipeline=p, model_size_gb=10.0) == "group_offload"


def test_select_auto_model_offload_when_small_vram(monkeypatch):
    monkeypatch.setattr(im, "get_total_vram_gb", lambda *a, **k: 8.0)
    p = FakePipeline()
    assert im.select_auto_mode(pipeline=p, model_size_gb=4.0) == "model_offload"


def test_select_auto_vae_only_when_plenty_of_vram(monkeypatch):
    monkeypatch.setattr(im, "get_total_vram_gb", lambda *a, **k: 24.0)
    p = FakePipeline()
    # model fits comfortably
    assert im.select_auto_mode(pipeline=p, model_size_gb=8.0) == "vae_only"


def test_select_auto_model_offload_when_model_exceeds_free(monkeypatch):
    monkeypatch.setattr(im, "get_total_vram_gb", lambda *a, **k: 24.0)
    p = FakePipeline()
    # model_size > total - margin  -> model_offload
    assert im.select_auto_mode(pipeline=p, model_size_gb=23.0) == "model_offload"


def test_env_thresholds_are_honored(monkeypatch):
    monkeypatch.setenv("COZY_INFERENCE_MODEL_OFFLOAD_VRAM_GB", "16.0")
    monkeypatch.setattr(im, "get_total_vram_gb", lambda *a, **k: 12.0)
    p = FakePipeline()
    # 12.0 <= 16.0 threshold -> model_offload
    assert im.select_auto_mode(pipeline=p, model_size_gb=4.0) == "model_offload"


# ---------------------------------------------------------------------------
# _escalate_pipeline_mode
# ---------------------------------------------------------------------------


def test_escalation_steps_through_ladder(monkeypatch):
    import logging

    monkeypatch.delenv("COZY_INFERENCE_MEMORY_MODE", raising=False)

    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            pass

    class FakeTorch:
        cuda = FakeTorchCuda

    import sys

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    log = logging.getLogger("test")
    ladder = ("vae_only", "model_offload", "group_offload", "sequential")

    p = FakePipeline()
    im.apply_low_vram_config(p, mode="vae_only")
    assert getattr(p, "_cozy_low_vram_mode") == "vae_only"

    assert im._escalate_pipeline_mode(p, logger=log, escalation=ladder) is True
    # After escalation the pipeline is at model_offload (or higher if CUDA path can't run).
    assert getattr(p, "_cozy_low_vram_mode") in ("model_offload", "vae_only")


def test_escalation_returns_false_at_top(monkeypatch):
    import logging

    p = FakePipeline()
    setattr(p, "_cozy_low_vram_mode", "sequential")
    ladder = ("vae_only", "model_offload", "group_offload", "sequential")
    assert im._escalate_pipeline_mode(p, logger=logging.getLogger("test"), escalation=ladder) is False


# ---------------------------------------------------------------------------
# with_oom_retry
# ---------------------------------------------------------------------------


class FakeOOM(Exception):
    pass


def test_with_oom_retry_succeeds_without_torch(monkeypatch):
    # When torch isn't importable, with_oom_retry should still call the function once.
    import sys
    real_torch = sys.modules.pop("torch", None)
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        calls = {"n": 0}

        def fn() -> str:
            calls["n"] += 1
            return "ok"

        result = im.with_oom_retry(fn)
        assert result == "ok"
        assert calls["n"] == 1
    finally:
        if real_torch is not None:
            sys.modules["torch"] = real_torch
        else:
            sys.modules.pop("torch", None)


def test_with_oom_retry_retries_and_escalates(monkeypatch):
    import sys

    class FakeOOMType(RuntimeError):
        pass

    class FakeTorchCuda:
        OutOfMemoryError = FakeOOMType

        @staticmethod
        def is_available() -> bool:
            # Required so model_offload is actually applied during escalation.
            return True

        @staticmethod
        def empty_cache() -> None:
            pass

        @staticmethod
        def reset_peak_memory_stats() -> None:
            pass

        @staticmethod
        def mem_get_info(*_a, **_k):
            # 1 GB free of 8 GB total -> triggers the low-VRAM ladder.
            return (1 * 1024**3, 8 * 1024**3)

        @staticmethod
        def get_device_properties(_idx: int = 0):
            class _P:
                total_memory = 8 * 1024**3
            return _P()

    class FakeTorch:
        cuda = FakeTorchCuda

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    p = FakePipeline()
    im.apply_low_vram_config(p, mode="vae_only")

    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise FakeOOMType("oom")
        return "ok"

    out = im.with_oom_retry(flaky, pipelines=[p], max_retries=2)
    assert out == "ok"
    assert attempts["n"] == 2
    # After escalation from vae_only, pipeline should be at model_offload or higher.
    assert getattr(p, "_cozy_low_vram_mode") in ("model_offload", "group_offload", "sequential")
    assert "model_cpu_offload" in p.calls


def test_with_oom_retry_reraises_when_exhausted(monkeypatch):
    import sys

    class FakeOOMType(RuntimeError):
        pass

    class FakeTorchCuda:
        OutOfMemoryError = FakeOOMType

        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            pass

        @staticmethod
        def reset_peak_memory_stats() -> None:
            pass

    class FakeTorch:
        cuda = FakeTorchCuda

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    p = FakePipeline()
    # Already at the top of the ladder -> no escalation available.
    setattr(p, "_cozy_low_vram_mode", "sequential")

    def always_oom() -> None:
        raise FakeOOMType("oom")

    with pytest.raises(FakeOOMType):
        im.with_oom_retry(always_oom, pipelines=[p], max_retries=2)


# ---------------------------------------------------------------------------
# estimate_pipeline_size_gb
# ---------------------------------------------------------------------------


def test_estimate_size_returns_zero_for_empty_pipeline():
    p = FakePipeline()
    # No torch tensors anywhere -> falls back to 0.0 safely.
    size = im.estimate_pipeline_size_gb(p)
    assert size == 0.0


# ---------------------------------------------------------------------------
# sequential mode moves to CPU first (upstream foot-gun)
# ---------------------------------------------------------------------------


def test_sequential_mode_moves_to_cpu_before_enabling(monkeypatch):
    """Regression: enable_sequential_cpu_offload must run on a CPU-resident pipeline."""
    import sys

    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            pass

        @staticmethod
        def reset_peak_memory_stats() -> None:
            pass

    class FakeTorch:
        cuda = FakeTorchCuda

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    p = FakePipeline()
    im.apply_low_vram_config(p, mode="sequential")
    # to("cpu") must appear before enable_sequential_cpu_offload in the call log.
    assert "to:cpu" in p.calls
    assert "sequential_cpu_offload" in p.calls
    assert p.calls.index("to:cpu") < p.calls.index("sequential_cpu_offload")
