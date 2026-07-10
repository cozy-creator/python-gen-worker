"""GEN_WORKER_FORBID_CPU_OFFLOAD=1 vetoes any placement that touches the CPU."""

from __future__ import annotations

import pytest

from gen_worker.models.memory import apply_low_vram_config


class _DummyPipeline:
    pass


def test_offload_modes_raise_when_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "1")
    for mode in ("model_offload", "group_offload", "sequential"):
        with pytest.raises(RuntimeError, match="FORBID_CPU_OFFLOAD"):
            apply_low_vram_config(_DummyPipeline(), mode=mode)


def test_gpu_resident_modes_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "1")
    result = apply_low_vram_config(_DummyPipeline(), mode="off")
    assert result["mode"] == "off"


def test_offload_allowed_without_veto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEN_WORKER_FORBID_CPU_OFFLOAD", raising=False)
    result = apply_low_vram_config(_DummyPipeline(), mode="model_offload")
    assert result["mode"] == "model_offload"
