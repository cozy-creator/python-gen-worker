"""GEN_WORKER_FORBID_CPU_OFFLOAD=1 is the dev-box kill-switch: it raises at
actual pipeline PLACEMENT time when real weights are about to touch the CPU.

Post Paul's ruling (2026-07-10) it no longer affects SERVE PLANNING — a worker
never refuses to advertise a function just because it would need offload (it
runs degraded and reports FnDegraded). This last-resort guard stays only to
protect a directly-invoked local worker on this shared box from melting it
with real CPU-offloaded inference; the orchestrated path is covered by the
th#657 capability gate. Unset everywhere in production.
"""

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


class _GGUFPipeline:
    _cozy_gguf_quant = "q4_k_m"


def test_gguf_resident_fit_beats_model_offload_under_forbid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # cl#27: ladder picked the rung on its RESIDENT bound; auto placement must
    # not turn a fitting gguf pipe into a FORBID refusal via the 2GB margin.
    import gen_worker.models.memory as memory

    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "1")
    monkeypatch.setattr(memory, "get_available_vram_gb", lambda *a, **k: 7.3)
    result = apply_low_vram_config(_GGUFPipeline(), mode="auto", model_size_gb=6.4)
    assert result["mode"] == "vae_only"


def test_gguf_that_truly_does_not_fit_still_refuses_under_forbid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gen_worker.models.memory as memory

    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "1")
    monkeypatch.setattr(memory, "get_available_vram_gb", lambda *a, **k: 7.3)
    with pytest.raises(RuntimeError, match="FORBID_CPU_OFFLOAD"):
        apply_low_vram_config(_GGUFPipeline(), mode="auto", model_size_gb=7.2)


def test_gguf_explicit_model_offload_is_not_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gen_worker.models.memory as memory

    monkeypatch.delenv("GEN_WORKER_FORBID_CPU_OFFLOAD", raising=False)
    monkeypatch.setattr(memory, "get_available_vram_gb", lambda *a, **k: 7.3)
    result = apply_low_vram_config(_GGUFPipeline(), mode="model_offload", model_size_gb=1.0)
    assert result["mode"] == "model_offload"
