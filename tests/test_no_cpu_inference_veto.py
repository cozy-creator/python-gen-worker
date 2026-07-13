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


# --- gw#521: mechanism-level coverage — every offload ENTRY POINT refuses, ---
# --- not just the apply_low_vram_config policy layer.                      ---


def test_group_offload_mechanism_refuses_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker.models.memory import _apply_group_offload

    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "1")
    with pytest.raises(RuntimeError, match="FORBID_CPU_OFFLOAD"):
        _apply_group_offload(_DummyPipeline(), {}, offload_to_disk_path=None)


def test_block_window_offload_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    from gen_worker.models.loading import apply_block_window_offload

    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "1")
    with pytest.raises(RuntimeError, match="FORBID_CPU_OFFLOAD"):
        apply_block_window_offload(_DummyPipeline())


def test_demote_pipeline_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    from gen_worker.models.memory import demote_pipeline

    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "1")
    with pytest.raises(RuntimeError, match="FORBID_CPU_OFFLOAD"):
        demote_pipeline(_DummyPipeline())
