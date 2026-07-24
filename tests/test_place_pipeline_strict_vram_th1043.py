"""th#1043: place_pipeline's reactive CUDA-OOM ladder must honor
Resources(strict_vram=True) instead of silently walking into a CPU-touching
offload rung. Plan-time already refuses a predicted-offload function under
strict_vram (serve_fit.plan_serve), but a load-time OOM the plan didn't
predict — e.g. a deploy-time binding that no longer matches the author's
vram_gb sizing — used to reach the offload ladder anyway, unconditionally.
"""

from __future__ import annotations

import pytest
import torch

from gen_worker.models import memory


class _FakePipeline:
    pass


def test_strict_vram_oom_refuses_instead_of_offloading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(memory, "select_auto_mode", lambda **_: "off")

    calls: list[str] = []

    def fake_apply(pipeline, *, mode, logger=None):
        calls.append(mode)
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(memory, "apply_low_vram_config", fake_apply)

    with pytest.raises(RuntimeError, match="strict_vram=True"):
        memory.place_pipeline(_FakePipeline(), mode="auto", ref="test/strict", strict_vram=True)

    # Refused on the FIRST OOM — never tried a CPU-touching rung.
    assert calls == ["off"]


def test_non_strict_vram_oom_still_demotes_to_offload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(memory, "select_auto_mode", lambda **_: "off")
    monkeypatch.setattr(memory, "_move_pipeline_to_cpu", lambda *_: None)
    monkeypatch.setattr(memory, "repair_device_placement", lambda *_: [])
    monkeypatch.setattr(memory, "flush_memory", lambda: None)

    calls: list[str] = []

    def fake_apply(pipeline, *, mode, logger=None):
        calls.append(mode)
        if mode == "off":
            raise RuntimeError("CUDA out of memory")
        return {"mode": mode}

    monkeypatch.setattr(memory, "apply_low_vram_config", fake_apply)

    applied = memory.place_pipeline(_FakePipeline(), mode="auto", ref="test/lenient", strict_vram=False)

    # Unchanged existing behavior: demotes into the offload ladder.
    assert applied["mode"] == "model_offload"
    assert calls == ["off", "model_offload"]
