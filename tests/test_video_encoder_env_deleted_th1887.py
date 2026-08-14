"""th#1887: GEN_WORKER_VIDEO_ENCODER is deleted — exporting it must be inert.

Paul: *"get rid of these envs please, unless you really need them. I hate envs
like this."*

The switch had three values and none of them could reach an encoder the probe
would not: ``x264`` only SKIPPED the probe, and ``nvenc`` already fell back to
x264 when the probe failed. So the probe was always the real decision, and the
only thing the env could do was make a pod encode on CPU while its NVENC ASIC
sat idle — without reporting the gap.
"""

from __future__ import annotations

import pytest

import gen_worker.video_encode as ve


def _fresh(monkeypatch: pytest.MonkeyPatch, *, nvenc: bool) -> None:
    """Force the probe's answer and clear the process-wide cache."""
    monkeypatch.setattr(ve, "_probe_nvenc", lambda: nvenc)
    monkeypatch.setattr(ve, "_detected", None, raising=False)


def test_x264_pin_no_longer_suppresses_a_working_nvenc(monkeypatch: pytest.MonkeyPatch) -> None:
    """The proof that matters: the env is set to the value that USED to win.

    With the probe forced POSITIVE, the old code would have returned libx264
    here because the pin skipped the probe entirely. Anything less than a
    forced-positive probe would pass on a CPU box for the wrong reason — both
    arms land on x264 and the test proves nothing.
    """
    monkeypatch.setenv("GEN_WORKER_VIDEO_ENCODER", "x264")
    _fresh(monkeypatch, nvenc=True)
    assert ve.detect_encoder().codec == "h264_nvenc"


def test_nvenc_pin_cannot_force_a_missing_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEN_WORKER_VIDEO_ENCODER", "nvenc")
    _fresh(monkeypatch, nvenc=False)
    assert ve.detect_encoder().codec == "libx264"


def test_a_nonsense_value_is_inert_rather_than_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEN_WORKER_VIDEO_ENCODER", "not-an-encoder")
    _fresh(monkeypatch, nvenc=True)
    assert ve.detect_encoder().codec == "h264_nvenc"


def test_the_probe_alone_decides_with_the_env_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEN_WORKER_VIDEO_ENCODER", raising=False)
    _fresh(monkeypatch, nvenc=True)
    assert ve.detect_encoder().codec == "h264_nvenc"
    _fresh(monkeypatch, nvenc=False)
    assert ve.detect_encoder().codec == "libx264"
