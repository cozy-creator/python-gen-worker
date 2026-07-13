"""RequestContext surface: cancellation, deadlines, events, typed save_* assets."""

from __future__ import annotations

import pytest

from gen_worker import CanceledError, RequestContext
from gen_worker.api.types import Asset, ImageAsset


def _ctx(**kw) -> RequestContext:
    return RequestContext(request_id="r1", **kw)


def test_cancel_trips_cancelled_and_raise_if_cancelled() -> None:
    ctx = _ctx()
    assert ctx.cancelled is False
    ctx.raise_if_cancelled()  # no-op
    ctx._cancel()
    assert ctx.cancelled is True
    with pytest.raises(CanceledError):
        ctx.raise_if_cancelled()


def test_deadline_and_time_remaining() -> None:
    assert _ctx().time_remaining() is None
    ctx = _ctx(timeout_ms=60_000)
    assert ctx.deadline is not None
    assert 0 < ctx.time_remaining() <= 60.0


def test_progress_and_log_emit_events() -> None:
    events = []
    ctx = RequestContext(request_id="r1", emitter=events.append)
    ctx.progress(0.5, stage="denoise")
    ctx.log("hello", level="warning")
    kinds = [e["type"] for e in events]
    assert kinds == ["request.progress", "request.log"]
    assert events[0]["payload"] == {"progress": 0.5, "stage": "denoise"}
    assert events[1]["payload"] == {"message": "hello", "level": "warning"}


def test_log_default_level_and_structured_fields() -> None:
    events = []
    ctx = RequestContext(request_id="r1", emitter=events.append)
    ctx.log("plain")
    ctx.log("OOM retry", level="warning", free_gb=2.1, rung="offload")
    assert events[0]["payload"] == {"message": "plain", "level": "info"}
    assert events[1]["payload"] == {
        "message": "OOM retry",
        "level": "warning",
        "fields": {"free_gb": 2.1, "rung": "offload"},
    }


def test_progress_carries_optional_step_and_total() -> None:
    events = []
    ctx = RequestContext(request_id="r1", emitter=events.append)
    ctx.progress(0.25, "denoise", step=5, total=20)
    ctx.progress(0.9)  # step/total omitted -> keys absent, not null
    assert events[0]["payload"] == {
        "progress": 0.25, "stage": "denoise", "step": 5, "total": 20,
    }
    assert events[1]["payload"] == {"progress": 0.9}


def test_save_bytes_and_typed_image_asset(tmp_path) -> None:
    ctx = RequestContext(request_id="r1", local_output_dir=str(tmp_path))
    asset = ctx.save_bytes("out/a.bin", b"hello")
    assert isinstance(asset, Asset)
    assert (tmp_path / "out/a.bin").read_bytes() == b"hello"

    pil = pytest.importorskip("PIL.Image")
    img = pil.new("RGB", (4, 4))
    out = ctx.save_image(img, "out/pic", format="png")
    assert isinstance(out, ImageAsset)
    assert out.ref.endswith(".png")
    assert (tmp_path / out.ref).exists()


def test_save_audio_bytes_passthrough(tmp_path) -> None:
    ctx = RequestContext(request_id="r1", local_output_dir=str(tmp_path))
    out = ctx.save_audio(b"RIFFfake", "out/a", format="wav")
    assert out.ref.endswith(".wav")
    assert (tmp_path / out.ref).read_bytes() == b"RIFFfake"


def test_save_video_from_file(tmp_path) -> None:
    src = tmp_path / "in.mp4"
    src.write_bytes(b"vid")
    ctx = RequestContext(request_id="r1", local_output_dir=str(tmp_path / "outs"))
    out = ctx.save_video(src, "clips/final")
    assert out.ref.endswith(".mp4")
    assert (tmp_path / "outs" / out.ref).read_bytes() == b"vid"


def test_models_property_copies() -> None:
    ctx = RequestContext(request_id="r1", models={"pipe": "o/r"})
    m = ctx.models
    m["pipe"] = "mutated"
    assert ctx.models["pipe"] == "o/r"
