"""RequestContext surface: <=15 public members, one cancellation spelling,
typed save_* assets, generator(seed), and real subclass inheritance for the
producer contexts (no import-time setattr)."""

from __future__ import annotations

import pytest

from gen_worker import (
    CanceledError,
    ConversionContext,
    DatasetContext,
    RequestContext,
    TrainingContext,
)
from gen_worker.api.types import Asset, ImageAsset


def _ctx(**kw) -> RequestContext:
    return RequestContext(request_id="r1", **kw)


def test_public_surface_is_capped_at_15_members() -> None:
    members = sorted(
        n for n in dir(RequestContext)
        if not n.startswith("_")
    )
    expected = {
        "request_id", "device", "deadline", "time_remaining",
        "cancelled", "raise_if_cancelled", "progress", "log",
        "save_bytes", "save_file", "save_image", "save_audio", "save_video",
        "generator", "models",
    }
    assert set(members) == expected
    assert len(members) <= 15


def test_one_cancellation_spelling() -> None:
    ctx = _ctx()
    assert ctx.cancelled is False
    ctx.raise_if_cancelled()  # no-op
    ctx._cancel()
    assert ctx.cancelled is True
    with pytest.raises(CanceledError):
        ctx.raise_if_cancelled()
    # the old spellings are gone
    for old in ("is_canceled", "raise_if_canceled", "cancel", "done", "emit"):
        assert not hasattr(ctx, old)


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


def test_producer_contexts_are_real_subclasses() -> None:
    for cls in (ConversionContext, DatasetContext, TrainingContext):
        assert issubclass(cls, RequestContext)
        ctx = cls(request_id="r1")
        # producer surface lives on the subclass, not the base
        assert hasattr(ctx, "save_checkpoint")
        assert hasattr(ctx, "set_repo_spec")
        assert hasattr(ctx, "hf_token")
        # checkpoint publishing is cozy_convert.publish_flavors, not a ctx RPC
        assert not hasattr(ctx, "publish_repo_revision")
    assert hasattr(ConversionContext(request_id="r1"), "mktemp")
    assert hasattr(DatasetContext(request_id="r1"), "resolve_dataset")
    base = RequestContext(request_id="r1")
    for producer_only in ("save_checkpoint", "mktemp"):
        assert not hasattr(base, producer_only)


def test_models_property_copies() -> None:
    ctx = RequestContext(request_id="r1", models={"pipe": "o/r"})
    m = ctx.models
    m["pipe"] = "mutated"
    assert ctx.models["pipe"] == "o/r"
