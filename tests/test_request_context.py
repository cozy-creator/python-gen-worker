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


# ---------------------------------------------------------------------------
# pgw#526: producer state lives on _PublisherMixin, not the inference base.
# ---------------------------------------------------------------------------

def test_inference_ctx_carries_no_producer_state() -> None:
    ctx = _ctx()
    for attr in ("_source_info", "_destination_info", "_source_path",
                 "_text_encoder_info", "_text_encoder_path",
                 "_hf_token", "_repo_spec"):
        assert not hasattr(ctx, attr), attr
    for surface in ("hf_token", "source", "destination", "source_path",
                    "text_encoder", "text_encoder_path",
                    "set_repo_spec", "save_checkpoint", "checkpoint_dir",
                    "compute"):
        assert not hasattr(ctx, surface), surface


def test_producer_ctx_carries_producer_state() -> None:
    from gen_worker.request_context import TrainingContext

    ctx = TrainingContext(
        request_id="r1",
        source_info={"ref": "o/base"},
        destination_info={"ref": "o/dest"},
        hf_token="  tok  ",
    )
    assert ctx.source == {"ref": "o/base"}
    assert ctx.destination == {"ref": "o/dest"}
    assert ctx.hf_token == "tok"
    assert ctx.source_path is None
    ctx._set_source_path("/models/base")
    assert ctx.source_path == "/models/base"
    ctx.set_repo_spec(kind="lora", model_family="sdxl")
    assert ctx._repo_spec == {"kind": "lora", "model_family": "sdxl"}


def test_producer_ctx_carries_text_encoder_state() -> None:
    """pgw#594/te#70: second reserved model input, independent of `source`."""
    from gen_worker.request_context import TrainingContext

    ctx = TrainingContext(
        request_id="r1",
        source_info={"ref": "o/dit-base"},
        text_encoder_info={"ref": "o/gemma-3-12b"},
    )
    assert ctx.text_encoder == {"ref": "o/gemma-3-12b"}
    assert ctx.text_encoder_path is None
    ctx._set_text_encoder_path("/models/text-encoder")
    assert ctx.text_encoder_path == "/models/text-encoder"
    # Independent of `source` — setting one never clobbers the other.
    assert ctx.source_path is None
    ctx._set_source_path("/models/dit-base")
    assert ctx.source_path == "/models/dit-base"
    assert ctx.text_encoder_path == "/models/text-encoder"


def test_producer_ctx_text_encoder_defaults_empty() -> None:
    from gen_worker.request_context import TrainingContext

    ctx = TrainingContext(request_id="r1", source_info={"ref": "o/base"})
    assert ctx.text_encoder == {}
    assert ctx.text_encoder_path is None


def test_producer_kwargs_rejected_on_inference_ctx() -> None:
    with pytest.raises(TypeError):
        RequestContext(request_id="r1", hf_token="tok")  # type: ignore[call-arg]


def test_save_file_create_flag_local_backend(tmp_path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    ctx = RequestContext(request_id="r1", local_output_dir=str(tmp_path / "outs"))
    out = ctx.save_file("outs/a.bin", src, create=True)
    assert isinstance(out, Asset)
    with pytest.raises(RuntimeError):
        ctx.save_file("outs/a.bin", src, create=True)
    # Plain save_file overwrites freely.
    out2 = ctx.save_file("outs/a.bin", src)
    assert out2.size_bytes == len(b"payload")
