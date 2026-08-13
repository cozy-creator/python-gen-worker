"""pgw#942: the recording test context and the SDK's declaration-shape contract.

These are integration tests in the sense that matters here: nothing below is
mocked. A real ``@endpoint`` class is declared, a real handler runs, and the
assertions read what the SDK's own ``save_*`` implementations produced —
encode, ref normalization, sha256 and size included. That is precisely the
path the 23 endpoint suites with a hand-rolled ``_Ctx.save_image`` override
never execute.
"""

from __future__ import annotations

import io
from typing import Optional

import msgspec
import pytest
from PIL import Image

from gen_worker import HF, Slot, endpoint
from gen_worker.api.decorators import Compile, Resources
from gen_worker.api.types import AudioAsset, ImageAsset, VideoAsset
from gen_worker.request_context import ConversionContext, RequestContext
from gen_worker.testing import (
    Recorder,
    fake_context,
)

from _example_family import ExampleDefaults


class Render(msgspec.Struct, frozen=True):
    size: int = 64
    image_format: str = "webp"
    ref: Optional[str] = None


class Rendered(msgspec.Struct, frozen=True):
    image: ImageAsset


@endpoint(
    resources=Resources(gpu=True, vram_gb_hint=12.0),
    compile=Compile(family="example", shapes=((512, 512),), text_len=77),
    models={"pipeline": Slot(str, default_checkpoint=HF("example/pipeline"))},
)
class ExampleEndpoint:
    """A minimally real endpoint: one root slot, one handler that saves."""

    def setup(self, pipeline: str) -> None:
        self.pipeline = pipeline

    def generate(self, ctx: RequestContext[ExampleDefaults], payload: Render) -> Rendered:
        ctx.progress(0.5, "denoise", step=1, total=2)
        ctx.log("rendering", level="info", steps=ctx.defaults.steps)
        # Real content, not a solid fill: pgw#1094's output-integrity floor
        # rejects a constant-fill render.
        image = Image.merge("RGB", (
            Image.linear_gradient("L"),
            Image.linear_gradient("L").rotate(90),
            Image.radial_gradient("L"),
        )).resize((payload.size, payload.size), Image.Resampling.BILINEAR)
        return Rendered(
            image=ctx.save_image(
                image, payload.ref, format=payload.image_format, quality=80
            )
        )


def _slots() -> dict:
    return {"pipeline": (HF("example/pipeline"), ExampleDefaults(steps=7))}


# --------------------------------------------------------------------------- #
# 1. recording mode: real encode, inspectable result                          #
# --------------------------------------------------------------------------- #


def test_recorder_captures_a_real_webp_encode() -> None:
    rec = Recorder()
    ctx = fake_context(request_id="req", slots=_slots(), recorder=rec)

    out = ExampleEndpoint().generate(ctx, Render())

    assert isinstance(out.image, ImageAsset)
    assert rec.refs == ["outputs/req/image.webp"]
    saved = rec.images[0]
    assert saved.call == {"format": "webp", "quality": 80}

    # The bytes exist and are a real webp — the encode the handler's own
    # save_image override would have skipped.
    payload = saved.read_bytes()
    assert payload[:4] == b"RIFF" and payload[8:12] == b"WEBP"
    decoded = Image.open(io.BytesIO(payload))
    assert decoded.size == (64, 64)

    # ...and the asset the HANDLER received carries the real attestation.
    assert saved.asset.sha256 and len(saved.asset.sha256) == 64
    assert saved.asset.size_bytes == len(payload)
    assert out.image.sha256 == saved.asset.sha256


def test_recorder_captures_the_format_the_handler_asked_for() -> None:
    rec = Recorder()
    ctx = fake_context(request_id="req", slots=_slots(), recorder=rec)

    ExampleEndpoint().generate(ctx, Render(image_format="png"))

    assert rec.refs == ["outputs/req/image.png"]
    assert rec.images[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_recorder_captures_log_and_progress_off_the_real_emitter() -> None:
    rec = Recorder()
    ctx = fake_context(request_id="req", slots=_slots(), recorder=rec)

    ExampleEndpoint().generate(ctx, Render())

    assert rec.messages == ["rendering"]
    assert rec.logs[0].payload["fields"] == {"steps": 7}
    assert [e.payload["step"] for e in rec.progress] == [1]


def test_nested_saves_record_once_under_the_handler_kind() -> None:
    """``save_image`` calls ``save_bytes`` internally; one handler call is one
    recorded artifact, and its kind is the one the handler asked for."""
    rec = Recorder()
    ctx = fake_context(request_id="req", slots=_slots(), recorder=rec)

    ExampleEndpoint().generate(ctx, Render())

    assert len(rec.saved) == 1
    assert rec.saved[0].kind == "image"


def test_recorder_covers_every_save_kind() -> None:
    rec = Recorder()
    ctx = fake_context(request_id="req", slots=_slots(), recorder=rec)

    ctx.save_bytes("outputs/req/blob.bin", b"raw")
    audio = ctx.save_audio(b"RIFFfake", "outputs/req/a.wav")
    video = ctx.save_video(b"\x00\x00\x00 ftypisom", "outputs/req/v.mp4")

    src = rec.output_dir / "src.txt"
    src.write_text("from disk")
    ctx.save_file("outputs/req/copy.txt", src)

    assert [a.kind for a in rec.saved] == ["bytes", "audio", "video", "file"]
    assert isinstance(audio, AudioAsset) and isinstance(video, VideoAsset)
    assert rec.files[0].read_bytes() == b"from disk"
    assert rec.audio[0].asset.size_bytes == len(b"RIFFfake")


def test_recorder_works_for_producer_context_subclasses() -> None:
    rec = Recorder()
    ctx = fake_context(request_id="req", cls=ConversionContext, recorder=rec)

    assert isinstance(ctx, ConversionContext)
    ctx.save_bytes("outputs/req/blob.bin", b"raw")
    assert rec.refs == ["outputs/req/blob.bin"]


def test_recorder_chains_a_caller_supplied_emitter() -> None:
    seen: list[dict] = []
    rec = Recorder()
    ctx = fake_context(
        request_id="req", slots=_slots(), recorder=rec, emitter=seen.append
    )

    ctx.log("both")

    assert rec.messages == ["both"]
    assert [e["payload"]["message"] for e in seen] == ["both"]


def test_fake_context_without_a_recorder_is_unchanged() -> None:
    ctx = fake_context(request_id="req", slots=_slots())

    assert type(ctx) is RequestContext
    assert ctx.slots["pipeline"].defaults.steps == 7


def test_recorder_output_dir_is_removed_with_the_recorder() -> None:
    rec = Recorder()
    path = rec.output_dir
    assert path.is_dir()
    del rec
    assert not path.exists()


# --------------------------------------------------------------------------- #
# 2. the SDK asserts its own deleted-field set, once                          #
# --------------------------------------------------------------------------- #



