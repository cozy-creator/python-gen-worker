"""gw#614: media-variant payload derivation — the synthesized edit-modality
warmup must be the BASE warmup payload plus exactly one generated media
field, so lane token/guidance/shape derivation matches a real request of
that modality (key fidelity is payload-equality, nothing else drifts)."""

from __future__ import annotations

from typing import Annotated, Optional

import msgspec

from gen_worker import warmup
from gen_worker.api.types import AudioAsset, ImageAsset, VideoAsset


class _QwenShaped(msgspec.Struct, forbid_unknown_fields=True):
    """The qwen merged-endpoint input shape (input-routed edit lane)."""

    prompt: str
    images: Annotated[list[ImageAsset], msgspec.Meta(max_length=3)] = []
    turbo: bool = False
    num_inference_steps: Optional[int] = None


def test_variant_is_base_plus_one_image_nothing_else(tmp_path):
    base = lambda d: _QwenShaped(prompt="warmup", num_inference_steps=10)  # noqa: E731
    variants = warmup.media_variants(_QwenShaped, base)
    assert [label for label, _ in variants] == ["media:images"]
    payload = variants[0][1](str(tmp_path))
    assert payload is not None
    assert payload.prompt == "warmup"
    assert payload.turbo is False
    assert payload.num_inference_steps == 10, (
        "regime fields must come from the base payload — any drift changes "
        "the compile keys the mint traces")
    assert len(payload.images) == 1
    assert isinstance(payload.images[0], ImageAsset)
    assert (tmp_path / "warmup.png").is_file()


def test_populated_base_media_yields_no_variant_payload(tmp_path):
    def base(d):
        return _QwenShaped(
            prompt="warmup",
            images=[ImageAsset(ref="user.png", local_path=warmup.synthetic_png(d))],
        )

    variants = warmup.media_variants(_QwenShaped, base)
    assert variants[0][1](str(tmp_path)) is None, (
        "a base that already carries media needs no synthesized variant")


def test_no_optional_media_fields_no_variants():
    class _Plain(msgspec.Struct):
        prompt: str = ""
        seed: Optional[int] = None

    assert warmup.media_variants(_Plain, lambda d: _Plain()) == []


def test_optional_scalar_and_audio_fields_are_variantable(tmp_path):
    class _Multi(msgspec.Struct):
        text: str = ""
        reference: Optional[ImageAsset] = None
        track: Optional[AudioAsset] = None
        clip: Optional[VideoAsset] = None  # not synthesizable: no variant

    variants = dict(warmup.media_variants(_Multi, lambda d: _Multi()))
    assert set(variants) == {"media:reference", "media:track"}
    img = variants["media:reference"](str(tmp_path))
    assert isinstance(img.reference, ImageAsset) and img.track is None
    aud = variants["media:track"](str(tmp_path))
    assert isinstance(aud.track, AudioAsset) and aud.reference is None
