from __future__ import annotations

from typing import Annotated

import msgspec
import pytest

from gen_worker import (
    Asset,
    AudioAsset,
    ExpectedOutput,
    HFRepo,
    ImageAsset,
    NegativePrompt,
    PositivePrompt,
    PromptRole,
    RequestContext,
    StringEnum,
    Tensors,
    VideoAsset,
    inference,
)
from gen_worker.discovery.discover import (
    _collect_payload_moderation_metadata,
    _collect_expected_output_metadata,
    _extract_class_function_methods,
)


class _Out(msgspec.Struct):
    ok: bool


class _ImageOut(msgspec.Struct):
    images: Annotated[
        list[ImageAsset],
        ExpectedOutput(
            count="input.num_images",
            width="input.width",
            height="input.height",
            mime_type="image/png",
        ),
    ]


class _Payload(msgspec.Struct):
    prompt: PositivePrompt
    negative_prompt: NegativePrompt | None = None
    image: ImageAsset | None = None
    video: list[VideoAsset] = msgspec.field(default_factory=list)
    audio: AudioAsset | None = None
    generic_file: Asset | None = None
    weights: Tensors | None = None


class _ImagePayload(msgspec.Struct):
    prompt: PositivePrompt
    num_images: int = 1
    width: int = 1024
    height: int = 1024


class AspectRatio(StringEnum):
    SQUARE = "1:1"
    WIDE = "16:9"


class _AspectRatioPayload(msgspec.Struct):
    prompt: PositivePrompt
    aspect_ratio: AspectRatio = AspectRatio.SQUARE


class _AspectRatioImageOut(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        ExpectedOutput(
            aspect_ratio="input.aspect_ratio",
            mime_type="image/png",
        ),
    ]


def test_prompt_and_media_types_are_public_msgspec_shapes() -> None:
    schema = msgspec.json.schema(_Payload)
    defs = schema["$defs"]
    assert defs["ImageAsset"]["properties"]["ref"]["type"] == "string"
    assert defs["ImageAsset"]["properties"] == defs["Asset"]["properties"]
    assert defs["VideoAsset"]["properties"] == defs["Asset"]["properties"]
    assert defs["AudioAsset"]["properties"] == defs["Asset"]["properties"]


def test_moderation_metadata_discovers_prompt_and_media_only() -> None:
    moderation = _collect_payload_moderation_metadata(_Payload)

    assert moderation["prompts"] == [
        {"field": "prompt", "role": "positive"},
        {"field": "negative_prompt", "role": "negative"},
    ]
    assert moderation["media"] == [
        {"field": "image", "kind": "image"},
        {"field": "video[]", "kind": "video"},
        {"field": "audio", "kind": "audio"},
    ]
    assert "generic_file" not in repr(moderation)
    assert "weights" not in repr(moderation)


def test_endpoint_declared_string_enum_emits_json_schema_enum() -> None:
    payload = msgspec.json.decode(b'{"prompt":"x","aspect_ratio":"16:9"}', type=_AspectRatioPayload)

    assert payload.aspect_ratio is AspectRatio.WIDE
    assert str(payload.aspect_ratio) == "16:9"

    schema = msgspec.json.schema(_AspectRatioPayload)
    aspect_ratio = schema["$defs"]["_AspectRatioPayload"]["properties"]["aspect_ratio"]
    assert aspect_ratio == {"$ref": "#/$defs/AspectRatio", "default": "1:1"}
    assert sorted(schema["$defs"]["AspectRatio"]["enum"]) == ["16:9", "1:1"]


def test_prompt_role_requires_string_annotation() -> None:
    class BadPayload(msgspec.Struct):
        prompt: Annotated[int, PromptRole("positive")]

    with pytest.raises(ValueError, match="must annotate str"):
        _collect_payload_moderation_metadata(BadPayload)


def test_expected_output_metadata_discovers_annotated_media_outputs() -> None:
    expected = _collect_expected_output_metadata(_ImagePayload, _ImageOut)

    assert expected == [
        {
            "field": "images",
            "type": "image",
            "count": "input.num_images",
            "width": "input.width",
            "height": "input.height",
            "mime_type": "image/png",
        }
    ]


def test_expected_output_metadata_discovers_aspect_ratio_ref() -> None:
    expected = _collect_expected_output_metadata(_AspectRatioPayload, _AspectRatioImageOut)

    assert expected == [
        {
            "field": "image",
            "type": "image",
            "count": 1,
            "aspect_ratio": "input.aspect_ratio",
            "mime_type": "image/png",
        }
    ]


def test_expected_output_metadata_rejects_unknown_input_refs() -> None:
    class BadOut(msgspec.Struct):
        image: Annotated[ImageAsset, ExpectedOutput(width="input.missing")]

    with pytest.raises(ValueError, match="unknown payload field"):
        _collect_expected_output_metadata(_ImagePayload, BadOut)


def test_class_discovery_emits_moderation_and_allow_lora_metadata() -> None:
    class _Pipe:
        pass

    @inference(models={"pipeline": HFRepo("base_model", "black-forest-labs/FLUX.2-klein-4B").allow_lora()})
    class Endpoint:
        def setup(self, pipeline: _Pipe) -> None:
            self.pipeline = pipeline

        @inference.function
        def generate(self, ctx: RequestContext, payload: _ImagePayload) -> _ImageOut:
            return _ImageOut(images=[])

    [entry] = _extract_class_function_methods(Endpoint, "endpoint")
    assert entry["moderation"]["prompts"][0] == {"field": "prompt", "role": "positive"}
    assert entry["expected_outputs"][0]["count"] == "input.num_images"
    assert entry["expected_outputs"][0]["mime_type"] == "image/png"
    assert entry["bindings"]["pipeline"]["slot_name"] == "base_model"
    assert entry["bindings"]["pipeline"]["allow_lora"] is True
