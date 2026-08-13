"""th#1771: the derived warm payload must SATISFY the payload's own invariants.

`_struct_factory` filled REQUIRED fields only. A nested struct whose fields are
all optional but which asserts "exactly one of these is set" in `__post_init__`
— the standard one-of media shape (minimax-h3 `ReferenceInput`: image | video |
audio) — therefore synthesized as the all-`None` instance and raised its own
`ValueError` the first time the warm plan built it.

That is not a sparse legal set (`IllegalCombination`, which the plan drops); it
is the synthesizer emitting a value the schema forbids unconditionally. Live it
cost a live H100 and a billed request: the endpoint served fine, the warm plan
could not build a payload, and the hub booked the ValueError as the CALLER's
fatal error on a request whose own `references[]` were perfectly formed.

The fix: when required-only construction is refused, add the synthesizable
OPTIONAL fields, in declaration order, until the struct constructs.
"""

from __future__ import annotations

import tempfile

import msgspec
import pytest

from gen_worker.api.types import AudioAsset, ImageAsset, VideoAsset
from gen_worker.warmup import synthesize_factory


class OneOfMedia(msgspec.Struct, forbid_unknown_fields=True):
    """minimax-h3's `ReferenceInput`, verbatim in shape."""

    image: ImageAsset | None = None
    video: VideoAsset | None = None
    audio: AudioAsset | None = None

    def __post_init__(self) -> None:
        populated = [
            name
            for name, value in (
                ("image", self.image), ("video", self.video), ("audio", self.audio))
            if value is not None
        ]
        if len(populated) != 1:
            raise ValueError(
                "each references[] compiled_graph carries exactly one of image / video / audio, "
                f"got {populated or 'none'}"
            )


class RefToVideoIn(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    references: list[OneOfMedia]
    duration_s: int = 5


class OnlyUnsynthesizableOptions(msgspec.Struct):
    """No arm the synthesizer can fill — must be a REASON, never a crash."""

    video: VideoAsset | None = None

    def __post_init__(self) -> None:
        if self.video is None:
            raise ValueError("video is required in practice")


def _build(payload_type: type):
    factory, reason = synthesize_factory(payload_type)
    assert factory is not None, reason
    with tempfile.TemporaryDirectory() as tmp:
        return factory(tmp)


def test_oneof_nested_struct_synthesizes_a_legal_instance() -> None:
    payload = _build(RefToVideoIn)
    assert len(payload.references) == 1
    compiled_graph = payload.references[0]
    assert compiled_graph.image is not None, "the one-of arm must be populated"
    assert compiled_graph.video is None and compiled_graph.audio is None
    assert compiled_graph.image.local_path, "the synthetic image must exist on disk"


def test_oneof_struct_alone_synthesizes() -> None:
    compiled_graph = _build(OneOfMedia)
    assert compiled_graph.image is not None


def test_unsatisfiable_oneof_is_a_reason_not_a_crash() -> None:
    factory, reason = synthesize_factory(OnlyUnsynthesizableOptions)
    if factory is None:
        assert reason
        return
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(Exception):
        factory(tmp)


def test_plain_struct_still_keeps_its_defaults() -> None:
    class Plain(msgspec.Struct):
        prompt: str
        steps: int = 7
        image: ImageAsset | None = None

    payload = _build(Plain)
    assert payload.steps == 7
    assert payload.image is None, "an unconstrained optional stays at its default"
