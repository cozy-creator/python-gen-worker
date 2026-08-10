"""th#1757: the opt-in reference contract, declaration to manifest.

The value type validates what it can see. What it CANNOT see — whether
``delivery`` names a media field this function actually declares — is the
hub's check at publish, deliberately: only the hub has the input schema and
the moderation media list side by side, and a wrong path must fail a release
rather than a paying request.
"""

import pytest

from gen_worker import AcceptsReferences


def test_manifest_shape_is_what_the_hub_parses():
    ar = AcceptsReferences(
        max=3, modalities=["image"], per_ref_images=(1, 4), delivery="references[].image"
    )
    assert ar.to_manifest() == {
        "max": 3,
        "modalities": ["image"],
        "per_ref_images": {"min": 1, "max": 4},
        "delivery": "references[].image",
    }


@pytest.mark.parametrize(
    "raw,expected",
    [
        (1, {"min": 1, "max": 1}),
        (4, {"min": 4, "max": 4}),
        ((1, 4), {"min": 1, "max": 4}),
        ([2, 3], {"min": 2, "max": 3}),
        ({"min": 1, "max": 9}, {"min": 1, "max": 9}),
        ({"min": 2}, {"min": 2, "max": 2}),
    ],
)
def test_per_ref_images_forms(raw, expected):
    ar = AcceptsReferences(max=1, modalities=["image"], per_ref_images=raw, delivery="d[]")
    assert ar.to_manifest()["per_ref_images"] == expected


def test_modalities_are_case_folded_and_trimmed():
    ar = AcceptsReferences(max=1, modalities=["IMAGE", " video "], delivery="d[]")
    assert ar.to_manifest()["modalities"] == ["image", "video"]


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(max=0, modalities=["image"], delivery="d"), "max must be a positive int"),
        (dict(max=-1, modalities=["image"], delivery="d"), "max must be a positive int"),
        (dict(max=1, modalities=[], delivery="d"), "modalities is required"),
        (dict(max=1, modalities=["hologram"], delivery="d"), "is not one of image|video|audio"),
        (dict(max=1, modalities=["image"], delivery="   "), "delivery is required"),
        (
            dict(max=1, modalities=["image"], delivery="d", per_ref_images=(4, 2)),
            "1 <= min <= max",
        ),
        (
            dict(max=1, modalities=["image"], delivery="d", per_ref_images=0),
            "1 <= min <= max",
        ),
        (
            dict(max=1, modalities=["image"], delivery="d", per_ref_images="lots"),
            "per_ref_images must be an int",
        ),
    ],
)
def test_refusals(kwargs, message):
    with pytest.raises(ValueError) as exc:
        AcceptsReferences(**kwargs)
    assert message in str(exc.value)


def test_worker_function_carries_the_declaration_into_the_registry_spec():
    from gen_worker import worker_function
    from gen_worker.api.decorators import WF_ATTR

    ar = AcceptsReferences(max=2, modalities=["image"], delivery="references[].image")

    @worker_function(accepts_references=ar)
    def reference_to_video(self, ctx, payload):  # pragma: no cover - never called
        raise AssertionError

    decl = getattr(reference_to_video, WF_ATTR)
    assert decl.accepts_references is ar


def test_undeclared_is_absent_not_empty():
    """Omitting the block means the endpoint never sees the concept. It must
    not become a default-permissive `{}` anywhere in the chain."""
    from gen_worker import worker_function
    from gen_worker.api.decorators import WF_ATTR

    @worker_function()
    def generate(self, ctx, payload):  # pragma: no cover - never called
        raise AssertionError

    assert getattr(generate, WF_ATTR).accepts_references is None
