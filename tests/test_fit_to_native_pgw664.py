"""pgw#664 / ie#599: fit-to-native geometry — mechanism tests.

Real PIL images through the real code path; no mocks. The properties under
test are exactly the ie#599 defect statements inverted.
"""

from __future__ import annotations

import math

import pytest
from PIL import Image

from gen_worker import (
    FamilyGeometry,
    FitMode,
    OutputSize,
    fit_to_native,
    nearest_bucket,
    restore,
    set_upscaler,
)
from gen_worker.api.errors import ValidationError


# The qwen-image EDIT table: the pipeline's own VAE_IMAGE_SIZE, and rows that
# already exist in the endpoint's declared Compile(shapes=...).
QWEN_EDIT = FamilyGeometry(
    name="qwen-image",
    native_area=1024 * 1024,
    buckets=(
        (1024, 1024), (1152, 864), (864, 1152),
        (1248, 832), (832, 1248), (1280, 720), (720, 1280),
    ),
)

KLEIN_EDIT = FamilyGeometry(
    name="flux2-klein",
    native_area=1024 * 1024,
    buckets=(
        (1024, 1024), (1184, 880), (880, 1184), (1248, 832), (832, 1248),
        (1392, 752), (752, 1392), (1568, 672), (672, 1568),
    ),
)


_TILE = Image.new("RGB", (64, 64))
_TILE.putdata([((x * 7) % 256, (y * 11) % 256, ((x ^ y) * 3) % 256)
               for y in range(64) for x in range(64)])


def _photo(width: int, height: int) -> Image.Image:
    """A deterministic image with structure in every region."""
    return _TILE.resize((width, height), Image.Resampling.NEAREST)


# -- the table itself refuses the defect -----------------------------------


def test_t2i_table_on_an_edit_lane_is_refused_at_declaration():
    """The exact ie#599 root cause: Qwen's ~1.7 MP t2i rows on a 1 MP lane."""
    with pytest.raises(ValueError, match="native area"):
        FamilyGeometry(
            name="qwen-image",
            native_area=1024 * 1024,
            buckets=((1328, 1328), (1472, 1104), (1664, 928)),
        )


def test_rows_must_be_latent_aligned():
    with pytest.raises(ValueError, match="multiple of"):
        FamilyGeometry(name="x", native_area=1024 * 1024, buckets=((1023, 1025),))


def test_assert_declared_catches_an_undeclared_row():
    QWEN_EDIT.assert_declared(QWEN_EDIT.buckets)
    with pytest.raises(ValueError, match="compile shape set"):
        QWEN_EDIT.assert_declared(((1024, 1024),))


# -- the input's area must never select the tier ---------------------------


@pytest.mark.parametrize("scale", [0.05, 0.25, 1.0, 4.0, 12.0])
def test_bucket_is_area_invariant(scale: float):
    """A 500x500 thumbnail and a 12 MP photo edit at the SAME bucket."""
    base = (1024, 1024)
    size = (int(base[0] * math.sqrt(scale)), int(base[1] * math.sqrt(scale)))
    assert nearest_bucket(*size, QWEN_EDIT) == (1024, 1024)


def test_bucket_follows_aspect_not_size():
    assert nearest_bucket(4032, 3024, QWEN_EDIT) == (1152, 864)   # 4:3 phone photo
    assert nearest_bucket(1920, 1080, QWEN_EDIT) == (1280, 720)   # 16:9
    assert nearest_bucket(1080, 1920, QWEN_EDIT) == (720, 1280)   # 9:16
    assert nearest_bucket(500, 500, QWEN_EDIT) == (1024, 1024)


def test_primary_condition_and_output_share_one_bucket():
    """The ie#599 pairing: the image being edited sits on the output's row."""
    images = [_photo(768, 512), _photo(400, 400)]
    plan = fit_to_native(images, QWEN_EDIT)
    assert plan.bucket == (1248, 832)
    assert plan.images[0].size == plan.bucket


def test_secondary_references_keep_their_own_native_row():
    """Content references are not framing: no 44% padding into a square."""
    images = [_photo(1024, 1024), _photo(1920, 1080), _photo(1080, 1920)]
    plan = fit_to_native(images, QWEN_EDIT)
    assert [image.size for image in plan.images] == [
        (1024, 1024), (1280, 720), (720, 1280)]
    # Same grid SCALE for every condition — that is what the target latent needs.
    areas = [w * h for w, h in (image.size for image in plan.images)]
    assert max(areas) / min(areas) < 1.2


# -- pad, never stretch; crop back to the user's framing -------------------


@pytest.mark.parametrize("size", [(768, 512), (500, 500), (1080, 1920), (4032, 3024), (900, 700)])
def test_pad_then_crop_returns_the_submitted_aspect(size):
    plan = fit_to_native([_photo(*size)], QWEN_EDIT)
    # The model's own output, faked at the plan's bucket.
    result = restore(_photo(*plan.bucket), plan)
    submitted = size[0] / size[1]
    returned = result.size[0] / result.size[1]
    assert abs(math.log(returned / submitted)) < 0.01, (size, result.size)


def test_the_fit_never_stretches():
    """Aspect through the fit is preserved to sub-percent, for every row."""
    for size in [(768, 512), (1080, 1920), (4032, 3024), (300, 900), (1600, 900)]:
        plan = fit_to_native([_photo(*size)], QWEN_EDIT)
        left, top, right, bottom = plan.crop_box
        inner = (right - left) / (bottom - top)
        assert abs(math.log(inner / (size[0] / size[1]))) < 0.01, size


def test_pad_is_reversible_pixel_for_pixel():
    """Crop(pad(x)) == resize(x) exactly: the pad adds nothing recoverable."""
    source = _photo(768, 512)
    plan = fit_to_native([source], QWEN_EDIT)
    inner = plan.images[0].crop(plan.crop_box)
    left, top, right, bottom = plan.crop_box
    direct = source.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    assert list(inner.getdata()) == list(direct.getdata())


def test_padding_is_only_ever_a_thin_margin():
    """With the declared rows, no input loses more than 12% of the bucket."""
    for size in [(768, 512), (500, 500), (1080, 1920), (4032, 3024), (1600, 900)]:
        plan = fit_to_native([_photo(*size)], QWEN_EDIT)
        left, top, right, bottom = plan.crop_box
        used = ((right - left) * (bottom - top)) / (plan.bucket[0] * plan.bucket[1])
        assert used > 0.88, (size, used)


# -- output_size ------------------------------------------------------------


def test_native_returns_the_bucket_unrestored():
    plan = fit_to_native([_photo(768, 512)], QWEN_EDIT, output_size=OutputSize.NATIVE)
    result = restore(_photo(*plan.bucket), plan)
    assert result.size == plan.bucket
    assert not result.upscaled and not result.composited


def test_preset_requires_a_declared_row():
    with pytest.raises(ValidationError, match="not a declared native row"):
        fit_to_native(
            [_photo(768, 512)], QWEN_EDIT,
            output_size=OutputSize.PRESET, preset=(1328, 1328),
        )
    plan = fit_to_native(
        [_photo(768, 512)], QWEN_EDIT,
        output_size=OutputSize.PRESET, preset=(1024, 1024),
    )
    assert plan.bucket == (1024, 1024)
    assert plan.images[0].size == (1024, 1024)


def test_below_native_input_defaults_to_the_native_bucket():
    """ie#599 §6: small inputs edit at native, not at their own size."""
    plan = fit_to_native([_photo(500, 500)], QWEN_EDIT)
    assert plan.bucket == (1024, 1024)
    assert plan.target_size == (1024, 1024)
    assert not plan.composite


# -- the super-resolution stage is declared, pluggable, and absent ---------


def test_no_upscaler_means_native_size_stated_honestly():
    source = _photo(4032, 3024)
    plan = fit_to_native([source], QWEN_EDIT)
    assert plan.composite and plan.target_size == (4032, 3024)
    result = restore(_photo(*plan.bucket), plan)
    assert result.size == (1152, 864)
    assert not result.upscaled and not result.composited
    assert "none is deployed" in result.note


def test_a_registered_upscaler_drives_the_composite():
    source = _photo(2304, 1728)
    plan = fit_to_native([source], QWEN_EDIT)
    # Stand-in for the capability that does not exist yet.
    def _stub(image, target):
        return image.resize(target, Image.Resampling.LANCZOS)

    edited = plan.images[0].copy()
    edited.paste(Image.new("RGB", (200, 200), (255, 0, 0)), (400, 300))
    result = restore(edited, plan, upscaler=_stub)
    assert result.size == (2304, 1728)
    assert result.upscaled and result.composited
    # Untouched corner comes from the SOURCE, not from an upscale of the edit.
    assert result.image.getpixel((5, 5)) == source.getpixel((5, 5))


def test_set_upscaler_is_a_registration_seam_with_no_default():
    from gen_worker.geometry import current_upscaler

    assert current_upscaler() is None
    previous = set_upscaler(lambda image, target: None)
    try:
        assert current_upscaler() is not None
    finally:
        set_upscaler(previous)
    assert current_upscaler() is None


# -- mode ------------------------------------------------------------------


def test_compose_mode_fits_references_and_leaves_output_free():
    refs = [_photo(768, 512), _photo(400, 900)]
    plan = fit_to_native(
        refs, KLEIN_EDIT, mode=FitMode.COMPOSE, preset=(1024, 1024),
    )
    assert plan.bucket == (1024, 1024)
    assert plan.images[0].size == (1248, 832)   # each reference at its own row
    assert plan.images[1].size == (672, 1568)
    result = restore(_photo(1024, 1024), plan)
    assert result.size == (1024, 1024) and not result.composited


def test_compose_never_inherits_reference_geometry_silently():
    """ie#600: compose output geometry is the caller's, not references[0]'s."""
    with pytest.raises(ValidationError, match="owns output geometry"):
        fit_to_native([_photo(1600, 900)], KLEIN_EDIT, mode=FitMode.COMPOSE)


def test_edit_mode_and_compose_mode_disagree_on_purpose():
    refs = [_photo(1600, 900)]
    edit = fit_to_native(refs, KLEIN_EDIT, mode=FitMode.EDIT)
    compose = fit_to_native(refs, KLEIN_EDIT, mode=FitMode.COMPOSE, preset=(1024, 1024))
    assert edit.bucket == (1392, 752) and compose.bucket == (1024, 1024)
    assert restore(_photo(*edit.bucket), edit).size != (1024, 1024)


def test_empty_input_is_a_typed_refusal():
    with pytest.raises(ValidationError):
        fit_to_native([], QWEN_EDIT)
