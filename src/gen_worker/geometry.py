"""Fit an image workload to a family's NATIVE geometry (pgw#664, grown by ie#599).

The library owns the MECHANISM (snap / pad / crop / composite / restore); the
family declares the DATA (:class:`FamilyGeometry`: native area + the declared
grid rows). One bucket is chosen for the CONDITION and the OUTPUT together —
picking them independently is the ie#599 defect (a t2i ~1.7 MP aspect table
copied onto an edit lane whose native area is 1 MP).

Two rules that are not negotiable:

* The INPUT's own area never selects the tier — the MODEL's native area does.
  A 12 MP phone photo and a 500x500 thumbnail both edit at the native bucket.
* Aspect mismatch is resolved by PAD (reversible), never by stretch or
  crop-to-fill. :func:`restore` crops the pad box back off, so an ``edit``
  returns the exact framing the user submitted.

Geometry rules differ BY MODE (:class:`FitMode`). ``edit`` treats the user's
framing as the contract; ``compose`` (multi-reference composition, ie#600)
treats output geometry as a free parameter and only fits the references.

Super-resolution is a DECLARED, PLUGGABLE post-stage (:func:`set_upscaler`).
No image upscaler exists in the fleet today (ie#599 catalog check: LTX-2.3's
``upscale_2x`` is a video *latent* upsampler), so the stage is a no-op and
:func:`restore` returns native-bucket-sized pixels at the user's framing.
:class:`RestoreResult` reports that honestly rather than faking it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import msgspec

from .api.errors import ValidationError
from .api.types import StringEnum

if TYPE_CHECKING:  # Pillow is a real dependency but lazily imported (see io.py).
    from PIL import Image as _PILImage

    PILImage = _PILImage.Image
else:
    PILImage = Any


__all__ = [
    "FitMode",
    "OutputSize",
    "FamilyGeometry",
    "FitPlan",
    "RestoreResult",
    "nearest_bucket",
    "fit_to_native",
    "restore",
    "set_upscaler",
    "current_upscaler",
]


class FitMode(StringEnum):
    """What the output geometry is FOR."""

    #: The user's framing is the contract: pad in, crop back out.
    EDIT = "edit"
    #: Multi-reference composition: references are fitted, output is free.
    COMPOSE = "compose"


class OutputSize(StringEnum):
    """The caller-facing geometry policy on an edit lane."""

    #: Fit to native for the edit, restore the caller's framing (default).
    MATCH_INPUT = "match_input"
    #: Return the model's bucket unrestored, for callers that post-process.
    NATIVE = "native"
    #: Honour an explicit ``aspect_ratio`` + ``megapixels`` reframe.
    PRESET = "preset"


# A declared row whose area strays outside this band of the family's native
# area is the ie#599 defect in table form; refuse it at declaration time.
_AREA_BAND = (0.75, 1.25)


class FamilyGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """A family's native geometry. DATA, declared by the endpoint package.

    ``buckets`` must be rows that already appear in the endpoint's
    ``Compile(shapes=...)`` set — :func:`fit_to_native` never invents a size,
    so the AOT compile shape set is unchanged by fitting. Assert the
    containment in the endpoint's tests; :meth:`assert_declared` does it.
    """

    name: str
    #: The pixel area the pipeline actually conditions and decodes at.
    native_area: int
    #: The declared grid rows at ``native_area``, as (width, height).
    buckets: tuple[tuple[int, int], ...]
    #: Latent/patch alignment every row must satisfy.
    multiple_of: int = 16

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError(f"{self.name}: FamilyGeometry needs at least one bucket")
        if self.native_area <= 0:
            raise ValueError(f"{self.name}: native_area must be positive")
        lo, hi = _AREA_BAND
        for width, height in self.buckets:
            if width <= 0 or height <= 0:
                raise ValueError(f"{self.name}: bucket {width}x{height} is not positive")
            if width % self.multiple_of or height % self.multiple_of:
                raise ValueError(
                    f"{self.name}: bucket {width}x{height} is not a multiple of "
                    f"{self.multiple_of}"
                )
            ratio = (width * height) / self.native_area
            if not lo <= ratio <= hi:
                raise ValueError(
                    f"{self.name}: bucket {width}x{height} is {ratio:.2f}x the declared "
                    f"native area {self.native_area} — a native table cannot carry a row "
                    f"outside {lo}..{hi}x (ie#599: this is exactly the t2i-table-on-an-"
                    f"edit-lane defect)"
                )

    def assert_declared(self, shapes: Sequence[tuple[int, int]]) -> None:
        """Every bucket must be a declared compile shape. Call from tests."""
        declared = {(int(w), int(h)) for w, h in shapes}
        missing = [b for b in self.buckets if b not in declared]
        if missing:
            raise ValueError(
                f"{self.name}: buckets {missing} are not in the declared compile shape "
                "set — fitting would mint an undeclared graph"
            )


class FitPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Everything the post-stage needs, DERIVED once rather than re-guessed."""

    mode: FitMode
    output_size: OutputSize
    geometry: FamilyGeometry
    #: The declared grid row the model runs at. Condition AND output.
    bucket: tuple[int, int]
    #: The fitted condition images, in the caller's order.
    images: tuple[PILImage, ...]
    #: The primary input's size exactly as submitted.
    source_size: tuple[int, int]
    #: The caller's framing inside ``bucket`` space (left, top, right, bottom).
    crop_box: tuple[int, int, int, int]
    #: What the caller should get back once super-resolution exists.
    target_size: tuple[int, int]
    #: Whether the restored edit should be composited onto the source pixels.
    composite: bool
    #: The primary input at source resolution, kept for the composite.
    source_image: Optional[PILImage] = None
    #: The primary condition cropped to ``crop_box``, for the composite mask.
    condition_native: Optional[PILImage] = None

    @property
    def padded(self) -> bool:
        left, top, right, bottom = self.crop_box
        return (right - left, bottom - top) != self.bucket


class RestoreResult(msgspec.Struct, frozen=True, kw_only=True):
    """What the caller actually got, and what it is missing."""

    image: PILImage
    size: tuple[int, int]
    #: True once a registered upscaler produced ``target_size`` pixels.
    upscaled: bool
    #: True once untouched regions were taken from the source at full detail.
    composited: bool
    note: str


# -- the pluggable super-resolution stage ---------------------------------
# A capability seam, not a toggle: nothing registers an upscaler today, so the
# default is "no super-resolution exists" and restore says so.

Upscaler = Callable[[PILImage, "tuple[int, int]"], "Optional[PILImage]"]

_UPSCALER: Optional[Upscaler] = None


def set_upscaler(upscaler: Optional[Upscaler]) -> Optional[Upscaler]:
    """Register the fleet's image super-resolution stage. Returns the previous."""
    global _UPSCALER
    previous, _UPSCALER = _UPSCALER, upscaler
    return previous


def current_upscaler() -> Optional[Upscaler]:
    return _UPSCALER


# -- snapping --------------------------------------------------------------


def nearest_bucket(
    width: int, height: int, geometry: FamilyGeometry
) -> tuple[int, int]:
    """Snap to the declared row with the closest LOG aspect ratio.

    Log distance, not ``abs(a - w/h)``: the linear form is scale-biased and
    ranks 16:9 against 9:16 differently depending on which way up you ask.
    The input's AREA is not read — the family's native area already fixed it.
    """
    ratio = math.log(max(int(width), 1) / max(int(height), 1))
    return min(
        geometry.buckets,
        key=lambda wh: abs(math.log(wh[0] / wh[1]) - ratio),
    )


def _contain_box(
    size: tuple[int, int], bucket: tuple[int, int], multiple_of: int
) -> tuple[int, int, int, int]:
    """Centre box inside ``bucket`` holding ``size``'s aspect at max scale."""
    width, height = size
    bucket_w, bucket_h = bucket
    scale = min(bucket_w / max(width, 1), bucket_h / max(height, 1))
    inner_w = min(bucket_w, max(multiple_of, int(round(width * scale))))
    inner_h = min(bucket_h, max(multiple_of, int(round(height * scale))))
    left = (bucket_w - inner_w) // 2
    top = (bucket_h - inner_h) // 2
    return (left, top, left + inner_w, top + inner_h)


def _edge_pad(image: PILImage, bucket: tuple[int, int], box: tuple[int, int, int, int]) -> PILImage:
    """Paste ``image`` at ``box`` and fill the margins by smearing the edges.

    Edge replication, not black bars or reflection: bars read as content the
    model will happily edit, reflection invents plausible objects.
    """
    from PIL import Image as PILImageModule

    left, top, right, bottom = box
    inner_w, inner_h = right - left, bottom - top
    if image.size != (inner_w, inner_h):
        image = image.resize((inner_w, inner_h), PILImageModule.Resampling.LANCZOS)
    if (inner_w, inner_h) == bucket:
        return image

    canvas = PILImageModule.new(image.mode, bucket)
    canvas.paste(image, (left, top))
    bucket_w, bucket_h = bucket
    if left > 0:
        canvas.paste(image.crop((0, 0, 1, inner_h)).resize((left, inner_h)), (0, top))
    if right < bucket_w:
        strip = image.crop((inner_w - 1, 0, inner_w, inner_h))
        canvas.paste(strip.resize((bucket_w - right, inner_h)), (right, top))
    if top > 0:
        band = canvas.crop((0, top, bucket_w, top + 1))
        canvas.paste(band.resize((bucket_w, top)), (0, 0))
    if bottom < bucket_h:
        band = canvas.crop((0, bottom - 1, bucket_w, bottom))
        canvas.paste(band.resize((bucket_w, bucket_h - bottom)), (0, bottom))
    return canvas


def fit_to_native(
    images: Sequence[PILImage],
    geometry: FamilyGeometry,
    *,
    mode: FitMode = FitMode.EDIT,
    output_size: OutputSize = OutputSize.MATCH_INPUT,
    preset: Optional[tuple[int, int]] = None,
    primary: int = 0,
) -> FitPlan:
    """Fit condition images to ONE shared native bucket and plan the restore.

    ``preset`` is required by (and only read under) ``OutputSize.PRESET``; it
    must be a declared row unless ``mode`` is ``COMPOSE``, where the caller
    owns output geometry outright.
    """
    if not images:
        raise ValidationError("fit_to_native: at least one image is required")
    mode = FitMode(mode)
    output_size = OutputSize(output_size)
    source = images[primary]
    source_size = (int(source.size[0]), int(source.size[1]))

    if mode is FitMode.COMPOSE:
        # References are fitted to native independently; output geometry is a
        # free parameter the caller supplies (ie#600's reference-composition
        # shape). Nothing is cropped back — there is no "user's framing".
        fitted_refs = []
        for image in images:
            ref_bucket = nearest_bucket(image.size[0], image.size[1], geometry)
            box = _contain_box(image.size, ref_bucket, geometry.multiple_of)
            fitted_refs.append(_edge_pad(image, ref_bucket, box))
        fitted = tuple(fitted_refs)
        if preset is None:
            # ie#600: compose output geometry is a FREE parameter the caller
            # supplies. It must NOT silently inherit references[0]'s aspect —
            # that inheritance is today's undocumented behaviour, not a design.
            raise ValidationError(
                "mode='compose' has no original framing to fit to: the caller "
                "owns output geometry and must pass an explicit bucket"
            )
        out_bucket = preset
        return FitPlan(
            mode=mode,
            output_size=output_size,
            geometry=geometry,
            bucket=(int(out_bucket[0]), int(out_bucket[1])),
            images=fitted,
            source_size=source_size,
            crop_box=(0, 0, int(out_bucket[0]), int(out_bucket[1])),
            target_size=(int(out_bucket[0]), int(out_bucket[1])),
            composite=False,
        )

    if output_size is OutputSize.PRESET:
        if preset is None:
            raise ValidationError("output_size='preset' requires an explicit bucket")
        bucket = (int(preset[0]), int(preset[1]))
        if bucket not in geometry.buckets:
            raise ValidationError(
                f"{geometry.name}: {bucket[0]}x{bucket[1]} is not a declared native row "
                f"for this family's edit lane"
            )
    else:
        bucket = nearest_bucket(source_size[0], source_size[1], geometry)

    # The PRIMARY condition shares the output's bucket — that pairing is the
    # ie#599 fix. Secondary references are content, not framing: each takes
    # its OWN nearest native row, so a 16:9 style reference is not padded 44%
    # into a square. Every row is ~native_area, so all conditions still sit on
    # the same grid scale as the target latent.
    boxes: list[tuple[int, int, int, int]] = []
    fitted_list = []
    for index, image in enumerate(images):
        row = bucket if index == primary else nearest_bucket(
            image.size[0], image.size[1], geometry)
        box = _contain_box(image.size, row, geometry.multiple_of)
        boxes.append(box)
        fitted_list.append(_edge_pad(image, row, box))
    fitted = tuple(fitted_list)

    if output_size is OutputSize.NATIVE:
        crop_box = (0, 0, bucket[0], bucket[1])
        target = bucket
        composite = False
    elif output_size is OutputSize.PRESET:
        # The caller explicitly asked to be reframed; do not crop back.
        crop_box = (0, 0, bucket[0], bucket[1])
        target = bucket
        composite = False
    else:
        crop_box = boxes[primary]
        # Inputs at or below the bucket default to the native bucket rather
        # than true pixels-in == pixels-out; the below-native variant is
        # UNPROVEN and is the A/B arm of the re-baseline run (ie#599 §6).
        larger = source_size[0] * source_size[1] > bucket[0] * bucket[1]
        target = source_size if larger else (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
        composite = larger

    return FitPlan(
        mode=mode,
        output_size=output_size,
        geometry=geometry,
        bucket=bucket,
        images=fitted,
        source_size=source_size,
        crop_box=crop_box,
        target_size=(int(target[0]), int(target[1])),
        composite=composite,
        source_image=source if composite else None,
        condition_native=fitted[primary].crop(crop_box) if composite else None,
    )


def _extrema(mask: PILImage) -> tuple[int, int]:
    """(min, max) of an 8-bit mask, from its histogram (typed, unlike getextrema)."""
    hist: list[int] = mask.histogram()
    used = [level for level, count in enumerate(hist) if count]
    return (used[0], used[-1]) if used else (0, 0)


def _soft_mask(edited: PILImage, condition: PILImage) -> PILImage:
    """Normalized |edit - condition| at native resolution, blurred to a soft mask."""
    from PIL import Image as PILImageModule
    from PIL import ImageChops, ImageFilter

    diff = ImageChops.difference(edited.convert("RGB"), condition.convert("RGB"))
    mask = diff.convert("L").filter(ImageFilter.GaussianBlur(radius=3))
    peak = _extrema(mask)[1]
    if peak <= 8:  # No usable localization: treat the edit as global.
        return PILImageModule.new("L", mask.size, 255)
    scale = 255.0 / peak
    return mask.point(lambda v: min(255, int(v * scale)))


def restore(
    image: PILImage,
    plan: FitPlan,
    *,
    upscaler: Optional[Upscaler] = None,
) -> RestoreResult:
    """Undo the fit: crop the pad box off, then run the super-resolution stage.

    With no upscaler registered — the fleet's state today — the caller gets
    native-bucket-sized pixels at exactly the framing they submitted, and
    ``upscaled``/``composited`` are False. That is the honest result, not a
    LANCZOS enlargement dressed up as detail.
    """
    from PIL import Image as PILImageModule

    if plan.output_size is not OutputSize.MATCH_INPUT or plan.mode is FitMode.COMPOSE:
        size = (int(image.size[0]), int(image.size[1]))
        return RestoreResult(
            image=image, size=size, upscaled=False, composited=False,
            note=f"returned at the model's native bucket {size[0]}x{size[1]}",
        )

    if image.size != plan.bucket:
        raise ValidationError(
            f"restore: result is {image.size[0]}x{image.size[1]}, plan bucket is "
            f"{plan.bucket[0]}x{plan.bucket[1]}"
        )

    framed = image.crop(plan.crop_box) if plan.padded else image
    if framed.size == plan.target_size:
        return RestoreResult(
            image=framed, size=framed.size, upscaled=False, composited=False,
            note=f"restored to the submitted framing at {framed.size[0]}x{framed.size[1]}",
        )

    stage = upscaler if upscaler is not None else _UPSCALER
    enlarged = stage(framed, plan.target_size) if stage is not None else None
    if enlarged is None or enlarged.size != plan.target_size:
        # ie#599: no image super-resolution capability exists in the fleet.
        return RestoreResult(
            image=framed, size=framed.size, upscaled=False, composited=False,
            note=(
                f"edited at native {framed.size[0]}x{framed.size[1]} with the submitted "
                f"framing; the {plan.target_size[0]}x{plan.target_size[1]} source "
                "resolution needs an image upscaler, and none is deployed (ie#599)"
            ),
        )

    if not plan.composite or plan.source_image is None or plan.condition_native is None:
        return RestoreResult(
            image=enlarged, size=enlarged.size, upscaled=True, composited=False,
            note=f"upscaled to {enlarged.size[0]}x{enlarged.size[1]}",
        )

    # Untouched regions keep FULL source detail; edited regions carry the
    # model's. A global edit has no usable mask and returns the upscale.
    mask = _soft_mask(framed, plan.condition_native)
    global_edit = _extrema(mask)[0] >= 250
    source = plan.source_image.convert("RGB")
    if source.size != plan.target_size:
        source = source.resize(plan.target_size, PILImageModule.Resampling.LANCZOS)
    if global_edit:
        return RestoreResult(
            image=enlarged, size=enlarged.size, upscaled=True, composited=False,
            note="global edit: no localized mask, returned the upscaled edit whole",
        )
    blended = PILImageModule.composite(
        enlarged.convert("RGB"), source,
        mask.resize(plan.target_size, PILImageModule.Resampling.BILINEAR),
    )
    return RestoreResult(
        image=blended, size=blended.size, upscaled=True, composited=True,
        note=(
            f"composited the edit back onto the {plan.target_size[0]}x"
            f"{plan.target_size[1]} source; untouched regions keep source detail"
        ),
    )
