"""Fit an image workload to a family's NATIVE geometry."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import msgspec

from .api.errors import ValidationError

if TYPE_CHECKING:
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


class FitMode(StrEnum):
    """What the output geometry is FOR."""

    EDIT = "edit"
    COMPOSE = "compose"


class OutputSize(StrEnum):
    """The caller-facing geometry policy on an edit lane."""

    MATCH_INPUT = "match_input"
    NATIVE = "native"
    PRESET = "preset"


_AREA_BAND = (0.75, 1.25)


class FamilyGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """A family's native geometry."""

    name: str
    native_area: int
    buckets: tuple[tuple[int, int], ...]
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
        """Every bucket must be a declared compile shape."""
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
    bucket: tuple[int, int]
    images: tuple[PILImage, ...]
    source_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    target_size: tuple[int, int]
    composite: bool
    source_image: Optional[PILImage] = None
    condition_native: Optional[PILImage] = None

    @property
    def padded(self) -> bool:
        left, top, right, bottom = self.crop_box
        return (right - left, bottom - top) != self.bucket


class RestoreResult(msgspec.Struct, frozen=True, kw_only=True):
    """What the caller actually got, and what it is missing."""

    image: PILImage
    size: tuple[int, int]
    upscaled: bool
    composited: bool
    note: str


Upscaler = Callable[[PILImage, "tuple[int, int]"], "Optional[PILImage]"]

_UPSCALER: Optional[Upscaler] = None


def set_upscaler(upscaler: Optional[Upscaler]) -> Optional[Upscaler]:
    """Register the fleet's image super-resolution stage."""
    global _UPSCALER
    previous, _UPSCALER = _UPSCALER, upscaler
    return previous


def current_upscaler() -> Optional[Upscaler]:
    return _UPSCALER


def nearest_bucket(
    width: int, height: int, geometry: FamilyGeometry
) -> tuple[int, int]:
    """Snap to the declared row with the closest LOG aspect ratio."""
    ratio = math.log(max(int(width), 1) / max(int(height), 1))
    return min(
        geometry.buckets,
        key=lambda wh: abs(math.log(wh[0] / wh[1]) - ratio),
    )


def _contain_box(
    size: tuple[int, int], bucket: tuple[int, int], multiple_of: int
) -> tuple[int, int, int, int]:
    width, height = size
    bucket_w, bucket_h = bucket
    scale = min(bucket_w / max(width, 1), bucket_h / max(height, 1))
    inner_w = min(bucket_w, max(multiple_of, int(round(width * scale))))
    inner_h = min(bucket_h, max(multiple_of, int(round(height * scale))))
    left = (bucket_w - inner_w) // 2
    top = (bucket_h - inner_h) // 2
    return (left, top, left + inner_w, top + inner_h)


def _edge_pad(image: PILImage, bucket: tuple[int, int], box: tuple[int, int, int, int]) -> PILImage:
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
    """Fit condition images to ONE shared native bucket and plan the restore."""
    if not images:
        raise ValidationError("fit_to_native: at least one image is required")
    mode = FitMode(mode)
    output_size = OutputSize(output_size)
    source = images[primary]
    source_size = (int(source.size[0]), int(source.size[1]))

    if mode is FitMode.COMPOSE:
        fitted_refs = []
        for image in images:
            ref_bucket = nearest_bucket(image.size[0], image.size[1], geometry)
            box = _contain_box(image.size, ref_bucket, geometry.multiple_of)
            fitted_refs.append(_edge_pad(image, ref_bucket, box))
        fitted = tuple(fitted_refs)
        if preset is None:
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
        crop_box = (0, 0, bucket[0], bucket[1])
        target = bucket
        composite = False
    else:
        crop_box = boxes[primary]
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
    hist: list[int] = mask.histogram()
    used = [level for level, count in enumerate(hist) if count]
    return (used[0], used[-1]) if used else (0, 0)


def _soft_mask(edited: PILImage, condition: PILImage) -> PILImage:
    from PIL import Image as PILImageModule
    from PIL import ImageChops, ImageFilter

    diff = ImageChops.difference(edited.convert("RGB"), condition.convert("RGB"))
    mask = diff.convert("L").filter(ImageFilter.GaussianBlur(radius=3))
    peak = _extrema(mask)[1]
    if peak <= 8:
        return PILImageModule.new("L", mask.size, 255)
    scale = 255.0 / peak
    return mask.point(lambda v: min(255, int(v * scale)))


def restore(
    image: PILImage,
    plan: FitPlan,
    *,
    upscaler: Optional[Upscaler] = None,
) -> RestoreResult:
    """Undo the fit: crop the pad box off, then run the super-resolution stage."""
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
