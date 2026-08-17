"""K9's packed shape axis, written once for the families that need it.

``Bucket`` takes a closed set of positive INTEGERS and a runner's variants are
the CROSS PRODUCT of its axes, so a family whose graph classes are a SET of
(width, height) pairs cannot spell them as two axes: nine SDXL shapes would
demand 81 traced classes, seven ERNIE presets 49, fourteen Qwen-Image presets
196. One packed decimal integer per pair is total and exact instead.

The packing is decimal on purpose: ``12000896`` reads as "1200 by 0896" in a
generated ``Literal``, which is the only place a human meets it. Endpoint
handlers never spell one — a :class:`~gen_worker.model.spec.BucketMap` binds the
payload's preset enum onto the axis.

⚠️ This is a WORKAROUND and pgw#1346 K9 says so: the real fix is a
tuple-valued bucket axis (``spec.py`` + codegen + the export digest).
``sdxl_serve.pack_shape`` is the elder copy of this arithmetic and predates
this module; folding it onto these functions is mechanical and belongs to
whoever takes K9's real fix, not to a family-authoring lane that would have to
re-key nothing to do it.
"""

from __future__ import annotations

from typing import Final

#: The packing base. A height below it round-trips exactly, which is the whole
#: property the encoding needs.
BASE: Final = 10000


def pack_shape(width: int, height: int) -> int:
    """One (width, height) pair as ONE bucket-axis integer."""

    if not 0 < height < BASE:
        raise ValueError(f"height {height} is outside the packing's range")
    if width <= 0:
        raise ValueError(f"width {width} must be positive")
    return width * BASE + height


def unpack_shape(code: int) -> tuple[int, int]:
    """The (width, height) pair one packed bucket value names."""

    return divmod(code, BASE)


def shape_buckets(shapes: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    """One preset grid as the sorted, deduplicated axis a ``Bucket`` declares.

    Deduplicated because two preset tiers may name the same pixel size (Qwen's
    1 MP grid and its edit lane both reach 1024x1024) and a bucket axis is a
    SET; sorted because ``Bucket`` refuses anything else.
    """

    return tuple(sorted({pack_shape(width, height) for width, height in shapes}))


def latent_shape(code: int, stride: int) -> tuple[int, int]:
    """Latent ``(rows, cols)`` for one packed shape at one spatial stride.

    Rows before columns, because that is the order the tensor carries them and
    the transposition is the defect this function exists to prevent: 1200x896
    and 896x1200 are two genuinely different conv graphs, so getting the order
    wrong picks the wrong compiled class rather than producing a wrong image.
    """

    width, height = unpack_shape(code)
    return height // stride, width // stride


__all__ = [
    "BASE",
    "latent_shape",
    "pack_shape",
    "shape_buckets",
    "unpack_shape",
]
