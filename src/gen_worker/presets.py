"""Shared authoring helpers for diffusion-style endpoints.

These absorb boilerplate that every image endpoint was hand-rolling: dimension
snapping/resolution, seeded RNG, and device detection. The SDK owns the
*logic*; each endpoint owns its model-family *values* — both the set of aspect
ratios it offers AND the pixel buckets they map to — so no resolution opinion
or ratio vocabulary is imposed.

All torch imports are lazy, so importing this module never requires torch.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional


def resolve_device(prefer: str = "cuda") -> str:
    """Return ``prefer`` if that accelerator is available, else ``"cpu"``.

    Replaces the ``"cuda" if torch.cuda.is_available() else "cpu"`` two-liner
    copied into every endpoint.
    """
    try:
        import torch
    except Exception:
        return "cpu"
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return prefer
    mps = getattr(torch.backends, "mps", None)
    if prefer == "mps" and mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def seed_generator(seed: Optional[int], *, device: Optional[str] = None) -> Any:
    """Build a seeded ``torch.Generator`` (or ``None`` when ``seed`` is None).

    Centralizes the seed->generator helper duplicated across endpoints. Returns
    ``None`` for a ``None`` seed so the result can be passed straight to a
    diffusers pipeline's ``generator=`` argument. ``device`` defaults to
    :func:`resolve_device`.
    """
    if seed is None:
        return None
    import torch

    dev = device if device is not None else resolve_device()
    return torch.Generator(device=dev).manual_seed(int(seed))


def snap_to_multiple(
    value: int,
    multiple: int,
    *,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    """Round ``value`` DOWN to the nearest ``multiple``, clamped to [minimum, maximum].

    Latent diffusion models require each spatial dim to be a multiple of the VAE
    downsample factor (commonly surfaced as 16 or 64 at the bucket level).
    """
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    snapped = (int(value) // multiple) * multiple
    snapped = max(minimum, snapped)
    if maximum is not None:
        snapped = min(maximum, snapped)
    return snapped


def resolve_dimensions(
    payload: Any,
    bucket_table: Mapping[Any, tuple[int, int]],
    *,
    multiple: int = 64,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> tuple[int, int]:
    """Resolve ``(width, height)`` from a payload using the endpoint's bucket table.

    Precedence: explicit ``width`` AND ``height`` on the payload win (each
    snapped to ``multiple``); otherwise the payload's ``aspect_ratio`` is looked
    up in ``bucket_table``. Both the set of ratios AND their pixel buckets are
    the endpoint's to declare — this function imposes neither. The enum type is
    inferred from the table's keys, so each endpoint keeps its own
    ``AspectRatio`` (or any key type whatsoever).
    """
    width = getattr(payload, "width", None)
    height = getattr(payload, "height", None)
    if width is not None and height is not None:
        return (
            snap_to_multiple(width, multiple, minimum=minimum, maximum=maximum),
            snap_to_multiple(height, multiple, minimum=minimum, maximum=maximum),
        )

    ar_raw = getattr(payload, "aspect_ratio", None)
    # Coerce the payload value against the TABLE's own key type (the endpoint's
    # enum), so no SDK-owned vocabulary is imposed.
    key: Any = ar_raw
    if bucket_table:
        key_type = type(next(iter(bucket_table)))
        try:
            key = key_type(str(getattr(ar_raw, "value", ar_raw) or "").strip())
        except (ValueError, TypeError):
            allowed = ", ".join(str(k) for k in bucket_table)
            raise ValueError(f"aspect_ratio must be one of: {allowed}")
    if key not in bucket_table:
        allowed = ", ".join(str(k) for k in bucket_table)
        raise ValueError(f"aspect_ratio {key!r} has no bucket; allowed: {allowed}")
    return bucket_table[key]