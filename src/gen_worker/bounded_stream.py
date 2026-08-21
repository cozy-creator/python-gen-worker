"""The one in-loop implementation of "a stream may not exceed its declared size", so no downloader hand-rolls a fifth copy of it."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

__all__ = [
    "StreamTooLarge",
    "copy_bounded",
    "free_space_bound",
    "DISK_RESERVE_BYTES",
]

DISK_RESERVE_BYTES = 1 << 30


class StreamTooLarge(ValueError):
    """A source sent more bytes than its declaration allows."""

    def __init__(self, what: str, limit_bytes: int, delivered: int) -> None:
        super().__init__(
            f"{what}: source sent more than its {limit_bytes}-byte declared "
            f"size (aborted at {delivered} bytes)"
        )
        self.what = what
        self.limit_bytes = limit_bytes
        self.delivered = delivered


def copy_bounded(
    chunks: Iterable[bytes],
    write: Callable[[bytes], Any],
    *,
    limit_bytes: int,
    what: str,
    hasher: Optional[Any] = None,
    on_bytes: Optional[Callable[[int], None]] = None,
    exceeded: Optional[Callable[[int, int], BaseException]] = None,
) -> int:
    """Copy ``chunks`` into ``write``, refusing at the byte that passes the cap."""
    if limit_bytes <= 0:
        raise ValueError(
            f"{what}: copy_bounded needs a positive byte bound; a caller with "
            "no declared size must derive one (free_space_bound) or refuse"
        )
    total = 0
    for chunk in chunks:
        if not chunk:
            continue
        total += len(chunk)
        if total > limit_bytes:
            if exceeded is not None:
                raise exceeded(limit_bytes, total)
            raise StreamTooLarge(what, limit_bytes, total)
        write(chunk)
        if hasher is not None:
            hasher.update(chunk)
        if on_bytes is not None:
            on_bytes(len(chunk))
    return total


def free_space_bound(path: "Path | str", *, reserve_bytes: int = DISK_RESERVE_BYTES) -> int:
    """The bound of last resort for a transfer whose source declared no size."""
    from .capability import InsufficientDiskError

    root = Path(path)
    try:
        free = int(shutil.disk_usage(root).free)
    except OSError as exc:
        raise InsufficientDiskError(
            f"cannot measure free space at {root}: {exc}", path=str(root)
        ) from exc
    bound = free - reserve_bytes
    if bound <= 0:
        raise InsufficientDiskError(
            f"no room at {root}: {free} bytes free, {reserve_bytes} reserved",
            available_bytes=free,
            required_bytes=reserve_bytes,
            path=str(root),
        )
    return bound
