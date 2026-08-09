"""pgw#1013: the one in-loop implementation of "a stream may not exceed its
declared size", so there is not a fifth hand-rolled copy of it.

THE DEFECT CLASS THIS CLOSES. Seven downloaders in this repo stream a remote
body to disk. Three checked the running byte count INSIDE the loop and aborted
at the first excess byte (`models/chunk_cas._fetch_chunk_to_offset`,
`models/cozy_cas._stream`, `input_assets._download`, `url_fetch._read_capped`).
Four wrote the whole body first and compared sizes AFTER the loop ended
(`request_context._download_blob_by_digest`, the cell-fetch whole-file branch,
`models/download._civitai_stream_one`, `request_context/_datasets.
_download_url_streamed`). Same threat, two verdicts.

A check after the loop is not a bound. It is a report on how far past the bound
the process already went: the bytes are on disk, the disk may be full, and the
pod may be dead before the comparison is reached. The declared size is known
before the first byte arrives, so the only place the check belongs is the loop.

WHY `limit_bytes` HAS NO DEFAULT AND MAY NOT BE ZERO. The sibling that was
already in-loop but written `if expected_size and downloaded > expected_size`
shows what a defaultable bound decays into: a manifest that omits its size
yields 0, the guard evaluates false, and the site is unbounded again while
still reading as guarded. Here the caller must produce a positive bound — from
the declaration when it has one, from :func:`free_space_bound` when it does
not, or by refusing the transfer. There is no spelling of "unbounded".
"""

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

#: Left unwritten on the destination filesystem by :func:`free_space_bound`.
#: A transfer that fills a pod's disk to the last byte takes down every other
#: writer on it — the model cache, the output staging dir, the logs.
DISK_RESERVE_BYTES = 1 << 30  # 1 GiB, matching cozy_snapshot's headroom


class StreamTooLarge(ValueError):
    """A source sent more bytes than its declaration allows.

    ``ValueError`` so the download retry loops already in this repo classify it
    as a verification failure (permanent, strike-limited) rather than a
    transport hiccup they should retry forever.
    """

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
    """Copy ``chunks`` into ``write``, refusing at the byte that passes the cap.

    Returns the number of bytes written. Raises before writing the chunk that
    would take the total past ``limit_bytes`` — nothing beyond the cap reaches
    the sink, and the source connection is abandoned by the caller's ``with``.

    ``hasher`` is updated with every accepted chunk, so a caller verifying a
    digest does it in the same pass rather than re-reading the file.
    ``exceeded`` lets a caller keep its own error vocabulary; the default is
    :class:`StreamTooLarge`.
    """
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
    """The bound of last resort for a transfer whose source declared no size.

    Not a policy number: it is the resource that actually runs out, measured on
    the filesystem the bytes are landing on. A transfer allowed to exceed it
    does not merely overrun a budget, it ENOSPCs every other writer on the pod.

    Raises :class:`~gen_worker.capability.InsufficientDiskError` when the
    filesystem is already inside the reserve, because the honest answer there
    is "do not start" rather than a bound of zero.
    """
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
