"""pgw#1612: an ENOSPC is a claim about the SHAPE, not about the attempt.

The hub already knows what to do with `insufficient_disk`: it is a real
model-failure reason with a whole migration path behind it — drop the oldest
resident non-hot disk goal, advance the capacity generation, clear the
failures, re-send desired state. The worker only ever raised it for a DOWNLOAD
that could not fit. An `OSError [Errno 28]` from anywhere else in the boot — a
load, a cache write, an artifact export — propagated as an ordinary exception
and reached the hub as a generic failure, so the hub requeued onto a machine
with the same `container_disk_gb_requested`.

MEASURED (th#2246, 2026-08-21): `qwen-image` 0.5.17 ENOSPC'd on
`8gpqows0j349gm` (A100-SXM4-80GB, 100 GB container disk) and requeued onto
`3zod6pwvn10f4y` — another A100-SXM4-80GB with the same 100 GB — and would have
kept going at $1.59/hr had a human not cancelled it. Deterministic failure,
unbounded retry, real money.

This module is the classifier, and it is deliberately ONE function used at ONE
seam. A per-raiser catch is how half of them stay generic.
"""

from __future__ import annotations

import errno
import shutil
from pathlib import Path
from typing import Any, Iterator, Optional

from ..capability import InsufficientDiskError
from . import disk_telemetry


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """The exception and everything it wraps, depth-bounded.

    ENOSPC arrives wrapped: `shutil.Error` carries the real `OSError` in its
    `args`, a retry wrapper carries it as `__cause__`, and a thread boundary
    carries it as `__context__`. Reading only the outermost type is how a
    deterministic disk failure reads as a generic one.
    """

    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack and len(seen) < 32:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for nested in (current.__cause__, current.__context__):
            if isinstance(nested, BaseException):
                stack.append(nested)
        if isinstance(current, shutil.Error):
            # `shutil.copytree` raises `Error([(src, dst, why), ...])` — the
            # exception is nested two containers deep, and `why` is sometimes
            # the exception and sometimes its formatted string. Flatten the
            # containers; a string `why` is simply not classifiable here, which
            # is the honest outcome (see `out_of_space`: no substring guessing).
            stack.extend(_nested_exceptions(current.args, depth=3))
        group = getattr(current, "exceptions", None)
        if isinstance(group, (list, tuple)):
            stack.extend(e for e in group if isinstance(e, BaseException))


def _nested_exceptions(value: Any, *, depth: int) -> Iterator[BaseException]:
    if isinstance(value, BaseException):
        yield value
        return
    if depth <= 0 or not isinstance(value, (list, tuple, set)):
        return
    for item in value:
        yield from _nested_exceptions(item, depth=depth - 1)


def out_of_space(exc: BaseException) -> Optional[OSError]:
    """The ENOSPC inside ``exc``, or None.

    Only `errno.ENOSPC` counts. Substring matching on the message is
    deliberately NOT done here: `_error_vocab`'s `"disk" in text` heuristic
    already exists on the download path and is exactly the kind of guess that
    mislabels an unrelated failure as a capacity claim the hub then acts on.
    """

    for item in _chain(exc):
        if isinstance(item, OSError) and item.errno == errno.ENOSPC:
            return item
    return None


def _mount_facts(path: Any) -> str:
    """Which mount ran out, and its real statvfs totals.

    `disk_telemetry` measures the real mount points the worker uses, so the
    reason quotes it rather than leaving "the container disk was 100 GB and the
    boot needed 121 GB" to be re-derived by a lane weeks later.
    """

    target = str(path or "") or "."
    totals = disk_telemetry._statvfs_totals(target)
    if totals is None:
        return f"mount={target} statvfs=unreadable"
    total, free = totals
    return f"mount={target} statvfs_total={total} statvfs_free={free}"


def as_insufficient_disk(
    exc: BaseException, *, doing: str, fallback_path: Any = None
) -> Optional[InsufficientDiskError]:
    """Re-type an ENOSPC as the reason the hub already knows how to act on.

    Returns None when ``exc`` is not an out-of-space failure — the caller then
    reports it exactly as before, because inventing a capacity claim out of an
    unrelated error is worse than the generic bucket.
    """

    oserr = out_of_space(exc)
    if oserr is None:
        return None
    where = getattr(oserr, "filename", None) or fallback_path
    facts = _mount_facts(Path(where).parent if where else fallback_path)
    return InsufficientDiskError(
        f"no space left while {doing}: {type(exc).__name__}: {exc}; "
        f"path={where or 'unknown'}; {facts}",
        available_bytes=_free_bytes(where or fallback_path),
        # Unknown: the raiser is a write that failed, not a plan with a total.
        # Zero is the honest answer — "we needed more than there was" — and it
        # must never be confused with a sized shortfall the planner computed.
        required_bytes=0,
        path=str(where or fallback_path or ""),
    )


def _free_bytes(path: Any) -> int:
    try:
        return int(shutil.disk_usage(str(path or ".")).free)
    except OSError:
        return 0


__all__ = ["as_insufficient_disk", "out_of_space"]
