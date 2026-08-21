from __future__ import annotations

import traceback
from typing import Optional

from .redact import sanitize_credentials

MAX_FRAMES = 20

MAX_BYTES = 16384

TRUNCATED_MARKER = (
    "…(traceback truncated: the head was dropped, the raising frames are below)\n"
)


def traceback_tail(exc: BaseException, *, max_bytes: int = MAX_BYTES) -> str:
    """Format ``exc`` as a bounded, scrubbed traceback tail."""
    try:
        frames = traceback.format_exception(
            type(exc), exc, exc.__traceback__, limit=-MAX_FRAMES
        )
    except Exception:  # noqa: BLE001 — see the module docstring's property 3
        try:
            return sanitize_credentials(f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001
            return ""
    return _bounded(sanitize_credentials("".join(frames).rstrip()), max_bytes)


def _bounded(text: str, max_bytes: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    budget = max_bytes - len(TRUNCATED_MARKER.encode("utf-8"))
    if budget <= 0:
        return TRUNCATED_MARKER
    tail = raw[-budget:]
    newline = tail.find(b"\n")
    if 0 <= newline < len(tail) - 1:
        tail = tail[newline + 1 :]
    return TRUNCATED_MARKER + tail.decode("utf-8", "replace")


def traceback_tail_of(exc: Optional[BaseException]) -> str:
    """``traceback_tail`` for a site that may hold no exception at all."""
    return "" if exc is None else traceback_tail(exc)
