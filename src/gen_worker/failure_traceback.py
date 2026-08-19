"""pgw#1474 / th#2201 — the traceback tail a failed job ships to the hub.

Until this module existed, the ENTIRE diagnostic surface of a body failure was
the string the catch site formatted::

    status, message = pb.JOB_STATUS_FATAL, f"{type(exc).__name__}: {exc}"

Measured (e2e#1919, on a rented H100): a `z-image-w8a8-quantization` job burned
455 GPU-seconds and $0.501, died inside a dependency, and delivered the five
characters ``'keys'`` to its submitter. No module, no line, no raising library.
The pod was retired 19 seconds later and RunPod has no logs API, so those five
characters were all anybody would ever have. jobs#306 stops exactly there:
deriving even a NEIGHBOURHOOD for that string took an exhaustive ``git grep``
over two repositories and found nothing — which is what a traceback prints on
line one.

The exception object is already in hand at every catch site (the classification
reads its type). This module turns it into something bounded enough to put on a
wire and on a row.

Three properties, and each one is a decision:

1. **The TAIL, never the head.** A recursion blow-up prints thousands of
   identical frames and then the one that matters. ``format_exception``'s
   negative ``limit`` keeps the last frames, and the byte bound cuts from the
   front — so the raising frame survives both.
2. **CREDENTIALS scrubbed, PATHS kept.** `redact.sanitize` would be wrong here:
   its third pattern eats absolute filesystem paths, and
   ``File "/opt/endpoint/jobs/quantize.py", line 118`` IS the diagnosis. What
   is scrubbed is what a presigned URL or an auth header drags into an
   exception message.
3. **It never raises.** A formatter that throws while formatting a failure
   turns a diagnosable job into a silent one, which is the defect it exists to
   fix.
"""

from __future__ import annotations

import traceback
from typing import Optional

from .redact import sanitize_credentials

#: How many stack frames survive. Twenty is deep enough to cross an endpoint
#: body, a framework and a library and still see where it started.
MAX_FRAMES = 20

#: The byte ceiling on the whole formatted tail, marker included. It is the
#: hub's own bound (th#2201, `jobs.MaxTracebackBytes`) so the hub's re-bound —
#: which exists because a peer's arithmetic is not a hub invariant — is a no-op
#: on a worker that is behaving.
MAX_BYTES = 16384

#: Word-for-word the hub's marker. One vocabulary for one fact, and it names
#: the DIRECTION of the cut: a traceback truncated at the head is still useful,
#: one truncated at the tail is not.
TRUNCATED_MARKER = (
    "…(traceback truncated: the head was dropped, the raising frames are below)\n"
)


def traceback_tail(exc: BaseException, *, max_bytes: int = MAX_BYTES) -> str:
    """Format ``exc`` as a bounded, scrubbed traceback tail.

    Returns the empty string only when there is genuinely nothing to format —
    the hub renders that as ``no_traceback_reported`` rather than as silence.
    """
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
    """Keep the last ``max_bytes`` of ``text``, cut on a line boundary.

    Bytes, not characters: the hub bounds the same field in bytes, and a
    traceback carrying a non-ASCII repr would otherwise be measured by two
    different rulers on the two sides of one wire.
    """
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    budget = max_bytes - len(TRUNCATED_MARKER.encode("utf-8"))
    if budget <= 0:  # a caller asked for a bound smaller than the marker
        return TRUNCATED_MARKER
    tail = raw[-budget:]
    # Drop the partial first line. Half a frame is not a frame, and a tail that
    # starts mid-token is a tail nobody can paste into a bug report.
    newline = tail.find(b"\n")
    if 0 <= newline < len(tail) - 1:
        tail = tail[newline + 1 :]
    return TRUNCATED_MARKER + tail.decode("utf-8", "replace")


def traceback_tail_of(exc: Optional[BaseException]) -> str:
    """``traceback_tail`` for a site that may hold no exception at all."""
    return "" if exc is None else traceback_tail(exc)
