"""Deferred output materialization — the image encode+upload tail runs AFTER
the handler returns, with the GPU permit already released.

Why: image endpoints call ``ctx.save_image`` INSIDE the handler, so a webp q95
encode (~1.1s for a 1328^2 frame on a loaded host) plus the upload would run
while the request still held the GPU permit. The terminal
``_release_gpu_slot_for_finalize`` cannot just be copied onto ``save_image``:
that release is terminal and once-only, while endpoints save mid-pipeline and in
N-image loops, so the first of N saves would free the permit with GPU work still
to come.

The terminality signal this needs already exists: the HANDLER'S RETURN. The
executor releases the permit right there anyway. So ``save_image`` snapshots
the pixels, hands back the ``ImageAsset`` the handler drops into its output
struct, and the executor drains the queue after the release — encode, C2PA
stamp and upload all land in the slotless tail while the next request denoises.
No endpoint changes, no new protocol, no early release.

Failure modes handled here:

* **Read-back.** Reading a not-yet-materialized field (``size_bytes``,
  ``sha256``, ``inline_bytes``, ...) forces the encode inline. Correct, just
  not overlapped, and it says so in the log.
* **Mutate-after-save.** ``save_image`` snapshots via ``image.copy()`` (a few
  ms of memcpy against a ~1.1s encode), so a handler that keeps painting on
  the PIL object cannot change what gets uploaded.
* **A failing tail.** The drain runs before the OK result is built, so an
  exception in the deferred encode fails the request like any handler error
  instead of reporting OK with a hollow asset.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, List, Optional

from .api.types import Asset, ImageAsset

logger = logging.getLogger(__name__)

#: Asset fields only the completed upload knows. Reading one before the tail
#: runs forces materialization.
PENDING_FIELDS = frozenset({
    "local_path", "mime_type", "size_bytes", "sha256", "blake3",
    "media_id", "download_token", "stream_mode", "inline_bytes",
})


def fill_from(target: Asset, done: Asset) -> None:
    """Copy the materialized upload's fields onto the handle the handler
    already returned (same object identity, so the output struct sees them)."""
    for field in done.__struct_fields__:
        setattr(target, field, getattr(done, field))


class PendingOutput:
    """One deferred encode+upload. Runs at most once, from whichever thread
    gets there first (the drain, or a read-back inside the handler)."""

    __slots__ = ("_materialize", "_lock", "_done", "forced", "ref")

    def __init__(self, ref: str, materialize: Callable[[], None]) -> None:
        self.ref = ref
        self._materialize = materialize
        self._lock = threading.Lock()
        self._done = False
        self.forced = False

    @property
    def done(self) -> bool:
        return self._done

    def run(self, *, forced: bool = False) -> None:
        with self._lock:
            if self._done:
                return
            # Marked done BEFORE running: a read-back from inside the
            # materialization must not recurse, and a raising tail is
            # terminal for this output either way.
            self._done = True
            self.forced = forced
            self._materialize()


class DeferredImageAsset(ImageAsset, dict=True):
    """``ImageAsset`` whose bytes-attestation fields materialize in the
    finalize tail. ``ref``/``owner`` are known eagerly (every live endpoint
    only ever puts the handle in its output struct); the rest come from the
    upload. Reading one of those early forces the encode inline."""

    def __getattribute__(self, name: str) -> Any:
        if name in PENDING_FIELDS:
            pending: Optional[PendingOutput] = object.__getattribute__(
                self, "__dict__").get("_gw_pending")
            if pending is not None and not pending.done:
                logger.warning(
                    "deferred output %s: handler read %r before the finalize "
                    "tail — encoding inline, so this request loses the "
                    "encode/upload overlap", pending.ref, name)
                pending.run(forced=True)
        return object.__getattribute__(self, name)


class DeferredTail:
    """One request's queue of deferred output materializations."""

    __slots__ = ("_lock", "_queue", "armed")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: List[PendingOutput] = []
        # Armed by the executor only where a post-handler tail exists: a
        # streaming handler serializes items MID-handler, so its outputs can
        # never be deferred. CLI runs and endpoint unit tests stay eager.
        self.armed = False

    def defer(self, pending: PendingOutput) -> None:
        with self._lock:
            self._queue.append(pending)

    def pending(self) -> int:
        with self._lock:
            return sum(1 for p in self._queue if not p.done)

    def drain(self) -> int:
        """Materialize everything still pending, in save order. Returns how
        many ran HERE (i.e. actually overlapped); raises the first failure."""
        with self._lock:
            queue = list(self._queue)
        ran = 0
        for pending in queue:
            if pending.done:
                continue
            pending.run()
            ran += 1
        return ran


__all__ = [
    "DeferredImageAsset",
    "DeferredTail",
    "PendingOutput",
    "PENDING_FIELDS",
    "fill_from",
]
