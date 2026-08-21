"""Deferred output materialization — the image encode+upload tail runs AFTER the handler returns, with the GPU permit already released."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, List, Optional

from .api.types import Asset, ImageAsset

logger = logging.getLogger(__name__)

PENDING_FIELDS = frozenset({
    "local_path", "mime_type", "size_bytes", "sha256", "blake3",
    "media_id", "download_token", "stream_mode", "inline_bytes",
})


def fill_from(target: Asset, done: Asset) -> None:
    """Copy the materialized upload's fields onto the handle the handler already returned (same object identity, so the output struct sees them)."""
    for field in done.__struct_fields__:
        setattr(target, field, getattr(done, field))


class PendingOutput:
    """One deferred encode+upload."""

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
            self._done = True
            self.forced = forced
            self._materialize()


class DeferredImageAsset(ImageAsset, dict=True):
    """``ImageAsset`` whose bytes-attestation fields materialize in the finalize tail."""

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
        self.armed = False

    def defer(self, pending: PendingOutput) -> None:
        with self._lock:
            self._queue.append(pending)

    def pending(self) -> int:
        with self._lock:
            return sum(1 for p in self._queue if not p.done)

    def drain(self) -> int:
        """Materialize everything still pending, in save order."""
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
