"""Typed download-failure errors (CONTRACT §9 ModelEvent.error vocabulary)."""
from __future__ import annotations


class UrlExpiredError(RuntimeError):
    """A presigned download URL was rejected with a permanent 4xx (expired
    signature, revoked object). Never retried worker-side: the orchestrator
    re-mints fresh URLs on ``ModelEvent{FAILED, error:"url_expired"}``."""

    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


__all__ = ["UrlExpiredError"]
