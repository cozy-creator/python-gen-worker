"""Typed download-failure errors (CONTRACT §9 ModelEvent.error vocabulary)."""
from __future__ import annotations


class UrlExpiredError(RuntimeError):
    """A presigned download URL was rejected with a permanent 4xx (expired
    signature, revoked object). Never retried worker-side: the orchestrator
    re-mints fresh URLs on ``ModelEvent{FAILED, error:"url_expired"}``."""

    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


class MissingSnapshotError(RuntimeError):
    """A tensorhub-CAS ref cannot be materialized: no orchestrator-resolved
    snapshot was provided, cached, or previously seen. Deterministic
    local condition — never retried worker-side; the orchestrator re-mints and
    re-sends DOWNLOAD on ``ModelEvent{FAILED, error:"missing_snapshot"}``."""


class PickleWeightRefused(RuntimeError):
    """A resolved snapshot contains a pickle-format weight (.bin/.ckpt/.pt/
    .pth/.pkl/.pickle). Unpickling is arbitrary code execution in THIS process,
    which holds hub credentials and other tenants' work, so the snapshot is
    refused at resolve time and its bytes are never downloaded.

    Terminal, never retried: the artifact must be republished as safetensors.
    tensorhub refuses these at publish; this is the defence in depth
    for blobs that predate that refusal or reach a worker by another path."""


__all__ = ["UrlExpiredError", "MissingSnapshotError", "PickleWeightRefused"]
