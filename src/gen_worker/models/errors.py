"""Typed download-failure errors (CONTRACT §9 ModelEvent.error vocabulary)."""
from __future__ import annotations

from typing import Iterable


class UrlExpiredError(RuntimeError):
    """A presigned download URL was rejected with a permanent 4xx (expired signature, revoked object)."""

    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


class MissingSnapshotError(RuntimeError):
    """A tensorhub-CAS ref cannot be materialized: no orchestrator-resolved snapshot was provided, cached, or previously seen. Deterministic local condition — never retried worker-side; the orchestrator re-mints and re-sends DOWNLOAD on ModelEvent{FAILED, error:"missing_snapshot"}."""


class NonCasWeightSourceRefused(RuntimeError):
    """Serving was asked to load weights from something that is not a tensor-layout-contract-cut tensorfs CAS snapshot."""

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message)
        self.provider = str(provider or "")


def non_cas_refusal(*, ref: str, provider: str) -> "NonCasWeightSourceRefused":
    """THE refusal, worded once."""
    return NonCasWeightSourceRefused(
        f"refusing to serve {ref!r} from {provider!r}: serving loads ONLY "
        "tensor-layout-contract-cut tensorfs CAS snapshots. "
        f"{provider!r} is an INGEST source — clone it into the platform "
        "(conversion endpoint / `cozy clone`), which normalizes it under a "
        "layout contract and publishes it to the tensorhub CAS, then bind the "
        "resulting tensorhub ref. There is no direct-serve path.",
        provider=provider,
    )


class PickleWeightRefused(RuntimeError):
    """A resolved snapshot contains a pickle-format weight (.bin/.ckpt/.pt/ .pth/.pkl/.pickle)."""


PICKLE_WEIGHT_EXTENSIONS = (".bin", ".ckpt", ".pt", ".pth", ".pkl", ".pickle")


def first_pickle_weight_path(paths: Iterable[str]) -> str:
    """The first pickle-format weight in ``paths``, or "" if there is none."""
    for raw in paths:
        base = str(raw or "").strip().lower().rsplit("/", 1)[-1]
        if base.endswith(PICKLE_WEIGHT_EXTENSIONS):
            return raw
    return ""


__all__ = [
    "UrlExpiredError",
    "MissingSnapshotError",
    "NonCasWeightSourceRefused",
    "non_cas_refusal",
    "PickleWeightRefused",
    "PICKLE_WEIGHT_EXTENSIONS",
    "first_pickle_weight_path",
]
