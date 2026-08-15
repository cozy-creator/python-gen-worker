"""Typed download-failure errors (CONTRACT §9 ModelEvent.error vocabulary)."""
from __future__ import annotations

from typing import Iterable


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


# ONE home for the extension set, beside the refusal it feeds (pgw#1273).
#
# There were five independent copies of this list in the tree and they had
# already drifted: convert/writer.py carried four entries where every other
# copy carried six, so a component holding only `weights.pkl` yielded ZERO
# tensors instead of raising. A list that decides whether arbitrary code runs
# must not be re-typed per call site.
PICKLE_WEIGHT_EXTENSIONS = (".bin", ".ckpt", ".pt", ".pth", ".pkl", ".pickle")


def first_pickle_weight_path(paths: Iterable[str]) -> str:
    """The first pickle-format weight in ``paths``, or "" if there is none.

    Matches on the BASENAME so a directory component that happens to end in a
    pickle extension cannot mask a real one further along the path.
    """
    for raw in paths:
        base = str(raw or "").strip().lower().rsplit("/", 1)[-1]
        if base.endswith(PICKLE_WEIGHT_EXTENSIONS):
            return raw
    return ""


__all__ = [
    "UrlExpiredError",
    "MissingSnapshotError",
    "PickleWeightRefused",
    "PICKLE_WEIGHT_EXTENSIONS",
    "first_pickle_weight_path",
]
