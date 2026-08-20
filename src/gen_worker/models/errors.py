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


class NonCasWeightSourceRefused(RuntimeError):
    """Serving was asked to load weights from something that is not a
    tensor-layout-contract-cut tensorfs CAS snapshot.

    Paul's 2026-08-19 hardcut ruling: *"only store + support loading our new
    tensorfs laid out files. Our tensor-layout-contract and tensorfs chunking
    system are mandatory; do not support old systems that lack this."* The
    serving path therefore has exactly ONE weight source — a projected CAS
    snapshot tree (``models/cozy_snapshot.py``) built from an
    orchestrator-resolved manifest.

    Hugging Face, Civitai and ModelScope survive as INGEST edges only:
    fetch -> normalize under a layout contract -> publish to the CAS -> serve
    from the CAS. There is no direct-serve fallback to fall back TO, so this is
    terminal and never retried — retrying cannot make an upstream registry into
    a CAS snapshot. The message names the ingest route, because an operator who
    hits this needs the next action, not the diagnosis.
    """

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message)
        self.provider = str(provider or "")


def non_cas_refusal(*, ref: str, provider: str) -> "NonCasWeightSourceRefused":
    """THE refusal, worded once. Every serve-path source class raises this one
    function's result so the operator reads the same route from every door."""
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
    "NonCasWeightSourceRefused",
    "non_cas_refusal",
    "PickleWeightRefused",
    "PICKLE_WEIGHT_EXTENSIONS",
    "first_pickle_weight_path",
]
