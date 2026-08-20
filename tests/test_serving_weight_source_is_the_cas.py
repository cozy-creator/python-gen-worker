"""The serving path has ONE weight source, and it is the tensorfs CAS.

# pgw#1524: Paul's hardcut — "only store + support loading our new tensorfs laid
# out files ... do not support old systems that lack this. Hardcut."

Two things are proven here:

1.  **Every door refuses, by source class.** Three doors — the ModelStore
    funnel, the hub-less CLI resolver, and the job plane's reserved-repo
    materializer — crossed with three source classes.

2.  **The end-to-end shape, at CPU scale.** One synthesized safetensors file.
    Fetch-shaped (an upstream ref, no snapshot) it is REFUSED; ingested — put
    in the CAS, named by a resolved manifest, projected — the very same bytes
    serve, and the tensor reads back byte-exact out of the projected tree.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, List, Tuple

import pytest

import projection_fixture
import gen_worker.models.cozy_snapshot as snap_mod
from gen_worker._vendor.tensorfs import CASRef
from gen_worker.api.errors import ValidationError
from gen_worker.models import provision
from gen_worker.models.download import ensure_local, set_provider_index
from gen_worker.models.errors import (
    MissingSnapshotError,
    NonCasWeightSourceRefused,
)
from gen_worker.models.projection import require_projection
from gen_worker.models.refs import WireRef
from gen_worker.models.store import ModelStore
from gen_worker.models.tensor_source import load_state_dict
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.serving.reserved_repos import materialize_reserved_inputs_async
from gen_worker.transfer.grants import TransferReport

#: One ref per SOURCE CLASS, in each class's own grammar. Parametrizing the
#: doors over this is what makes "per source class" a real axis rather than
#: three copies of the huggingface case.
SOURCE_CLASSES: Tuple[Tuple[str, str], ...] = (
    ("hf", "org/model"),
    ("civitai", "123456"),
    ("modelscope", "org/model"),
)


# ---------------------------------------------------------------------------
# 1. Every door refuses, per source class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider,ref", SOURCE_CLASSES)
def test_the_model_store_funnel_refuses_an_upstream_source(
    provider: str, ref: str
) -> None:
    """Door 1: ``models.download.ensure_local`` — the free function both the
    ModelStore funnel and the job plane enter through."""
    with pytest.raises(NonCasWeightSourceRefused) as caught:
        asyncio.run(ensure_local(ref, provider=provider))
    message = str(caught.value)
    assert caught.value.provider == provider
    assert provider in message
    assert "ingest" in message.lower(), (
        f"the refusal must name the ROUTE, not just the diagnosis: {message}")
    assert "tensorhub" in message.lower(), (
        f"the refusal must say what to bind instead: {message}")


@pytest.mark.parametrize("provider,ref", SOURCE_CLASSES)
def test_the_hubless_cli_resolver_refuses_an_upstream_source(
    provider: str, ref: str
) -> None:
    """Door 2: ``provision.resolve_local_path`` — what ``gen-worker run`` /
    ``serve`` / ``up`` drive, which is also cozy-local's door."""
    with pytest.raises(provision.ModelResolutionError) as caught:
        provision.resolve_local_path(
            ref=ref, provider=provider, offline=False, emit=lambda _e: None
        )
    assert "ingest" in str(caught.value).lower()


@pytest.mark.parametrize("provider,ref", SOURCE_CLASSES)
def test_the_hubless_cli_resolver_refuses_offline_too(
    provider: str, ref: str
) -> None:
    """``--offline`` was its own direct-serve rung: it served whatever the HF
    cache happened to hold. A cached upstream snapshot is still not a CAS
    snapshot, so it refuses on the same rule."""
    with pytest.raises(provision.ModelResolutionError):
        provision.resolve_local_path(
            ref=ref, provider=provider, offline=True, emit=lambda _e: None
        )


class _Ctx:
    """A producer context that WOULD accept the path, so a refusal is the
    guard's and not an accident of the fixture."""

    def __init__(self) -> None:
        self.source_path: str = ""
        self.cancelled = False

    def _set_source_path(self, path: str) -> None:
        self.source_path = path

    def raise_if_cancelled(self, _why: str) -> None:
        return None


class _Payload:
    def __init__(self, ref: str) -> None:
        self.source = {"ref": ref}


def test_the_job_planes_reserved_repo_refuses_an_upstream_bound_repo() -> None:
    """Door 3: the reserved-repo materializer.

    Its ref is normalized through the tensorhub grammar (pgw#1217), so the
    SOURCE CLASS arrives from the endpoint.lock binding index rather than from
    the ref's spelling — which is why this door is driven through that index.
    A repo the index says came from an upstream registry, with no resolved
    snapshot, used to fall through to a provider-direct download. Now it
    refuses.
    """
    set_provider_index({"org/mirrored": "hf"})
    try:
        ctx = _Ctx()
        with pytest.raises(NonCasWeightSourceRefused):
            asyncio.run(
                materialize_reserved_inputs_async(ctx, _Payload("org/mirrored"), {})
            )
        assert ctx.source_path == "", (
            "the refusal must land BEFORE the producer is handed a path")
    finally:
        set_provider_index({})


def test_the_job_planes_reserved_repo_keeps_the_hubs_own_failure_distinct() -> None:
    """The SAME door, one axis over: a tensorhub-bound repo the hub simply did
    not resolve is the hub owing a resolve, NOT an unservable source class.
    Collapsing the two would send an operator down the ingest route for what is
    really a resolve outage — so the types must stay apart at this door too."""
    set_provider_index({})
    ctx = _Ctx()
    with pytest.raises(MissingSnapshotError):
        asyncio.run(materialize_reserved_inputs_async(ctx, _Payload("acme/model@1"), {}))
    assert ctx.source_path == ""


def test_a_tensorhub_ref_without_a_snapshot_is_a_DIFFERENT_refusal() -> None:
    """The two conditions must stay distinguishable by TYPE. A tensorhub ref
    with no snapshot is the orchestrator owing a resolve (retryable there); an
    upstream ref is a thing that can never be served (terminal). Collapsing
    them would send an operator to the ingest route for a hub outage."""
    with pytest.raises(MissingSnapshotError):
        asyncio.run(ensure_local("acme/model@1", provider="tensorhub"))
    assert not issubclass(MissingSnapshotError, NonCasWeightSourceRefused)
    assert not issubclass(NonCasWeightSourceRefused, MissingSnapshotError)


def test_an_unknown_provider_is_still_an_input_error_not_a_source_refusal() -> None:
    """The refusal must not swallow a typo. An unparseable provider is bad
    input (``ValidationError`` -> INVALID), which is a different verdict from
    "this source class cannot be served"."""
    with pytest.raises((ValidationError, ValueError)):
        asyncio.run(ensure_local("whatever", provider="nosuchregistry"))


# ---------------------------------------------------------------------------
# 2. End to end: the SAME bytes refuse direct and serve after ingest
# ---------------------------------------------------------------------------

_PAYLOAD = projection_fixture.varied(64, seed=1524)
_WEIGHTS = projection_fixture.safetensors_bytes(
    {"model.weight": ("F32", (4, 4), _PAYLOAD)}
)
_HEX = hashlib.sha256(_WEIGHTS).hexdigest()
_SNAPSHOT_DIGEST = "5f" * 32


def _serve_bytes_from_a_platform_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the transfer: the ingest already put these bytes in the
    platform's object store, so the worker's fetch resolves them. The CAS,
    manifest, projection and read-back below are all REAL."""

    def _download(grants: Any, cas: Any, *, progress: Any = None) -> TransferReport:
        for grant in grants:
            cas.put_bytes(_WEIGHTS, expected=grant.digest)
            if progress is not None:
                progress(grant.digest, grant.size_bytes)
        return TransferReport(
            examined=len(grants),
            succeeded=len(grants),
            bytes_transferred=sum(g.size_bytes for g in grants),
        )

    monkeypatch.setattr(snap_mod, "download", _download)


def test_a_fetched_model_serves_ONLY_after_it_has_been_ingested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One artifact, two routes, opposite verdicts.

    Route A — fetch-shaped: an upstream ref with no resolved snapshot, which is
    exactly what "point the worker at the HF repo" produced before the cut.
    REFUSED.

    Route B — ingested: the same bytes named by a resolved manifest, pulled
    into the CAS and projected. SERVES, and the tensor reads back byte-exact
    THROUGH the projection — which is the part that proves the tree is a
    tensorfs projection and not a plain directory of files.
    """
    sent: List[Any] = []

    async def _emit(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    cas_root = tmp_path / "cas"
    store = ModelStore(_emit, cache_dir=cas_root)

    # Route A ---------------------------------------------------------------
    with pytest.raises(NonCasWeightSourceRefused):
        asyncio.run(ensure_local("acme/tiny-model", provider="hf"))

    # Route B ---------------------------------------------------------------
    _serve_bytes_from_a_platform_ingest(monkeypatch)
    snapshot = pb.Snapshot(
        digest=_SNAPSHOT_DIGEST,
        files=[
            pb.SnapshotFile(
                path="model.safetensors",
                size_bytes=len(_WEIGHTS),
                digest="sha256:" + _HEX,
                url="https://tensorhub.invalid/ingested-blob",
            )
        ],
    )
    tree = asyncio.run(store.ensure_local(WireRef("acme/tiny-model"), snapshot))

    # It is a PROJECTION, not a directory of real files: the container on disk
    # is a pointer stub and the bytes live in the CAS.
    projected = require_projection(tree, why="pgw#1524 end-to-end proof")
    container = Path(tree) / "model.safetensors"
    assert container.stat().st_size < len(_WEIGHTS), (
        "a projected tensor container is a ~128B stub; a full-size file here "
        "means the tree was materialized, not projected")
    assert projected.cas.contains(CASRef(_HEX), size=len(_WEIGHTS))

    # And it READS: the tensor comes back byte-exact through the CAS.
    state = load_state_dict(container, why="pgw#1524 end-to-end proof")
    assert set(state) == {"model.weight"}
    assert state["model.weight"].numpy().tobytes() == _PAYLOAD, (
        "the tensor must come back BYTE-EXACT out of the CAS; a varied payload "
        "is what makes a wrong offset visible")


def test_the_end_to_end_proof_would_notice_a_silently_empty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The e2e test's own red-arm: if the ingest route produced nothing, the
    assertions above must fail rather than pass over an absent tree."""
    _serve_bytes_from_a_platform_ingest(monkeypatch)
    empty = tmp_path / "cas" / "snapshots" / "deadbeef"
    empty.mkdir(parents=True)
    with pytest.raises(Exception):
        require_projection(empty, why="red-arm")
