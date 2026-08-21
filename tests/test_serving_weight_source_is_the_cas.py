"""The serving path has ONE weight source, and it is the tensorfs CAS."""

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

SOURCE_CLASSES: Tuple[Tuple[str, str], ...] = (
    ("hf", "org/model"),
    ("civitai", "123456"),
    ("modelscope", "org/model"),
)


@pytest.mark.parametrize("provider,ref", SOURCE_CLASSES)
def test_the_model_store_funnel_refuses_an_upstream_source(
    provider: str, ref: str
) -> None:
    """Door 1: ``models.download.ensure_local`` — the free function both the ModelStore funnel and the job plane enter through."""
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
    """Door 2: ``provision.resolve_local_path`` — what ``gen-worker run`` / ``serve`` / ``up`` drive, which is also cozy-local's door."""
    with pytest.raises(provision.ModelResolutionError) as caught:
        provision.resolve_local_path(
            ref=ref, provider=provider, offline=False, emit=lambda _e: None
        )
    assert "ingest" in str(caught.value).lower()


@pytest.mark.parametrize("provider,ref", SOURCE_CLASSES)
def test_the_hubless_cli_resolver_refuses_offline_too(
    provider: str, ref: str
) -> None:
    """``--offline`` was its own direct-serve rung: it served whatever the HF cache happened to hold."""
    with pytest.raises(provision.ModelResolutionError):
        provision.resolve_local_path(
            ref=ref, provider=provider, offline=True, emit=lambda _e: None
        )


class _Ctx:

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
    """Door 3: the reserved-repo materializer."""
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
    """The SAME door, one axis over: a tensorhub-bound repo the hub simply did not resolve is the hub owing a resolve, NOT an unservable source class."""
    set_provider_index({})
    ctx = _Ctx()
    with pytest.raises(MissingSnapshotError):
        asyncio.run(materialize_reserved_inputs_async(ctx, _Payload("acme/model@1"), {}))
    assert ctx.source_path == ""


def test_a_tensorhub_ref_without_a_snapshot_is_a_DIFFERENT_refusal() -> None:
    """The two conditions must stay distinguishable by TYPE."""
    with pytest.raises(MissingSnapshotError):
        asyncio.run(ensure_local("acme/model@1", provider="tensorhub"))
    assert not issubclass(MissingSnapshotError, NonCasWeightSourceRefused)
    assert not issubclass(NonCasWeightSourceRefused, MissingSnapshotError)


def test_an_unknown_provider_is_still_an_input_error_not_a_source_refusal() -> None:
    """The refusal must not swallow a typo."""
    with pytest.raises((ValidationError, ValueError)):
        asyncio.run(ensure_local("whatever", provider="nosuchregistry"))


_PAYLOAD = projection_fixture.varied(64, seed=1524)
_WEIGHTS = projection_fixture.safetensors_bytes(
    {"model.weight": ("F32", (4, 4), _PAYLOAD)}
)
_HEX = hashlib.sha256(_WEIGHTS).hexdigest()
_SNAPSHOT_DIGEST = "5f" * 32


def _serve_bytes_from_a_platform_ingest(monkeypatch: pytest.MonkeyPatch) -> None:

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
    """One artifact, two routes, opposite verdicts."""
    sent: List[Any] = []

    async def _emit(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    cas_root = tmp_path / "cas"
    store = ModelStore(_emit, cache_dir=cas_root)

    with pytest.raises(NonCasWeightSourceRefused):
        asyncio.run(ensure_local("acme/tiny-model", provider="hf"))

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

    projected = require_projection(tree, why="pgw#1524 end-to-end proof")
    container = Path(tree) / "model.safetensors"
    assert container.stat().st_size < len(_WEIGHTS), (
        "a projected tensor container is a ~128B stub; a full-size file here "
        "means the tree was materialized, not projected")
    assert projected.cas.contains(CASRef(_HEX), size=len(_WEIGHTS))

    state = load_state_dict(container, why="pgw#1524 end-to-end proof")
    assert set(state) == {"model.weight"}
    assert state["model.weight"].numpy().tobytes() == _PAYLOAD, (
        "the tensor must come back BYTE-EXACT out of the CAS; a varied payload "
        "is what makes a wrong offset visible")


def test_the_end_to_end_proof_would_notice_a_silently_empty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The e2e test's own red-arm: if the ingest route produced nothing, the assertions above must fail rather than pass over an absent tree."""
    _serve_bytes_from_a_platform_ingest(monkeypatch)
    empty = tmp_path / "cas" / "snapshots" / "deadbeef"
    empty.mkdir(parents=True)
    with pytest.raises(Exception):
        require_projection(empty, why="red-arm")
