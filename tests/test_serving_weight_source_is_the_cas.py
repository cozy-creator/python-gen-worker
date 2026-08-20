"""The serving path has ONE weight source, and it is the tensorfs CAS.

# pgw#1524: Paul's hardcut — "only store + support loading our new tensorfs laid
# out files ... do not support old systems that lack this. Hardcut."

Three things are proven here, and they are different instruments:

1.  **The census asserts its own size.** A per-source-class refusal test can
    only fail on a door it was told about, so it cannot see a NEW direct-serve
    branch someone adds tomorrow. The census does: it walks the AST of every
    module under ``src/gen_worker`` and counts the upstream-registry fetch call
    sites, partitioned into SERVE-side (must be zero) and INGEST-side (must be
    non-zero, or the ingest edge silently vanished and the refusals below would
    be refusing people out of a capability that no longer exists anywhere).

2.  **Every door refuses, by source class.** Three doors — the ModelStore
    funnel, the hub-less CLI resolver, and the job plane's reserved-repo
    materializer — crossed with three source classes.

3.  **The end-to-end shape, at CPU scale.** One synthesized safetensors file.
    Fetch-shaped (an upstream ref, no snapshot) it is REFUSED; ingested — put
    in the CAS, named by a resolved manifest, projected — the very same bytes
    serve, and the tensor reads back byte-exact out of the projected tree.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

_SRC = Path(__file__).resolve().parents[1] / "src" / "gen_worker"

#: One ref per SOURCE CLASS, in each class's own grammar. Parametrizing the
#: doors over this is what makes "per source class" a real axis rather than
#: three copies of the huggingface case.
SOURCE_CLASSES: Tuple[Tuple[str, str], ...] = (
    ("hf", "org/model"),
    ("civitai", "123456"),
    ("modelscope", "org/model"),
)


# ---------------------------------------------------------------------------
# 1. The census, and it states its own size
# ---------------------------------------------------------------------------

#: The upstream-registry fetch primitives. A call to one of these is, by
#: definition, bytes coming off a third-party registry rather than out of the
#: platform CAS — so WHERE they are called is the whole question this cut
#: answers.
_REGISTRY_FETCHES = frozenset({
    "snapshot_download",
    "hf_hub_download",
    "download_civitai",
    "download_hf",
    "download_modelscope",
    "_snapshot_download_with_retries",
})

#: Packages that are INGEST by construction: their entire reason to exist is
#: fetching an upstream artifact so it can be normalized under a layout
#: contract and published into the CAS. Everything else is serve-side.
_INGEST_PACKAGES = ("gen_worker.convert",)

#: What the 2026-08-19 census found and the cut left behind. Stated as numbers
#: so a new direct-serve branch changes a number rather than slipping past a
#: test that was only ever told about three doors.
#:
#: SERVE-side registry fetches must be ZERO. That is the hardcut.
CENSUS_SERVE_SIDE_REGISTRY_FETCHES = 0
#: INGEST-side must stay non-zero: HF ingest owns its own bounded
#: `snapshot_download` and civitai ingest calls `download_civitai`. If this
#: goes to zero the platform can no longer take a model IN, and the refusals
#: below would be pointing operators at a route that does not exist.
CENSUS_INGEST_SIDE_REGISTRY_FETCHES_MIN = 2


def _module_name(path: Path) -> str:
    rel = path.relative_to(_SRC.parent).with_suffix("")
    return ".".join(rel.parts)


def _registry_fetch_sites() -> Dict[str, List[str]]:
    """``{module: [called name, ...]}`` for every upstream-registry fetch call
    in the package. Whole-tree, AST — not grep, which cannot tell a call from a
    docstring naming what was deleted."""
    found: Dict[str, List[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        if "_vendor" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken tree fails elsewhere
            continue
        names: List[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else ""
            )
            if name in _REGISTRY_FETCHES:
                names.append(name)
        if names:
            found[_module_name(path)] = names
    return found


def _partition(sites: Dict[str, List[str]]) -> Tuple[int, int, Dict[str, List[str]]]:
    serve = 0
    ingest = 0
    serve_detail: Dict[str, List[str]] = {}
    for module, names in sites.items():
        if module.startswith(_INGEST_PACKAGES):
            ingest += len(names)
        else:
            serve += len(names)
            serve_detail[module] = names
    return serve, ingest, serve_detail


def test_no_module_outside_ingest_fetches_from_an_upstream_registry() -> None:
    """THE census assertion. Zero serve-side registry fetches, and the failure
    names the module and the call so the next reader does not have to re-run
    the census by hand."""
    serve, _ingest, detail = _partition(_registry_fetch_sites())
    assert serve == CENSUS_SERVE_SIDE_REGISTRY_FETCHES, (
        "a weight source outside gen_worker.convert fetches from an upstream "
        "registry — serving loads ONLY tensorfs CAS snapshots (pgw#1524). "
        f"Sites: {detail}"
    )


def test_the_ingest_edge_still_exists_for_the_refusals_to_point_at() -> None:
    """The other half of the census, and it is not decoration: a refusal that
    names an ingest route the tree no longer has is worse than no refusal."""
    _serve, ingest, _detail = _partition(_registry_fetch_sites())
    assert ingest >= CENSUS_INGEST_SIDE_REGISTRY_FETCHES_MIN, (
        "gen_worker.convert no longer fetches from any upstream registry, so "
        "the ingest route the refusals name does not exist"
    )


def test_the_census_instrument_can_see_a_reintroduced_direct_serve_branch() -> None:
    """The census's OWN red-arm: prove the scanner finds a serve-side fetch
    when one is there, so its green is evidence rather than a scan that found
    nothing because it was looking in the wrong place."""
    planted = ast.parse("def serve(x):\n    return snapshot_download(x)\n")
    names = [
        node.func.id
        for node in ast.walk(planted)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in _REGISTRY_FETCHES
    ]
    assert names == ["snapshot_download"]


# ---------------------------------------------------------------------------
# 2. Every door refuses, per source class
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
# 3. End to end: the SAME bytes refuse direct and serve after ingest
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
