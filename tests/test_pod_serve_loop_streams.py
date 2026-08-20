"""The test that was always missing: the POD's caller, on a projected tree.

pgw#1551. For ~21 h no endpoint in the fleet completed a serve, and every
local red/green stayed green throughout. The reason was not a missing
assertion. It was that **every test's caller was ``EndpointHost``, and no
pod has ever used ``EndpointHost``.** A pod builds ``ServeLoop``
(``worker.py:582``), ``ServeLoop`` built its load contexts with no engine, and
so nothing on a pod ever asked for the streaming loader. The bug lived
entirely in the gap between the caller under test and the caller in
production — *a unit test is a caller that the production path is not*.

So this suite fixes the CALLER, not the coverage:

* the object under test is ``ServeLoop``, constructed with exactly the keyword
  arguments ``worker.py`` passes on a pod and no others;
* the tree is a REAL projected tensorfs snapshot — a real ``LocalCAS``, a real
  ingested manifest, a real pin, projected by the same ``project_snapshot``
  the chokepoint calls — so its tensor containers really are ~128 B
  ``TFSSTUB1`` pointer stubs whose bytes no path-based read can reach;
* the store is a REAL ``ModelStore``;
* **nothing is mocked on the seam.** A fake engine is what let this defect
  live for weeks: an injected engine makes "was one asked for?" — the only
  question that mattered — unaskable.

The red arm is not hypothetical. ``test_the_pre_1544_state_FAILS_this_test``
reconstructs the exact production state of the outage (``engine_for`` never
reached from ``ctx.load``) and requires this suite to go red for it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from gen_worker._vendor.tensorfs import LocalCAS, project_snapshot  # noqa: E402
from cas_fixture import ingest_repository  # noqa: E402
from gen_worker.models import projection, store as store_mod  # noqa: E402
from gen_worker.models.projection import REF_PREFIX, SNAPSHOTS_DIR  # noqa: E402
from gen_worker.models.refs import WireRef  # noqa: E402
from gen_worker.models.store import ModelStore  # noqa: E402
from gen_worker.pb import worker_scheduler_pb2 as pb  # noqa: E402
from gen_worker.serving import DeployBinding, load_endpoint  # noqa: E402
from gen_worker.serving.residency import ResidencyManager  # noqa: E402
from gen_worker.serving.serve_loop import ServeLoop, manifest_sizer  # noqa: E402

from streaming_fixture import build_source  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "serving_projected_endpoint"
GB = 1024**3
REF = "fixture/streamer@1"
LANE = "fixture.diffusers-bf16@1"
KEY = "e5" * 32


class _PodResolver:
    """The ``BindingResolver`` seam, answering the projected tree.

    Deliberately answers a checkpoint_ref that is NOT the string the store
    banks under: pgw#1543 proved a ref-keyed lookup silently no-ops on a
    spelling mismatch, and the pod holds the resolver's `pick.ref`.
    """

    def __init__(self, tree: Path) -> None:
        self.tree = tree

    def resolve(self, model_cls: type, checkpoint_ref: str) -> DeployBinding:
        return DeployBinding(
            checkpoint_ref=checkpoint_ref, checkpoint_dir=self.tree,
            model="streamer", defaults={},
        )

    def default_pick(self, model_cls: type, slot_name: str) -> str:
        return REF


@pytest.fixture(scope="module")
def projected(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """A real bf16 pipeline, ingested, pinned, and projected. No mocks."""
    base = tmp_path_factory.mktemp("pod-cas")
    source = base / "source-model"
    build_source(source)

    cas = LocalCAS(base)
    manifest = ingest_repository(cas, source)
    # Exactly what `cozy_snapshot._pin_manifest` does, so `resolve_projection`
    # runs against the production pinning and not a test convention.
    cas.compare_and_swap_ref(
        REF_PREFIX + KEY, cas.store_manifest(manifest), expected=None)
    tree = base / SNAPSHOTS_DIR / KEY
    project_snapshot(cas, manifest, tree)

    stubs = [p for p in tree.rglob("*") if p.is_file() and not p.is_symlink()
             and projection.stub_at(p) is not None]
    assert stubs, (
        f"{tree} projected no pointer stubs, so this fixture cannot witness "
        f"the defect it exists for — a tree of real files is exactly the "
        f"shape that stayed green through the outage")
    assert projection.resolve_projection(tree) is not None, "fixture must be PINNED"
    return {"base": base, "tree": tree, "manifest": manifest, "stubs": stubs}


@pytest.fixture()
def bound_store(projected: Dict[str, Any]) -> Iterator[ModelStore]:
    """A REAL ModelStore, bound as the process's active store."""

    async def noop(_message: Any) -> None:
        return None

    store = ModelStore(noop, cache_dir=projected["base"])
    snapshot = pb.Snapshot(digest="sha256:" + KEY)
    for entry in projected["manifest"].files:
        snapshot.files.add(path=entry.path, size_bytes=entry.size_bytes,
                           digest=str(entry.digest))
    # Banked under a DIFFERENT spelling than the one served, on purpose.
    store.bank_snapshot(WireRef("fixture/streamer"), snapshot)
    store_mod.bind_active_store(store)
    try:
        yield store
    finally:
        store_mod.bind_active_store(cast(Any, None))


def pod_serve_loop(projected: Dict[str, Any], tmp_path: Path) -> ServeLoop:
    """``ServeLoop`` as ``worker.py`` builds it on a pod — and ONLY that.

    The keyword set is copied from `worker.py`'s construction. Nothing is
    added here that a pod does not pass, because the whole finding is that a
    test which passes one extra argument is testing a different program: for
    ~21 h the extra argument was `engine=`, supplied by every test's
    `EndpointHost` and by no pod.
    """
    loaded = load_endpoint(FIXTURE)
    return ServeLoop(
        loaded,
        residency=ResidencyManager(
            64 * GB, manifest_sizer({REF: 1 * GB}, headroom_bytes=1 * GB)),
        resolver=_PodResolver(projected["tree"]),
        lane_contract=LANE,
        output_dir=tmp_path / "outputs",
        compile_sink_for=None,
        on_loaded=None,
        hf_token="",
    )


def test_the_POD_loop_serves_a_projected_tree_through_the_streaming_engine(
    projected: Dict[str, Any], bound_store: ModelStore, tmp_path: Path,
) -> None:
    """The whole point, in one request, on the caller a pod actually uses.

    Before pgw#1544 this request raised `ProjectedTreeNotStreamable` — the
    fleet's exact refusal — because `ServeLoop` handed `engine=None` down and
    `ctx.load` never asked for one. The fix moved the ask into `ctx.load`,
    which is the one place that always has the tree; pgw#1549 then deleted the
    second construction site so there is nothing left to disagree with it.
    """
    loop = pod_serve_loop(projected, tmp_path)

    outcome = loop.invoke("probe", {"model": REF, "input": {}},
                          request_id="pod-1")
    evidence = outcome.result

    assert evidence.engine_bound, (
        "NOBODY ASKED FOR THE ENGINE. `ServeLoop` built a LoadContext with no "
        "engine and `ctx.load` did not ask, so this projected tree fell to the "
        "eager `from_pretrained` bridge and met a pointer stub — the ~21 h "
        "fleet outage, reproduced")
    # BOTH stores read the chunked CAS objects; they differ in speed, not in
    # source (`bridge` copies in Python at ~0.59 GiB/s, `native` at ~6.4). The
    # claim under test is that the bytes came out of the CHUNK STORE at all —
    # which of the two answered is a property of the vendored tensorfs build,
    # and pinning it here would make this suite fail for a reason that has
    # nothing to do with the outage.
    assert evidence.stream_source in ("native", "bridge"), (
        f"source={evidence.stream_source!r} is neither chunk-store reader, so "
        f"these weights did not come out of the CAS")
    assert evidence.tensors_streamed > 0, (
        "an engine that bound and streamed nothing is a skeleton with no "
        "weights, which generates noise rather than failing")
    assert evidence.meta_parameters == 0, (
        f"{evidence.meta_parameters} parameter(s) never left the meta device: "
        f"the pipeline was built but its weights did not arrive")
    assert evidence.unet_dtype == "torch.bfloat16", (
        f"bytes land VERBATIM in the container's own dtype and this article "
        f"is bf16 on disk; got {evidence.unet_dtype}")
    assert evidence.unet_checksum > 0.0, (
        "a real parameter's magnitude, so 'the bytes arrived' is a "
        "measurement rather than a tensor of the right shape")


def test_the_POD_loop_reads_no_tensor_file_and_materializes_nothing(
    projected: Dict[str, Any], bound_store: ModelStore, tmp_path: Path,
) -> None:
    """The stubs stay stubs, and no second copy of the tree appears.

    The load succeeding is not by itself proof it streamed: a tier-3
    materialization would also produce a working pipeline, at 2x the disk, and
    is exactly what Paul's 2026-08-19 no-fill ruling removed from the serving
    path. So this measures the FILE SYSTEM afterwards.
    """
    from gen_worker.models import materialized_view

    tree = projected["tree"]
    before = {p: p.stat().st_size for p in projected["stubs"]}

    loop = pod_serve_loop(projected, tmp_path)
    loop.invoke("probe", {"model": REF, "input": {}}, request_id="pod-2")

    for path, size in before.items():
        assert projection.stub_at(path) is not None, (
            f"{path} stopped being a pointer stub — something materialized a "
            f"tensor container onto the serving tree")
        assert path.stat().st_size == size, f"{path} changed size"

    view = materialized_view.view_root_for(tree)
    assert not view.exists(), (
        f"a materialized view appeared at {view}: the pod filled a full second "
        f"copy of the tree to read weights it already had in the chunk store")
    assert materialized_view.serving_streams_weights(), (
        "binding the engine must ARM the no-fill defect signal, so any later "
        "tier-3 call in this process logs as a bug and not a burn-down row")


def test_the_pre_1544_state_FAILS_this_test(
    projected: Dict[str, Any], bound_store: ModelStore, tmp_path: Path,
) -> None:
    """THE RED ARM. Put the outage back and require this suite to catch it.

    A fence that has never been observed red proves nothing when green, and
    this suite's entire claim is that it would have caught the outage. So the
    pre-pgw#1544 production state is reconstructed exactly — `ctx.load`'s
    projected branch cannot reach `engine_for` — and the request must refuse
    with the fleet's own error rather than quietly serving from somewhere else.

    Reconstructed by neutering the ASK (`_bind_streaming_engine` answering
    "no engine bound", which is what the pre-fix code path did when it fell
    through to the refusal), NOT by injecting a fake engine: an injected
    engine is the very thing that hid this.
    """
    from gen_worker.serving import context as context_mod
    from gen_worker.serving.context import ProjectedTreeNotStreamable

    original = context_mod.LoadContext._bind_streaming_engine

    def never_asks(self: Any, *, pinned: bool) -> tuple:
        return False, pinned

    context_mod.LoadContext._bind_streaming_engine = never_asks  # type: ignore[method-assign]
    try:
        loop = pod_serve_loop(projected, tmp_path)
        with pytest.raises(Exception) as caught:
            loop.invoke("probe", {"model": REF, "input": {}},
                        request_id="pod-red")
    finally:
        context_mod.LoadContext._bind_streaming_engine = original  # type: ignore[method-assign]

    chain = []
    error: BaseException | None = caught.value
    while error is not None:
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert any(isinstance(e, ProjectedTreeNotStreamable) for e in chain), (
        f"the pre-pgw#1544 state must FAIL this suite with the fleet's own "
        f"refusal. It raised {type(caught.value).__name__}: {caught.value}. "
        f"If this suite can stay green with the engine ask removed, it does "
        f"not measure the outage and its green means nothing.")


def test_the_pod_loop_hands_down_the_workers_placement_decision(
    projected: Dict[str, Any], bound_store: ModelStore, tmp_path: Path,
) -> None:
    """pgw#1549: `ServeLoop` named no device, so `_placed` placed nothing.

    pgw#1452 fixed exactly this defect — the eager bridge returning a pipeline
    on the CPU because nobody handed it the worker's placement decision — and
    landed the fix on `EndpointHost`, which is not the caller a pod uses. On
    the pod path `LoadContext._device` stayed `""` and `_placed` returned
    early, for the entire life of the v2 worker.

    The device this box probes to is whatever it is; the claim under test is
    that the decision is MADE and handed down, never left empty.
    """
    from gen_worker.serving.placement import serving_device

    loop = pod_serve_loop(projected, tmp_path)
    loop.invoke("probe", {"model": REF, "input": {}}, request_id="pod-3")

    contexts = [backend.load_context
                for backend in loop._backends.values()]
    assert contexts, "the invoke built no backend, so there is nothing to read"
    for ctx in contexts:
        assert ctx._device == serving_device(), (
            f"the pod's LoadContext carries device={ctx._device!r}; a pod that "
            f"names no device runs the eager bridge on the CPU and nothing "
            f"fails — it is simply the wrong processor (pgw#1452)")
