"""The test that was always missing: the POD's caller, on a projected tree."""

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
LANE = "sd15.diffusers@1+plain.bf16@1"
KEY = "e5" * 32


class _PodResolver:

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
    """A real bf16 pipeline, ingested, pinned, and projected."""
    base = tmp_path_factory.mktemp("pod-cas")
    source = base / "source-model"
    build_source(source)

    cas = LocalCAS(base)
    manifest = ingest_repository(cas, source)
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
    store.bank_snapshot(WireRef("fixture/streamer"), snapshot)
    store_mod.bind_active_store(store)
    try:
        yield store
    finally:
        store_mod.bind_active_store(cast(Any, None))


def pod_serve_loop(projected: Dict[str, Any], tmp_path: Path) -> ServeLoop:
    """``ServeLoop`` as ``worker.py`` builds it on a pod — and ONLY that."""
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
    """The whole point, in one request, on the caller a pod actually uses."""
    loop = pod_serve_loop(projected, tmp_path)

    outcome = loop.invoke("probe", {"model": REF, "input": {}},
                          request_id="pod-1")
    evidence = outcome.result

    assert evidence.engine_bound, (
        "NOBODY ASKED FOR THE ENGINE. `ServeLoop` built a LoadContext with no "
        "engine and `ctx.load` did not ask, so this projected tree fell to the "
        "eager `from_pretrained` bridge and met a pointer stub — the ~21 h "
        "fleet outage, reproduced")
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
    """The stubs stay stubs, and no second copy of the tree appears."""
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
    """THE RED ARM."""
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
