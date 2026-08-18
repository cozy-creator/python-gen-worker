"""pgw#1380 x tensorfs#115: the engine over the REAL native stream surface.

The other suite proves the engine against the storage plane pgw carries today.
This one proves it against the one it is DESIGNED for: the maturin extension's
``ObjectStore`` + ``TensorStreamReader`` — ordered iteration as an API
contract, ``readinto`` copying in Rust with the GIL released, and ``O_DIRECT``
plumbed. No adapter code stands between them: ``TensorStreamReader`` satisfies
the engine's ``TensorStream`` protocol structurally.

The proof that the load really comes from the chunk store and not from a file
is blunt: **every safetensors file is deleted after ingest**, and the pipeline
loads anyway. Only ``model_index.json`` and the component configs remain, which
is exactly the projected tree's shape (non-tensor files stay symlinks).

Skipped until the tensorfs#57 wheel is a declared dependency of this package.
It is not a hypothetical: run it with the wheel on the path and it is green.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")
native = pytest.importorskip(
    "tensorfs.native",
    reason="the tensorfs#57 wheel is not installed; tensorfs#115's native "
           "stream surface is what this suite exercises",
)
if not hasattr(native, "TensorStreamReader"):
    # Presence of the PACKAGE is not the question, and asking only that is why
    # this suite was RED rather than skipped: pgw resolves a tensorfs
    # transitively through torchcg, pinned BELOW #115. Ask for the surface.
    pytest.skip(
        "the installed tensorfs predates tensorfs#115 — it carries no "
        "TensorStreamReader, which is the whole subject of this suite",
        allow_module_level=True,
    )

from gen_worker.serving.streaming import NativeWeightStore, StreamingLoader  # noqa: E402
from gen_worker.serving.streaming.skeleton import meta_survivors  # noqa: E402
from streaming_fixture import (  # noqa: E402
    Lane,
    TracedStore,
    build_source,
    source_tensors,
    write_bytes_now,
)

WINDOW = 4096


def _admit(store: Any, path: Path) -> List[Any]:
    """One file into the CAS, as its ordered record run — the TFM1 shape."""
    _plan, admitted = store.admit_file(path)
    return [native.FileRecord.data(item.digest, item.length) for item in admitted]


@pytest.fixture(scope="module")
def article(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    base = tmp_path_factory.mktemp("pgw1380-native")
    source = base / "source-model"
    pipeline_cls = build_source(source)
    expected = source_tensors(source)

    store = native.ObjectStore(base / "objects")
    records: Dict[str, List[Any]] = {}
    for container in sorted(source.rglob("*.safetensors")):
        relative = container.relative_to(source).as_posix()
        records[relative] = _admit(store, container)
    assert records, "the fixture produced no tensor containers"

    # THE PROOF: the bytes are only in the chunk store now.
    for container in sorted(source.rglob("*.safetensors")):
        container.unlink()

    return {
        "source": source,
        "store": store,
        "records": records,
        "expected": expected,
        "pipeline_cls": pipeline_cls,
    }


def _weight_store(article: Dict[str, Any]) -> NativeWeightStore:
    return NativeWeightStore(article["store"], article["records"])


def test_the_native_reader_loads_a_pipeline_with_no_tensor_file_on_disk(
    article: Dict[str, Any]
) -> None:
    source: Path = article["source"]
    assert not list(source.rglob("*.safetensors")), (
        "a tensor file survived the fixture; the load could be reading it"
    )

    store = TracedStore(_weight_store(article))
    loader = StreamingLoader(store, device="cpu", buffer_bytes=WINDOW, buffers=3)

    before = write_bytes_now()
    pipeline = loader.build(
        article["pipeline_cls"], checkpoint_dir=source, lane=Lane()
    )
    written = write_bytes_now() - before

    report = loader.last_report
    assert report is not None
    assert report.containers == 4
    assert report.weights_streamed_bytes > 0

    checked = 0
    for component, tensors in article["expected"].items():
        module = getattr(pipeline, component)
        live = dict(module.named_parameters(remove_duplicate=False))
        live.update(dict(module.named_buffers(remove_duplicate=False)))
        for name, want in tensors.items():
            got = live[name]
            assert got.dtype == want.dtype
            assert torch.equal(
                got.reshape(-1).view(torch.uint8),
                want.reshape(-1).view(torch.uint8),
            ), f"{component}/{name} is not byte-equal to the ingested source"
            checked += 1
    assert checked > 100

    for component in article["expected"]:
        assert meta_survivors(getattr(pipeline, component)) == ()

    windows = store.assert_file_order()
    assert windows > 20
    assert written < 1 << 20


def test_the_native_readers_order_is_the_order_the_engine_walks(
    article: Dict[str, Any]
) -> None:
    """``TensorStreamReader.tensors`` is ascending file offset BY CONTRACT
    (tensorfs#115), which is what makes the engine's forward walk sequential
    rather than merely monotonic-by-our-own-sort."""
    store = _weight_store(article)
    for container in store.containers():
        offsets = [tensor.offset for tensor in store.open(container).tensors]
        assert offsets == sorted(offsets), (
            f"{container}: the native reader handed back header order, not "
            f"file-offset order"
        )


def test_o_direct_is_plumbed_and_byte_identical_to_buffered(
    article: Dict[str, Any]
) -> None:
    """``direct=True`` is the flagged variant the benchmark decides on
    (e2e#1906): the default stays buffered, but the arm must exist and must
    not change a single byte."""
    loader = StreamingLoader(
        _weight_store(article), device="cpu", io="direct",
        buffer_bytes=WINDOW, buffers=3,
    )
    try:
        pipeline = loader.build(
            article["pipeline_cls"], checkpoint_dir=article["source"], lane=Lane()
        )
    except OSError as exc:  # tmpfs and friends refuse O_DIRECT outright
        pytest.skip(f"this filesystem refuses O_DIRECT: {exc}")

    report = loader.last_report
    assert report is not None
    assert report.io == "direct"
    for component, tensors in article["expected"].items():
        module = getattr(pipeline, component)
        live = dict(module.named_parameters(remove_duplicate=False))
        live.update(dict(module.named_buffers(remove_duplicate=False)))
        for name, want in tensors.items():
            assert torch.equal(
                live[name].reshape(-1).view(torch.uint8),
                want.reshape(-1).view(torch.uint8),
            ), f"{component}/{name} differs between the direct and buffered arms"


def test_the_native_reader_satisfies_the_engines_protocol(
    article: Dict[str, Any]
) -> None:
    """Structural, not nominal: nothing adapts the extension to the engine.
    If tensorfs renames a member, this is where it is caught rather than in a
    load that half-works."""
    from gen_worker.serving.streaming import TensorStream

    store = _weight_store(article)
    stream = store.open(next(iter(store.containers())))
    assert isinstance(stream, TensorStream)
    first = stream.tensors[0]
    for member in ("name", "dtype", "shape", "offset", "nbytes"):
        assert hasattr(first, member), f"StreamTensor lost {member}"


def test_store_for_selects_the_native_reader_over_a_projected_tree(
    tmp_path: Path,
) -> None:
    """The wiring, not the reader: what a WORKER resolving a real projected
    snapshot tree actually gets back.

    ``NativeWeightStore`` was constructible long before it was constructed —
    ``store_for`` returned the bridge unconditionally, so an image carrying a
    real tensorfs (se#756) still served every request through the GIL-bound
    copy. This asserts the selection itself, over a store written by the
    vendored plane the worker really uses: the two planes share
    ``objects/sha256/aa/bb/<hex>`` exactly, so the native reader maps the very
    objects the projection wrote.
    """
    from gen_worker._vendor.tensorfs.local import LocalCAS
    from gen_worker._vendor.tensorfs.project import project_snapshot
    from gen_worker.serving.streaming import BridgeWeightStore, store_for
    from gen_worker.serving.streaming.source import native_available

    assert native_available()

    source = tmp_path / "source-model"
    pipeline_cls = build_source(source)
    expected = source_tensors(source)

    cas = LocalCAS(tmp_path / "store")
    manifest = cas.ingest_repository(source)
    cas.compare_and_swap_ref(
        "snapshot:demo", cas.store_manifest(manifest), expected=None
    )
    tree = project_snapshot(cas, manifest, tmp_path / "store/snapshots/demo")

    selected = store_for(tree)
    assert isinstance(selected, NativeWeightStore), (
        f"a projected tree with a real tensorfs installed resolved "
        f"{type(selected).__name__} — the ~10x is being left on the floor"
    )
    assert sorted(selected.containers()) == sorted(
        BridgeWeightStore(cas, manifest).containers()
    ), "the two planes disagree about which files are tensor containers"

    loader = StreamingLoader(selected, device="cpu", buffer_bytes=WINDOW, buffers=3)
    pipeline = loader.build(pipeline_cls, checkpoint_dir=tree, lane=Lane())
    report = loader.last_report
    assert report is not None and report.source == "native"
    for component, tensors in expected.items():
        module = getattr(pipeline, component)
        live = dict(module.named_parameters(remove_duplicate=False))
        live.update(dict(module.named_buffers(remove_duplicate=False)))
        for name, want in tensors.items():
            assert torch.equal(
                live[name].reshape(-1).view(torch.uint8),
                want.reshape(-1).view(torch.uint8),
            ), f"{component}/{name} is not byte-equal through the native store"

    # The bridge stays reachable, because it is the fallback.
    assert isinstance(store_for(tree, native=False), BridgeWeightStore)
