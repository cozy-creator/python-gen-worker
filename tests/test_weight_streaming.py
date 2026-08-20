"""pgw#1380: ``ctx.load`` streams a checkpoint store->memory, writing nothing.

No mocks anywhere. A REAL diffusers pipeline is built and ``save_pretrained``
to real safetensors files; those files are ingested into a REAL chunked
``LocalCAS``; the tree is projected exactly as the chokepoint projects it (so
every tensor file in it is a POINTER STUB, not bytes); and the engine loads
from the chunk store. What is asserted is what the ruling asked for:

* every parameter is byte-equal to the source file's bytes;
* the store is read in ASCENDING FILE OFFSET order, per container, over MANY
  windows — proven by a read trace, because a scrambled walk is still correct
  and merely slow, which is the regression that would otherwise hide forever;
* the load writes ~zero bytes to disk (``/proc/self/io``);
* nothing survives on ``meta``.

The CUDA half (pinned staging, ``cudaMemcpyAsync`` on a copy stream, per-buffer
events) rides the SAME driver loop this exercises — only the copy primitive
differs — and its measurement is e2e#1906's, on a pod.

The same engine over the REAL tensorfs stream surface (the tensorfs#115
extension) is ``test_ctx_load_native_tensorfs_pgw1380.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from gen_worker._vendor.tensorfs import LocalCAS, project_snapshot  # noqa: E402
from cas_fixture import ingest_repository  # noqa: E402
from gen_worker.models.projection import REF_PREFIX, SNAPSHOTS_DIR  # noqa: E402
from gen_worker.serving.streaming import (  # noqa: E402
    BridgeWeightStore,
    NameMismatch,
    StreamingLoader,
    engine_for,
)
from gen_worker.serving.streaming.skeleton import (  # noqa: E402
    SkeletonError,
    _resolve,
    meta_survivors,
)
from streaming_fixture import (  # noqa: E402
    Lane,
    TracedStore,
    assert_byte_equal,
    build_source,
    header_order_differs_from_offset_order,
    write_bytes_now,
)

#: Small on purpose: the walk must span MANY windows, because a load that
#: fits in one window is trivially in file order and proves nothing.
WINDOW = 4096


def _project(base: Path, source: Path, key: str) -> Path:
    """Ingest into a real CAS and project the tree, the chokepoint's way."""
    cas = LocalCAS(base)
    manifest = ingest_repository(cas, source)
    cas.compare_and_swap_ref(
        REF_PREFIX + key, cas.store_manifest(manifest), expected=None
    )
    tree = base / SNAPSHOTS_DIR / key
    project_snapshot(cas, manifest, tree)
    return tree


@pytest.fixture(scope="module")
def article(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("pgw1380")
    source = base / "source-model"
    pipeline_cls = build_source(source)
    tree = _project(base, source, key="b" * 64)
    return {"base": base, "source": source, "tree": tree,
            "pipeline_cls": pipeline_cls}


def _cas_manifest(tree: Path) -> Tuple[Any, Any]:
    from gen_worker.models import projection

    projected = projection.resolve_projection(tree)
    assert projected is not None
    return projected.cas, projected.manifest


# -- the acceptance ---------------------------------------------------------


def test_the_fixture_can_actually_witness_a_scrambled_walk(
    article: dict[str, Any]
) -> None:
    """The guard on the guard.

    ``safetensors`` writes tensors in header-key order, so a library-saved
    checkpoint has name order == offset order — and against one of those, a
    loader walking in NAME order looks perfectly sequential and every
    file-order assertion below passes on a lie. This is the assertion that
    keeps the rest honest, and it is separate so a fixture that silently
    stopped scrambling fails HERE, by name, rather than nowhere.
    """
    containers = sorted(article["source"].rglob("*.safetensors"))
    assert containers
    for container in containers:
        assert header_order_differs_from_offset_order(container), (
            f"{container.name}: header order IS offset order, so this "
            f"fixture cannot tell a file-order walk from a name-order one"
        )


def test_ctx_load_streams_store_to_memory_writing_nothing(
    article: dict[str, Any]
) -> None:
    tree: Path = article["tree"]
    source: Path = article["source"]
    pipeline_cls: type = article["pipeline_cls"]

    # The tree really is stub-only: reading a weight file at its path yields
    # a pointer, not weights. If this stops holding the rest proves nothing.
    from gen_worker.models.projection import stub_at

    stubs = [p for p in sorted(tree.rglob("*.safetensors")) if stub_at(p) is not None]
    assert stubs, f"{tree} projected no pointer stubs — nothing to stream"

    store = TracedStore(BridgeWeightStore(*_cas_manifest(tree)))
    loader = StreamingLoader(store, device="cpu", buffer_bytes=WINDOW, buffers=3)

    before = write_bytes_now()
    pipeline = loader.build(pipeline_cls, checkpoint_dir=tree, lane=Lane())
    written = write_bytes_now() - before

    assert isinstance(pipeline, pipeline_cls)
    report = loader.last_report
    assert report is not None
    assert report.weights_streamed_bytes > 0
    assert report.staging == "pageable"  # no CUDA here; the fact is reported
    assert report.io == "buffered"
    assert report.containers == 4

    # 1. BYTE EQUALITY, per parameter, against the source files.
    assert assert_byte_equal(pipeline, source) > 100

    # 2. NOTHING ON META.
    for component in ("unet", "vae", "text_encoder", "text_encoder_2"):
        assert meta_survivors(getattr(pipeline, component)) == ()

    # 3. FILE ORDER, over many windows.
    windows = store.assert_file_order()
    assert windows > 20, (
        f"only {windows} window(s) were read; a walk that fits in one window "
        f"is ordered by accident and cannot witness a scrambled one"
    )
    assert report.windows == windows

    # 4. ZERO DISK WRITES. Not "few files created": no bytes. A little slack
    #    for the process's own logging/journal noise, orders of magnitude
    #    below the weights a fill would have written.
    assert written < 1 << 20, (
        f"the streamed load wrote {written} bytes; the whole point of the "
        f"2026-08-19 ruling is that it writes none"
    )


def test_every_container_is_read_end_to_end_exactly_once(
    article: dict[str, Any]
) -> None:
    """The windows tile each container's data range: no gap (a tensor read
    from nowhere) and no overlap (a byte paid for twice)."""
    store = TracedStore(BridgeWeightStore(*_cas_manifest(article["tree"])))
    loader = StreamingLoader(store, device="cpu", buffer_bytes=WINDOW, buffers=3)
    loader.build(article["pipeline_cls"], checkpoint_dir=article["tree"], lane=Lane())

    per_container: dict[str, list[tuple[int, int]]] = {}
    for container, offset, length in store.reads:
        per_container.setdefault(container, []).append((offset, length))
    for container, reads in per_container.items():
        cursor = reads[0][0]
        for offset, length in reads:
            assert offset == cursor, (
                f"{container}: window starts at {offset}, previous ended at "
                f"{cursor} — the walk is not a contiguous forward pass"
            )
            cursor = offset + length


def test_the_engine_binds_off_the_projected_tree_alone(article: dict[str, Any]) -> None:
    """A worker holding only the checkpoint DIRECTORY can bind the engine —
    the projected tree carries its own chunk store."""
    engine = engine_for(article["tree"], device="cpu")
    assert engine is not None
    pipeline = engine.build(
        article["pipeline_cls"], checkpoint_dir=article["tree"], lane=Lane()
    )
    assert meta_survivors(pipeline.unet) == ()

    from gen_worker.models import materialized_view

    assert materialized_view.serving_streams_weights(), (
        "binding the streaming engine must arm the no-fill defect signal"
    )


def test_a_tree_with_no_store_behind_it_binds_no_engine(tmp_path: Path) -> None:
    bare = tmp_path / "bare-download"
    bare.mkdir()
    assert engine_for(bare, device="cpu") is None


def test_an_unplaceable_name_refuses_instead_of_guessing(
    article: dict[str, Any], tmp_path: Path
) -> None:
    """A container carrying a name the skeleton has no slot for is a
    checkpoint the code does not match — a typed refusal, never a shrug."""
    source = tmp_path / "source-model"
    _copy_tree(article["source"], source)
    _append_tensor(source / "vae" / "diffusion_pytorch_model.safetensors",
                   "not_a_real_parameter")
    tree = _project(tmp_path, source, key="c" * 64)

    engine = engine_for(tree, device="cpu")
    assert engine is not None
    with pytest.raises(NameMismatch) as caught:
        engine.build(article["pipeline_cls"], checkpoint_dir=tree, lane=Lane())
    assert "not_a_real_parameter" in str(caught.value)


def test_a_missing_name_refuses_rather_than_serving_meta(
    article: dict[str, Any], tmp_path: Path
) -> None:
    """A tensor the checkpoint does not carry would serve uninitialized
    memory on the first request. It refuses, naming the names."""
    source = tmp_path / "source-model"
    _copy_tree(article["source"], source)
    victim = _drop_tensor(source / "vae" / "diffusion_pytorch_model.safetensors")
    tree = _project(tmp_path, source, key="d" * 64)

    engine = engine_for(tree, device="cpu")
    assert engine is not None
    with pytest.raises(NameMismatch) as caught:
        engine.build(article["pipeline_cls"], checkpoint_dir=tree, lane=Lane())
    assert "still on meta" in str(caught.value)
    assert victim in str(caught.value)


def test_a_load_reads_no_tensor_bytes_to_build_the_skeleton(
    article: dict[str, Any]
) -> None:
    """Step 1 is configs only: the skeleton exists before a byte is read."""
    from gen_worker.serving.streaming import skeleton

    built = skeleton.build(article["pipeline_cls"], article["tree"])
    assert set(built.modules) == {"unet", "vae", "text_encoder", "text_encoder_2"}
    assert built.passthrough == ("scheduler",)
    for module in built.modules.values():
        assert meta_survivors(module), "a config-only build must hold NO weights"
        assert all(
            parameter.device.type == "meta" for parameter in module.parameters()
        )


def test_two_components_sharing_every_name_do_not_collide(
    article: dict[str, Any]
) -> None:
    """``text_encoder`` and ``text_encoder_2`` are the same architecture and
    share every parameter name, with DIFFERENT weights. Names are unique only
    within a container, which is the only scope this engine resolves them in."""
    engine = engine_for(article["tree"], device="cpu")
    assert engine is not None
    pipeline = engine.build(
        article["pipeline_cls"], checkpoint_dir=article["tree"], lane=Lane()
    )
    first = dict(pipeline.text_encoder.named_parameters())
    second = dict(pipeline.text_encoder_2.named_parameters())
    assert set(first) == set(second)
    differing = [
        name for name in first
        if not torch.equal(first[name], second[name])
    ]
    assert differing, (
        "the two text encoders came back identical — one container's bytes "
        "were served for both"
    )


# -- fixture surgery --------------------------------------------------------


def _copy_tree(source: Path, target: Path) -> None:
    import shutil

    shutil.copytree(source, target)


def _read_header(path: Path) -> Tuple[dict[str, Any], bytes]:
    import struct

    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    return json.loads(raw[8 : 8 + size]), raw[8 + size :]


def _write_container(path: Path, header: dict[str, Any], body: bytes) -> None:
    import struct

    blob = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + body)


def _append_tensor(path: Path, name: str) -> None:
    header, body = _read_header(path)
    start = len(body)
    body = body + bytes(range(16))
    header[name] = {"dtype": "U8", "shape": [16], "data_offsets": [start, start + 16]}
    _write_container(path, header, body)


def _drop_tensor(path: Path) -> str:
    """Rewrite the container WITHOUT one tensor — a real checkpoint that is
    missing a name, not a malformed one.

    The body is rebuilt and every surviving span reindexed, because popping a
    header key alone leaves the victim's bytes in the body addressed by
    nothing. The vendored reader shrugs at that gap; the native reader refuses
    the file outright ("these records are not a tensor container"), and it is
    right to — safetensors spans tile the body. Leaving the gap in would test
    the store's tolerance for a corrupt file rather than the engine's refusal
    to serve meta.
    """
    header, body = _read_header(path)
    victim = sorted(key for key in header if key != "__metadata__")[0]
    header.pop(victim)
    rebuilt = bytearray()
    for name in sorted(
        (key for key in header if key != "__metadata__"),
        key=lambda key: header[key]["data_offsets"][0],
    ):
        start, end = header[name]["data_offsets"]
        header[name]["data_offsets"] = [len(rebuilt), len(rebuilt) + (end - start)]
        rebuilt += body[start:end]
    _write_container(path, header, bytes(rebuilt))
    return victim


# -- pgw#1410: the skeleton must carry what the loader is about to fill ------


def test_a_pipeline_that_drops_its_components_is_refused_pgw1410(
    article: dict[str, Any],
) -> None:
    """A `ModularPipeline` fails OPEN without this, and expensively.

    `ModularPipeline.__init__` routes `**kwargs` to `load_config` and drops
    every component, then registers each one as None. The skeleton returned a
    pipeline whose components were all None while `modules` held the real
    objects; `StreamingLoader` then streamed the whole checkpoint into those
    ORPHANS. `meta_survivors` passed — it is per-module and the modules were
    fine; it was the PIPELINE that was empty — so every layer reported success
    and the defect surfaced as `None` where a component belongs, on a rented
    pod, after a full weight load had been paid for.

    The stand-in reproduces the SHAPE (accept the components, keep none) rather
    than importing `ModularPipeline`, so the guard is proven against the
    contract violation itself and not against one vendor class.
    """
    from gen_worker.serving.streaming import skeleton as skeleton_mod

    base_cls = article["pipeline_cls"]

    class _DropsComponents(base_cls):  # type: ignore[valid-type, misc]
        def __init__(self, **kwargs: Any) -> None:  # noqa: D107
            super().__init__(**kwargs)
            for name in list(kwargs):
                object.__setattr__(self, name, None)

    with pytest.raises(skeleton_mod.SkeletonError) as caught:
        skeleton_mod.build(_DropsComponents, checkpoint_dir=article["tree"])

    message = str(caught.value)
    assert "did not keep the component" in message, message
    # It must NAME them — a refusal that says only "something is wrong" leaves
    # the author exactly where the silent None did.
    assert "unet" in message, message
    assert "update_components" in message, message


def test_a_well_behaved_pipeline_still_builds_pgw1410(
    article: dict[str, Any],
) -> None:
    """The red control's twin: the guard must not refuse the normal shape.

    Without this, deleting the check entirely would leave the suite green.
    """
    from gen_worker.serving.streaming import skeleton as skeleton_mod

    built = skeleton_mod.build(
        article["pipeline_cls"], checkpoint_dir=article["tree"]
    )
    for name, module in built.modules.items():
        if name:
            assert getattr(built.pipeline, name) is module


# ---------------------------------------------------------------------------
# The skeleton's LIBRARY RESOLUTION — same subsystem, same file
# (`serving/streaming/skeleton.py`), so it lives with the streaming tests
# rather than in a module of its own.
# ---------------------------------------------------------------------------

# pgw#1518: a `model_index.json` library entry may be a diffusers PIPELINE
# SUBMODULE, not an importable module. Every sd15 checkpoint on the hub names
# one, and the streaming skeleton refused all of them — so the first boot ever
# to drive a CAS-backed sd15 tree through this loader died here.


def test_pipeline_submodule_library_resolves() -> None:
    """`stable_diffusion` is not importable as a top-level module; it is a
    diffusers pipeline submodule. This is the exact entry every sd15
    model_index.json carries, and the boot that found it died here."""
    cls = _resolve("stable_diffusion", "StableDiffusionSafetyChecker")
    assert cls.__name__ == "StableDiffusionSafetyChecker"
    with pytest.raises(ImportError):
        __import__("stable_diffusion")


def test_ordinary_module_libraries_still_resolve() -> None:
    assert _resolve("transformers", "CLIPTextModel").__name__ == "CLIPTextModel"
    assert _resolve("diffusers", "UNet2DConditionModel").__name__ == "UNet2DConditionModel"


def test_unknown_library_still_refuses_by_name() -> None:
    """A genuinely absent library must still be a typed refusal — the fallback
    widens what resolves, it must not turn a miss into a silent None."""
    with pytest.raises(SkeletonError) as caught:
        _resolve("no_such_library_anywhere", "Thing")
    assert "no_such_library_anywhere" in str(caught.value)


def test_known_library_unknown_class_refuses_by_name() -> None:
    with pytest.raises(SkeletonError) as caught:
        _resolve("diffusers", "NoSuchClassInThisVersion")
    assert "NoSuchClassInThisVersion" in str(caught.value)
