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

WINDOW = 4096


def _project(base: Path, source: Path, key: str) -> Path:
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


def test_the_fixture_can_actually_witness_a_scrambled_walk(
    article: dict[str, Any]
) -> None:
    """The guard on the guard."""
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
    assert report.staging == "pageable"
    assert report.io == "buffered"
    assert report.containers == 4

    assert assert_byte_equal(pipeline, source) > 100

    for component in ("unet", "vae", "text_encoder", "text_encoder_2"):
        assert meta_survivors(getattr(pipeline, component)) == ()

    windows = store.assert_file_order()
    assert windows > 20, (
        f"only {windows} window(s) were read; a walk that fits in one window "
        f"is ordered by accident and cannot witness a scrambled one"
    )
    assert report.windows == windows

    assert written < 1 << 20, (
        f"the streamed load wrote {written} bytes; the whole point of the "
        f"2026-08-19 ruling is that it writes none"
    )


def test_every_container_is_read_end_to_end_exactly_once(
    article: dict[str, Any]
) -> None:
    """The windows tile each container's data range: no gap (a tensor read from nowhere) and no overlap (a byte paid for twice)."""
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
    """A worker holding only the checkpoint DIRECTORY can bind the engine — the projected tree carries its own chunk store."""
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
    """A container carrying a name the skeleton has no slot for is a checkpoint the code does not match — a typed refusal, never a shrug."""
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
    """A tensor the checkpoint does not carry would serve uninitialized memory on the first request."""
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
    """``text_encoder`` and ``text_encoder_2`` are the same architecture and share every parameter name, with DIFFERENT weights."""
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


def test_a_pipeline_that_drops_its_components_is_refused_pgw1410(
    article: dict[str, Any],
) -> None:
    """A `ModularPipeline` fails OPEN without this, and expensively."""
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
    assert "unet" in message, message
    assert "update_components" in message, message


def test_a_well_behaved_pipeline_still_builds_pgw1410(
    article: dict[str, Any],
) -> None:
    """The red control's twin: the guard must not refuse the normal shape."""
    from gen_worker.serving.streaming import skeleton as skeleton_mod

    built = skeleton_mod.build(
        article["pipeline_cls"], checkpoint_dir=article["tree"]
    )
    for name, module in built.modules.items():
        if name:
            assert getattr(built.pipeline, name) is module


def test_pipeline_submodule_library_resolves() -> None:
    """`stable_diffusion` is not importable as a top-level module; it is a diffusers pipeline submodule."""
    cls = _resolve("stable_diffusion", "StableDiffusionSafetyChecker")
    assert cls.__name__ == "StableDiffusionSafetyChecker"
    with pytest.raises(ImportError):
        __import__("stable_diffusion")


def test_ordinary_module_libraries_still_resolve() -> None:
    assert _resolve("transformers", "CLIPTextModel").__name__ == "CLIPTextModel"
    assert _resolve("diffusers", "UNet2DConditionModel").__name__ == "UNet2DConditionModel"


def test_unknown_library_still_refuses_by_name() -> None:
    """A genuinely absent library must still be a typed refusal — the fallback widens what resolves, it must not turn a miss into a silent None."""
    with pytest.raises(SkeletonError) as caught:
        _resolve("no_such_library_anywhere", "Thing")
    assert "no_such_library_anywhere" in str(caught.value)


def test_known_library_unknown_class_refuses_by_name() -> None:
    with pytest.raises(SkeletonError) as caught:
        _resolve("diffusers", "NoSuchClassInThisVersion")
    assert "NoSuchClassInThisVersion" in str(caught.value)
