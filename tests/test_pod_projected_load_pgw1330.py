from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

import projection_fixture as fixture  # noqa: E402
from gen_worker._vendor.tensorfs import (  # noqa: E402
    LocalCAS,
    project_snapshot,
    read_entry,
    tree_bytes,
)
from cas_fixture import ingest_repository  # noqa: E402
from gen_worker.models import projection  # noqa: E402
from gen_worker.models import svdq_native as native  # noqa: E402
from gen_worker.models.loading import (  # noqa: E402
    detect_on_disk_dtype,
    snapshot_component_weight_bytes,
)
from gen_worker.models.projection import REF_PREFIX, SNAPSHOTS_DIR  # noqa: E402
from gen_worker.models.store import ModelStore  # noqa: E402
from gen_worker.models.svdq import detect_svdq_artifact  # noqa: E402
from test_svdq_load_device import _Art, _write_multiunit  # noqa: E402

KEY = "d" * 64


def _restamp_nunchaku(checkpoint: Path) -> bytes:

    import struct

    raw = checkpoint.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n])
    meta = dict(header.get("__metadata__") or {})
    meta["model_class"] = "NunchakuQwenImageTransformer2DModel"
    meta["quantization_config"] = json.dumps(
        {"method": "svdquant", "weight": {"dtype": "fp4_e2m1_all"}, "rank": 128}
    )
    header["__metadata__"] = meta
    blob = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(blob)) + blob + raw[8 + n :]


def _same_bytes(a: Any, b: Any) -> bool:
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)))


@pytest.fixture(scope="module")
def pod(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """One multi-component model: the source tree, the CAS, both projections."""

    root = tmp_path_factory.mktemp("pod")
    source = root / "source-model"
    scratch = root / "scratch"
    scratch.mkdir()

    checkpoint, _state, _dim = _write_multiunit(scratch)
    (source / "transformer").mkdir(parents=True)
    (source / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(
        _restamp_nunchaku(checkpoint)
    )
    (source / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "QwenImageTransformer2DModel"}, indent=2)
    )
    (source / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImagePipeline",
                "transformer": ["diffusers", "QwenImageTransformer2DModel"],
                "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
                "vae": ["diffusers", "AutoencoderKLQwenImage"],
            },
            indent=2,
        )
    )
    for component, seed in (("text_encoder", 5), ("vae", 7)):
        directory = source / component
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(
            json.dumps({"_class_name": component, "hidden_size": 8}, indent=2)
        )
        (directory / "model.safetensors").write_bytes(
            fixture.safetensors_bytes(
                {
                    f"{component}.weight": ("BF16", (4, 8), fixture.varied(64, seed)),
                    f"{component}.bias": ("BF16", (8,), fixture.varied(16, seed + 1)),
                }
            )
        )
    (source / "tokenizer").mkdir()
    (source / "tokenizer" / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 1024})
    )

    base = root / "store"
    cas = LocalCAS(base)
    manifest = ingest_repository(cas, source)
    cas.compare_and_swap_ref(
        REF_PREFIX + KEY, cas.store_manifest(manifest), expected=None
    )
    projected = base / SNAPSHOTS_DIR / KEY
    project_snapshot(cas, manifest, projected)

    materialized = base / SNAPSHOTS_DIR / "materialized"
    materialized.mkdir(parents=True)
    for entry in manifest.files:
        target = materialized / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(read_entry(cas, entry))

    return {
        "base": base,
        "source": source,
        "cas": cas,
        "manifest": manifest,
        "projected": projected,
        "materialized": materialized,
        "snapshot": fixture.Fixture(
            base=base, source=source, tree=projected, key=KEY, cas=cas, manifest=manifest
        ).snapshot_message(),
    }


def test_no_tensor_byte_touches_a_filesystem_path(pod: Dict[str, Any]) -> None:

    tree: Path = pod["projected"]
    source: Path = pod["source"]
    weights = sorted(tree.rglob("*.safetensors"))
    assert weights, "the fixture has no weight files at all"
    for path in weights:
        stub = projection.stub_at(path)
        assert stub is not None, f"{path} holds real tensor bytes"
        assert path.lstat().st_size < 512
        rel = path.relative_to(tree).as_posix()
        assert stub.size == (source / rel).stat().st_size, rel

    model_bytes = sum(e.size_bytes for e in pod["manifest"].files)
    assert tree_bytes(tree) * 50 < model_bytes, (
        f"the projected tree costs {tree_bytes(tree)} B against a "
        f"{model_bytes} B model — that is a materialization"
    )


def test_configs_arrive_via_symlinks_as_the_original_bytes(pod: Dict[str, Any]) -> None:
    source: Path = pod["source"]
    tree: Path = pod["projected"]
    configs = ["model_index.json", "transformer/config.json", "vae/config.json",
               "tokenizer/tokenizer_config.json"]
    for rel in configs:
        link = tree / rel
        assert link.is_symlink(), f"{rel} is not a symlink into the CAS"
        assert projection.object_of_symlink(link) is not None, rel
        assert link.read_bytes() == (source / rel).read_bytes(), rel
        assert json.loads(link.read_text())


def test_nothing_mounted_and_nothing_privileged(pod: Dict[str, Any]) -> None:
    """No FUSE, no CAP_SYS_ADMIN: a projection is directories, symlinks and small regular files, and this asserts the tree contains nothing else."""

    for path in sorted(pod["projected"].rglob("*")):
        assert path.is_symlink() or path.is_dir() or path.is_file(), path
        if path.is_file() and not path.is_symlink():
            assert path.lstat().st_size < 4096, f"{path} holds bulk data"


def test_the_store_trusts_the_tree_on_first_use(pod: Dict[str, Any]) -> None:
    """The boot path."""

    store = ModelStore.__new__(ModelStore)
    ok, bad = ModelStore._verify_snapshot_tree(
        store, pod["projected"], pod["snapshot"]
    )
    assert ok, f"the store scored a healthy projected tree corrupt: {bad}"


def test_the_routing_facts_are_identical_projected_and_materialized(
    pod: Dict[str, Any],
) -> None:
    """Everything the worker reads to DECIDE how to load."""

    projected, material = pod["projected"], pod["materialized"]
    assert detect_on_disk_dtype(projected) == detect_on_disk_dtype(material)
    assert snapshot_component_weight_bytes(projected) == snapshot_component_weight_bytes(
        material
    )

    art_projected = detect_svdq_artifact(projected)
    art_material = detect_svdq_artifact(material)
    assert art_material is not None, "the fixture is not an svdq artifact at all"
    assert art_projected is not None, "the svdq artifact vanished under stubs"
    assert art_projected.component == art_material.component == "transformer"
    assert art_projected.model_class == art_material.model_class
    assert art_projected.precision == art_material.precision
    assert art_projected.rank == art_material.rank


def test_the_denoiser_built_from_the_projected_tree_is_bit_identical(
    pod: Dict[str, Any],
) -> None:
    """THE POD-SHAPED CLAUSE."""

    art_projected = detect_svdq_artifact(pod["projected"])
    art_material = detect_svdq_artifact(pod["materialized"])
    assert art_projected is not None and art_material is not None
    assert projection.stub_at(art_projected.file) is not None, (
        "the loader was handed a real file, so this proves nothing"
    )

    from_tree = native.load_svdq_native_denoiser(
        _Art(art_projected.file), mode="dense", device="cpu"
    )
    from_file = native.load_svdq_native_denoiser(
        _Art(art_material.file), mode="dense", device="cpu"
    )

    want: Dict[str, Any] = dict(from_file.named_parameters())
    want.update(dict(from_file.named_buffers()))
    got: Dict[str, Any] = dict(from_tree.named_parameters())
    got.update(dict(from_tree.named_buffers()))
    assert want and set(want) == set(got)
    for name in sorted(want):
        assert got[name].device.type != "meta", f"{name} never got filled"
        assert _same_bytes(want[name], got[name]), name


def test_the_tree_is_still_a_projection_after_the_load(pod: Dict[str, Any]) -> None:
    """The load must not have quietly materialized anything behind our back."""

    for path in sorted(pod["projected"].rglob("*.safetensors")):
        assert projection.stub_at(path) is not None, f"{path} was materialized"
