from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import projection_fixture as fixture
from gen_worker._vendor.tensorfs import parse_stub, stub_bytes
from gen_worker.models import projection
from gen_worker.models.loading import detect_on_disk_dtype
from gen_worker.models.store import ModelStore


def _store(base: Path) -> ModelStore:
    return ModelStore.__new__(ModelStore)


def _verify(base: Path, tree: Path, snapshot: Any) -> tuple[bool, list[str]]:
    return ModelStore._verify_snapshot_tree(_store(base), tree, snapshot)


def test_boot_verification_of_a_projected_tree_is_clean(tmp_path: Path) -> None:
    """THE RED PROOF for the infinite re-download."""

    built = fixture.build(tmp_path)
    ok, bad = _verify(tmp_path, built.tree, built.snapshot_message())
    assert ok, f"a projected tree was scored corrupt: {bad}"
    assert bad == []


def test_boot_verification_reads_no_tensor_bytes(tmp_path: Path) -> None:
    """It must be clean because it checked the right thing, not because it skipped: hashing a stub at its path is a check of the wrong bytes."""

    from gen_worker.models.volume_verify import (
        snapshot_verify_targets,
        split_projection_targets,
        verify_projection,
    )

    built = fixture.build(tmp_path)
    targets, skipped = snapshot_verify_targets(built.snapshot_message().files, built.tree)
    assert skipped == []
    projected, material = split_projection_targets(targets)
    assert material == [], "a projected tree has no file holding real bytes"
    assert len(projected) == len(built.manifest.files)
    report = verify_projection(projected)
    assert report.bad == [] and report.findings == []
    assert report.projected == len(targets) and report.hashed == 0
    assert report.bytes_hashed == 0


def test_a_stub_naming_the_wrong_body_is_still_corrupt(tmp_path: Path) -> None:
    """The fix must still DISCRIMINATE."""

    built = fixture.build(tmp_path)
    victim = next(iter(fixture.iter_stubs(built.tree)))
    original = parse_stub(victim.read_bytes())
    assert original is not None
    victim.chmod(0o644)
    victim.write_bytes(stub_bytes("f" * 64, original.size))
    ok, bad = _verify(tmp_path, built.tree, built.snapshot_message())
    assert not ok and bad, "a stub naming the wrong body passed verification"


def test_a_stub_declaring_the_wrong_size_is_still_corrupt(tmp_path: Path) -> None:
    built = fixture.build(tmp_path)
    victim = next(iter(fixture.iter_stubs(built.tree)))
    original = parse_stub(victim.read_bytes())
    assert original is not None
    victim.chmod(0o644)
    victim.write_bytes(stub_bytes(original.body_sha256, original.size + 1))
    ok, bad = _verify(tmp_path, built.tree, built.snapshot_message())
    assert not ok and bad


def test_a_symlink_pointing_out_of_the_store_is_still_corrupt(tmp_path: Path) -> None:
    built = fixture.build(tmp_path)
    victim = built.tree / "model_index.json"
    assert victim.is_symlink()
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps({"_class_name": "Impostor"}))
    victim.unlink()
    victim.symlink_to(elsewhere)
    ok, bad = _verify(tmp_path, built.tree, built.snapshot_message())
    assert not ok and bad


def test_a_truncated_materialized_shard_is_still_corrupt(tmp_path: Path) -> None:
    """THE CONTROL ARM."""

    built = fixture.build(tmp_path)
    tree = fixture.read_entry_tree(tmp_path, built)
    ok, bad = _verify(tmp_path, tree, built.snapshot_message())
    assert ok, f"an intact materialized tree was scored corrupt: {bad}"

    victim = sorted(tree.rglob("*.safetensors"))[0]
    victim.chmod(0o644)
    data = victim.read_bytes()
    victim.write_bytes(data[: len(data) - 4])
    ok, bad = _verify(tmp_path, tree, built.snapshot_message())
    assert not ok and bad, "a truncated real shard passed verification"


def test_a_manifestless_projected_tree_is_not_scored_corrupt(tmp_path: Path) -> None:
    """The structural sweep runs on trees the manifest cannot cover (hf, civitai, single-file)."""

    built = fixture.build(tmp_path)
    ok, bad = _verify(tmp_path, built.tree, None)
    assert ok, f"the structural sweep scored a projected tree corrupt: {bad}"


@pytest.mark.parametrize(
    "safetensors_dtype,expected", [("BF16", "bf16"), ("F16", "fp16"), ("F8_E4M3", "fp8")]
)
def test_dtype_over_a_projected_tree_comes_from_the_manifest(
    tmp_path: Path, safetensors_dtype: str, expected: str
) -> None:
    """THE RED PROOF for the silent 2x VRAM."""

    built = fixture.build(tmp_path, dtype=safetensors_dtype)
    assert detect_on_disk_dtype(built.tree) == expected


def test_dtype_is_identical_projected_and_materialized(tmp_path: Path) -> None:
    """The projection must not change the answer — the point of the layout."""

    built = fixture.build(tmp_path)
    material = fixture.read_entry_tree(tmp_path, built)
    assert detect_on_disk_dtype(built.tree) == detect_on_disk_dtype(material) == "bf16"


def test_an_unresolvable_projection_REFUSES_rather_than_defaulting(
    tmp_path: Path,
) -> None:
    """The half that matters most: when the manifest cannot be recovered the answer is an exception, NOT ""."""

    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "model.safetensors").write_bytes(
        stub_bytes("b" * 64, 4096)
    )
    with pytest.raises(projection.UnresolvedProjection):
        detect_on_disk_dtype(orphan)


def test_a_projected_tree_whose_pin_is_gone_REFUSES(tmp_path: Path) -> None:
    """Same refusal via the production path: the tree is where it belongs but its manifest ref has been collected."""

    built = fixture.build(tmp_path)
    (tmp_path / "refs").rename(tmp_path / "refs-gone")
    (tmp_path / "refs").mkdir()
    with pytest.raises(projection.UnresolvedProjection):
        detect_on_disk_dtype(built.tree)


def test_dtype_of_a_tree_with_no_weights_is_still_empty(tmp_path: Path) -> None:
    """"" survives where it is HONEST: nothing to read is not a stub."""

    empty = tmp_path / "configs-only"
    empty.mkdir()
    (empty / "model_index.json").write_text("{}")
    assert detect_on_disk_dtype(empty) == ""


def test_a_quantized_artifact_is_still_detected_as_quantized(tmp_path: Path) -> None:
    """``_quantized_layers`` returning () routes an fp8 artifact to the plain bf16 lane."""

    from gen_worker.models.w8a8 import detect_w8a8_artifacts

    built = fixture.build(tmp_path, fp8=True)
    material = fixture.read_entry_tree(tmp_path, built)

    projected = detect_w8a8_artifacts(built.tree)
    baseline = detect_w8a8_artifacts(material)
    assert baseline, "the fixture is not a w8a8 artifact at all"
    assert [a.quantized for a in projected] == [a.quantized for a in baseline]
    assert [a.component for a in projected] == [a.component for a in baseline]


def test_component_weight_bytes_are_the_real_bytes(tmp_path: Path) -> None:
    """A stub read as zero data bytes plans VRAM for a model that is not there."""

    from gen_worker.models.loading import snapshot_component_weight_bytes

    built = fixture.build(tmp_path)
    material = fixture.read_entry_tree(tmp_path, built)
    projected = snapshot_component_weight_bytes(built.tree)
    assert projected == snapshot_component_weight_bytes(material)
    assert projected and all(v > 0 for v in projected.values())


def test_adapter_sizes_are_the_LOGICAL_sizes(tmp_path: Path) -> None:

    from gen_worker._vendor.tensorfs import LocalCAS, project_snapshot
    from cas_fixture import ingest_repository
    from gen_worker.models.projection import logical_size
    from gen_worker.utils.lora import find_adapter_file

    base = tmp_path / "store"
    source = base / "source-model"
    source.mkdir(parents=True)
    (source / "aaa-small.safetensors").write_bytes(
        fixture.safetensors_bytes({"w": ("BF16", (2, 8), fixture.varied(32, 4))})
    )
    (source / "zzz-large.safetensors").write_bytes(
        fixture.safetensors_bytes({"w": ("BF16", (64, 8), fixture.varied(1024, 3))})
    )

    cas = LocalCAS(base)
    manifest = ingest_repository(cas, source)
    cas.compare_and_swap_ref(
        projection.REF_PREFIX + "e" * 64, cas.store_manifest(manifest), expected=None
    )
    tree = base / projection.SNAPSHOTS_DIR / ("e" * 64)
    project_snapshot(cas, manifest, tree)

    for name in ("aaa-small.safetensors", "zzz-large.safetensors"):
        assert logical_size(tree / name) == (source / name).stat().st_size, name
        assert (tree / name).stat().st_size < 512, "the fixture is not projected"
    assert logical_size(tree / "aaa-small.safetensors") < logical_size(
        tree / "zzz-large.safetensors"
    )
    assert find_adapter_file(tree, ref="x").name == "zzz-large.safetensors"


def test_the_adapter_size_cap_still_REFUSES_over_a_stub(tmp_path: Path) -> None:
    """The consumer-level half, and the one with teeth."""

    from gen_worker.api.errors import ValidationError
    from gen_worker.utils.lora import MAX_LORA_FILE_BYTES, load_adapter_state_dict

    tree = tmp_path / "snapshots" / ("f" * 64)
    tree.mkdir(parents=True)
    oversize = tree / "adapter.safetensors"
    oversize.write_bytes(stub_bytes("a" * 64, MAX_LORA_FILE_BYTES + 1))
    assert oversize.stat().st_size < 512

    with pytest.raises(ValidationError, match="too large"):
        load_adapter_state_dict(oversize, ref="x")


def test_resolve_projection_recovers_the_manifest_from_the_tree_path(
    tmp_path: Path,
) -> None:
    built = fixture.build(tmp_path)
    resolved = projection.resolve_projection(built.tree)
    assert resolved is not None
    assert resolved.manifest == built.manifest


def test_resolve_projection_of_an_ordinary_directory_creates_no_store(
    tmp_path: Path,
) -> None:
    """Probing must not leave a CAS behind — it runs on every consumer path."""

    plain = tmp_path / "snapshots" / "not-a-snapshot"
    plain.mkdir(parents=True)
    assert projection.resolve_projection(plain) is None
    assert not (tmp_path / "objects").exists()
    assert not (tmp_path / "refs").exists()


def test_tensors_read_through_the_manifest_are_byte_exact(tmp_path: Path) -> None:
    """No tensor byte is at a file path, and the bytes are still the originals."""

    built = fixture.build(tmp_path)
    resolved = projection.require_projection(built.tree, why="test")
    with resolved.open_tensors() as reader:
        assert reader
        for name, view in reader.items():
            source = built.source / view.file
            raw = source.read_bytes()
            header_len = int.from_bytes(raw[:8], "little")
            header = json.loads(raw[8 : 8 + header_len])
            start, end = header[name]["data_offsets"]
            assert view.tobytes() == raw[8 + header_len + start : 8 + header_len + end]
    for path in built.weight_paths():
        assert projection.stub_at(path) is not None, f"{path} holds real bytes"
