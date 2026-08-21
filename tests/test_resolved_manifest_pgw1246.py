from __future__ import annotations


import pytest

from gen_worker.models import cozy_snapshot, download
from gen_worker.models.refs import WireRef
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.wire_snapshots import index_snapshots


def _snap(digest: str, *paths: str) -> pb.Snapshot:
    return pb.Snapshot(
        digest=digest,
        files=[pb.SnapshotFile(path=p, size_bytes=1) for p in paths],
    )


def test_the_fetch_key_is_the_canonical_ref() -> None:
    """th#2208 leaves one ref-keyed channel for snapshot fetch facts."""
    wire = {
        "acme/sdxl@prod": _snap("d-a", "model_index.json", "vae/a.safetensors"),
        "acme/sdxl-r@prod": _snap("d-b", "model_index.json", "vae/b.safetensors"),
    }
    out = index_snapshots(wire)
    assert out[WireRef("acme/sdxl@prod")].digest == "d-a"
    assert out[WireRef("acme/sdxl-r@prod")].digest == "d-b"
    assert [f.path for f in out[WireRef("acme/sdxl@prod")].files] == [
        "model_index.json", "vae/a.safetensors"]


def test_an_artifact_with_no_composition_still_resolves_by_its_own_key() -> None:
    """LoRA overlays and payload source refs carry no ModelBinding, so the hub keys them by ref."""
    wire = {"acme/lora@prod": _snap("dl", "adapter.safetensors")}
    assert index_snapshots(wire) == {
        WireRef("acme/lora@prod"): wire["acme/lora@prod"]}


def test_the_snapshot_directory_IS_the_composed_digest() -> None:
    """One key, one meaning: fetch identity, dispatch key and directory name."""
    assert cozy_snapshot.snapshot_dir_key("sha256:cafe") == "sha256:cafe"
    assert cozy_snapshot.snapshot_dir_key(
        "sha256:cafe", ("vae",)) != "sha256:cafe"


def test_positive_component_selection_survives_and_negative_does_not() -> None:
    paths = ["model_index.json", "vae/a", "unet/b"]
    assert download.select_component_paths(paths, ("vae",)) == {
        "model_index.json", "vae/a"}
    assert download.select_component_paths(paths, ()) == set(paths)
    with pytest.raises(TypeError):
        download.select_component_paths(paths, (), ("vae",))  # type: ignore[call-arg]
