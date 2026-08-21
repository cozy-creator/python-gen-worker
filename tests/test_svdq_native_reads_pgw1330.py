from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from gen_worker._vendor.tensorfs import LocalCAS, project_snapshot  # noqa: E402
from gen_worker.cas import ingest_file  # noqa: E402
from gen_worker.models import projection  # noqa: E402
from gen_worker.models import svdq_native as native  # noqa: E402
from gen_worker.models.projection import REF_PREFIX, SNAPSHOTS_DIR  # noqa: E402
from gen_worker.models.svdq import _read_safetensors_metadata  # noqa: E402
from gen_worker.models.tensor_source import open_tensor_source  # noqa: E402
from test_svdq_load_device import _Art, _write_multiunit  # noqa: E402


def _same_bytes(a: Any, b: Any) -> bool:

    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(
        torch.equal(
            a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
        )
    )


def _project(checkpoint: Path, base: Path, key: str = "c" * 64) -> Path:

    cas = LocalCAS(base)
    entry = ingest_file(cas, checkpoint, manifest_path=checkpoint.name)
    from gen_worker._vendor.tensorfs import RepositoryManifest

    manifest = RepositoryManifest((entry,))
    cas.compare_and_swap_ref(
        REF_PREFIX + key, cas.store_manifest(manifest), expected=None
    )
    tree = base / SNAPSHOTS_DIR / key
    project_snapshot(cas, manifest, tree)
    return tree / entry.path


def test_the_projected_checkpoint_holds_no_tensor_bytes(tmp_path: Path) -> None:
    checkpoint, _state, _dim = _write_multiunit(tmp_path)
    stubbed = _project(checkpoint, tmp_path / "store")
    stub = projection.stub_at(stubbed)
    assert stub is not None, "the projected checkpoint is not a stub"
    assert stubbed.stat().st_size < 512
    assert stub.size == checkpoint.stat().st_size


def test_metadata_is_served_from_the_manifest(tmp_path: Path) -> None:
    """The metadata read that DECIDES this is an svdq checkpoint at all."""

    checkpoint, _state, _dim = _write_multiunit(tmp_path)
    stubbed = _project(checkpoint, tmp_path / "store")
    from_file = _read_safetensors_metadata(checkpoint)
    from_stub = _read_safetensors_metadata(stubbed)
    assert from_file and from_file == from_stub
    assert from_stub["model_class"] == "QwenImageTransformer2DModel"
    assert json.loads(from_stub["config"])


def test_tensor_source_serves_the_same_tensors_from_both(tmp_path: Path) -> None:
    checkpoint, _state, _dim = _write_multiunit(tmp_path)
    stubbed = _project(checkpoint, tmp_path / "store")
    with open_tensor_source(checkpoint, why="test") as direct:
        with open_tensor_source(stubbed, why="test") as native_source:
            assert sorted(direct.keys()) == sorted(native_source.keys())
            for name in sorted(direct.keys()):
                a, b = direct.get_tensor(name), native_source.get_tensor(name)
                assert a.dtype == b.dtype and a.shape == b.shape, name
                assert _same_bytes(a, b), name


def test_the_denoiser_from_a_projected_tree_is_bit_identical(tmp_path: Path) -> None:
    """THE CUT, through the real consumer."""

    checkpoint, _state, _dim = _write_multiunit(tmp_path)
    stubbed = _project(checkpoint, tmp_path / "store")

    from_file = native.load_svdq_native_denoiser(
        _Art(checkpoint), mode="dense", device="cpu"
    )
    from_tree = native.load_svdq_native_denoiser(
        _Art(stubbed), mode="dense", device="cpu"
    )

    want: Dict[str, Any] = dict(from_file.named_parameters())
    want.update(dict(from_file.named_buffers()))
    got: Dict[str, Any] = dict(from_tree.named_parameters())
    got.update(dict(from_tree.named_buffers()))
    assert want and set(want) == set(got)
    for name in sorted(want):
        assert got[name].device.type != "meta", name
        assert _same_bytes(want[name], got[name]), name


def test_a_stub_outside_a_snapshot_tree_REFUSES(tmp_path: Path) -> None:
    """No silent fallback: a stub whose manifest cannot be found is an error, not an empty key list that yields an all-meta model."""

    checkpoint, _state, _dim = _write_multiunit(tmp_path)
    stubbed = _project(checkpoint, tmp_path / "store")
    orphan = tmp_path / "orphan.safetensors"
    orphan.write_bytes(stubbed.read_bytes())
    with pytest.raises(projection.UnresolvedProjection):
        with open_tensor_source(orphan, why="test") as source:
            source.keys()
