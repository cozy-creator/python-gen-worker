from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from gen_worker._vendor.tensorfs import (
    CASRef,
    Chunk,
    FileEntry,
    LocalCAS,
    RepositoryManifest,
    project_snapshot,
    read_entry,
    tree_bytes,
)

import projection_fixture as fixture
from gen_worker.models import cozy_snapshot, projection
from gen_worker.models.cozy_snapshot import ensure_snapshot_async
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef
from gen_worker.models.store import ModelStore
from gen_worker.models.volume_verify import (
    VerifyTarget,
    split_projection_targets,
    verify_projection,
)

_SHARD_TENSOR_BYTES = 1 << 20


def _ref() -> TensorhubRef:
    return TensorhubRef(owner="acme", repo="model", release="latest")


def _write_model(source: Path) -> None:

    source.mkdir(parents=True, exist_ok=True)
    (source / "model_index.json").write_text(
        json.dumps({"_class_name": "TinyPipeline", "transformer": ["diffusers", "X"]})
    )
    for component in ("transformer", "vae"):
        directory = source / component
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps({"_class_name": "X"}))
        (directory / "diffusion_pytorch_model.safetensors").write_bytes(
            fixture.safetensors_bytes(
                {
                    f"{component}.weight": (
                        "F32",
                        (_SHARD_TENSOR_BYTES // 4,),
                        fixture.varied(_SHARD_TENSOR_BYTES, 7 + len(component)),
                    )
                }
            )
        )
    tokenizer = source / "tokenizer"
    tokenizer.mkdir(parents=True, exist_ok=True)
    (tokenizer / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 77}))
    media = source / "dataset"
    media.mkdir(parents=True, exist_ok=True)
    (media / "sample.mp4").write_bytes(fixture.varied(4096, 11))
    (media / "empty.txt").write_bytes(b"")


def _resident(base: Path, source: Path) -> WorkerResolvedRepo:

    cas = LocalCAS(base)
    files: list[WorkerResolvedRepoFile] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        body = path.read_bytes()
        digest = cas.put_bytes(body)
        files.append(
            WorkerResolvedRepoFile(
                path.relative_to(source).as_posix(),
                len(body),
                "http://127.0.0.1:1/must-not-fetch",
                digest=str(digest),
            )
        )
    return WorkerResolvedRepo(snapshot_digest="sha256:" + "e" * 64, files=files)


def _du(root: Path) -> int:

    total = 0
    for directory, _subdirs, names in os.walk(root, followlinks=False):
        for name in names:
            info = os.lstat(os.path.join(directory, name))
            if not stat.S_ISLNK(info.st_mode):
                total += info.st_size
    return total


@pytest.fixture
def published(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "source-model"
    _write_model(source)
    base = tmp_path / "store"
    base.mkdir()
    resolved = _resident(base, source)
    tree = asyncio.run(
        ensure_snapshot_async(base_dir=base, ref=_ref(), resolved=resolved)
    )
    return {"base": base, "source": source, "tree": tree}


def test_the_chokepoint_publishes_symlinks_and_stubs(published: dict[str, Path]) -> None:
    """Every file in the tree is a pointer."""

    tree, source = published["tree"], published["source"]
    seen: set[str] = set()
    for path in sorted(tree.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        rel = path.relative_to(tree).as_posix()
        seen.add(rel)
        original = source / rel
        if rel.endswith(".safetensors"):
            stub = projection.stub_at(path)
            assert stub is not None, f"{rel} is not a pointer stub"
            assert stub.size == original.stat().st_size
            assert stub.body_sha256 == hashlib.sha256(original.read_bytes()).hexdigest()
        elif original.stat().st_size == 0:
            assert path.read_bytes() == b""
        else:
            assert path.is_symlink(), f"{rel} is a copy, not a symlink"
            assert projection.object_of_symlink(path) is not None
            assert path.read_bytes() == original.read_bytes()
    assert seen == {
        p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()
    }


def test_no_tensor_byte_is_at_a_filesystem_path(published: dict[str, Path]) -> None:
    """The claim the whole layout exists to make true, asserted directly."""

    tree = published["tree"]
    for path in sorted(tree.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        assert path.stat().st_size < 4096, f"{path} holds {path.stat().st_size} bytes"
    assert tree_bytes(tree) < 4096


def test_a_resident_model_occupies_disk_ONCE(published: dict[str, Path]) -> None:

    base, tree = published["base"], published["tree"]
    snapshot = projection.require_projection(tree, why="pgw#1308 residency arm")
    objects = _du(base / "objects")
    projected_total = _du(base)

    control = base / "control-materialized"
    control.mkdir()
    for entry in snapshot.manifest.files:
        destination = control / entry.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(read_entry(snapshot.cas, entry))
    materialized_total = projected_total + _du(control)

    model = sum(entry.size_bytes for entry in snapshot.manifest.files)
    assert objects >= model
    assert projected_total < objects + 4096
    assert materialized_total > objects + model - 4096
    assert projected_total / model < 1.01
    assert materialized_total / model > 1.99


def test_the_published_tree_resolves_to_its_own_manifest(
    published: dict[str, Path]
) -> None:
    """`resolve_projection` needs the ref the chokepoint pins, and gets it."""

    tree = published["tree"]
    snapshot = projection.resolve_projection(tree)
    assert snapshot is not None
    assert snapshot.root == tree
    assert {entry.path for entry in snapshot.manifest.files} == {
        p.relative_to(published["source"]).as_posix()
        for p in published["source"].rglob("*")
        if p.is_file()
    }


def test_weights_read_byte_exact_out_of_the_published_tree(
    published: dict[str, Path]
) -> None:
    """The tree serves the model, or it is a smaller way of serving nothing."""

    tree, source = published["tree"], published["source"]
    snapshot = projection.require_projection(tree, why="pgw#1308 read-back arm")
    for component in ("transformer", "vae"):
        rel = f"{component}/diffusion_pytorch_model.safetensors"
        assert read_entry(snapshot.cas, snapshot.entry(rel)) == (
            (source / rel).read_bytes()
        )
    with snapshot.open_tensors() as reader:
        for component in ("transformer", "vae"):
            view = reader[f"{component}.weight"]
            assert view.nbytes == _SHARD_TENSOR_BYTES
            assert view.tobytes() == fixture.varied(
                _SHARD_TENSOR_BYTES, 7 + len(component)
            )


def test_boot_verification_of_the_published_tree_is_clean(
    published: dict[str, Path]
) -> None:

    tree = published["tree"]
    snapshot = projection.require_projection(tree, why="pgw#1308 boot arm")
    targets = [
        VerifyTarget(
            path=tree / entry.path,
            ref=str(entry.digest),
            size=entry.size_bytes,
            label=str(entry.digest),
        )
        for entry in snapshot.manifest.files
    ]
    projected, material = split_projection_targets(targets)
    assert {t.path.name for t in material} <= {"empty.txt"}
    report = verify_projection(projected)
    assert report.ok
    assert report.projected == len(projected)
    assert report.hashed == 0

    ok, bad = ModelStore._verify_snapshot_tree(
        object.__new__(ModelStore), tree, None
    )
    assert (ok, bad) == (True, [])


def test_the_headroom_prediction_is_what_the_projection_writes(
    tmp_path: Path
) -> None:

    base = tmp_path / "store"
    cas = LocalCAS(base)
    source = tmp_path / "source"
    _write_model(source)
    parts = [fixture.varied(2048, 3), fixture.varied(2048, 5)]
    for part in parts:
        cas.put_bytes(part)
    chunked = b"".join(parts)

    entries = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        body = path.read_bytes()
        entries.append(
            FileEntry(
                path.relative_to(source).as_posix(),
                len(body),
                cas.put_bytes(body),
            )
        )
    entries.append(
        FileEntry(
            "dataset/chunked.mp4",
            len(chunked),
            CASRef.digest_bytes(chunked),
            tuple(Chunk(CASRef.digest_bytes(part), len(part)) for part in parts),
        )
    )
    manifest = RepositoryManifest(tuple(entries))

    shapes = {
        "stub": any(e.path.endswith(".safetensors") for e in manifest.files),
        "empty": any(e.size_bytes == 0 for e in manifest.files),
        "symlink": any(
            not e.chunks and e.size_bytes and not e.path.endswith(".safetensors")
            for e in manifest.files
        ),
        "chunked": any(len(e.chunks) > 1 for e in manifest.files),
    }
    assert all(shapes.values()), shapes

    predicted = projection.projection_write_bytes(manifest, symlinks=True)
    tree = base / "snapshots" / "predicted"
    project_snapshot(cas, manifest, tree, symlinks=True)

    assert predicted == tree_bytes(tree)
    assert predicted < sum(e.size_bytes for e in manifest.files) // 4


def test_whole_tree_materialization_is_unreachable_from_the_chokepoint() -> None:
    """A state assertion, not an intent: the wrapper is GONE, not disabled."""

    assert not hasattr(cozy_snapshot, "_materialize_repository")
    source = Path(cozy_snapshot.__file__).read_text()
    assert "materialize" + "_repository" not in source
