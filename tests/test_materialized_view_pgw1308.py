"""pgw#1308 step ⑥: the pgw#1303 gate's mechanism, and what it costs.

The flip makes every snapshot a projected tree. A site that hands a DIRECTORY
to a third party (`diffusers.from_pretrained`, `ComponentSpec.load`,
`from_single_file`, `gguf.GGUFReader`, `llama-server -m`) is handing it pointer
stubs, and nothing in this repo can change those loaders. Until pgw#1303 rules,
they stay materialized — through ONE seam, per subtree, priced.

What each arm holds is a property the old unconditional whole-tree copy did
NOT have, so none of them is a restatement of the status quo.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

import projection_fixture as fixture
from gen_worker.models import disk_gc, materialized_view, projection
from gen_worker.models.materialized_view import third_party_dir, view_root_for


def _du(root: Path) -> int:
    total = 0
    for directory, _subdirs, names in os.walk(root, followlinks=False):
        for name in names:
            info = os.lstat(os.path.join(directory, name))
            if not stat.S_ISLNK(info.st_mode):
                total += info.st_size
    return total


def test_a_gated_site_gets_real_files_out_of_a_projected_tree(
    tmp_path: Path,
) -> None:
    """The whole point: `from_pretrained` can read what it is handed."""

    built = fixture.build(tmp_path)
    shard = built.tree / "unet" / "diffusion_pytorch_model.safetensors"
    assert projection.stub_at(shard) is not None, "fixture is not projected"

    real = third_party_dir(built.tree, why="test: diffusers from_pretrained")

    assert real != built.tree
    for entry in built.manifest.files:
        made = real / entry.path
        assert not made.is_symlink()
        assert projection.stub_at(made) is None
        assert made.stat().st_size == entry.size_bytes
        assert made.read_bytes() == (built.source / entry.path).read_bytes()


def test_a_component_costs_the_component_and_not_the_model(
    tmp_path: Path,
) -> None:
    """The change that lands BEFORE Paul's ruling, and the reason it matters.

    Materialization used to be unconditional and whole-tree: every snapshot on
    every pod carried a second complete copy. Now a gated site that wants one
    component pays for one component. Asserted as bytes, because "it is
    narrower" is the kind of claim that is true in a diff and false on a pod.
    """

    built = fixture.build(tmp_path)
    whole = sum(entry.size_bytes for entry in built.manifest.files)
    component = sum(
        entry.size_bytes
        for entry in built.manifest.files
        if entry.path.startswith("unet/")
    )
    assert 0 < component < whole  # or the arm proves nothing

    made = third_party_dir(built.tree / "unet", why="test: one component")

    assert _du(view_root_for(built.tree)) == component
    assert _du(view_root_for(built.tree)) < whole
    assert (made / "config.json").exists()
    assert not (view_root_for(built.tree) / "vae").exists()


def test_the_seam_is_a_no_op_on_a_tree_that_is_not_projected(
    tmp_path: Path,
) -> None:
    """Call sites are unconditional on purpose.

    A caller forced to ask "is this projected?" is a caller that gets the
    branch wrong somewhere — an HF cache directory, a bare download, a test
    fixture. So the seam answers with the path it was given, and nothing is
    copied.
    """

    ordinary = tmp_path / "hf-cache" / "models--acme--x" / "snapshots" / "abc"
    ordinary.mkdir(parents=True)
    (ordinary / "config.json").write_text(json.dumps({"a": 1}))

    assert third_party_dir(ordinary, why="test: not projected") == ordinary
    assert third_party_dir(
        ordinary / "config.json", why="test: not projected"
    ) == (ordinary / "config.json")


def test_a_view_is_idempotent_and_built_once(tmp_path: Path) -> None:
    """A second ask returns the same files without copying them again."""

    built = fixture.build(tmp_path)
    first = third_party_dir(built.tree, why="test: first")
    marker = (first / "model_index.json").stat().st_ino
    before = _du(view_root_for(built.tree))

    second = third_party_dir(built.tree, why="test: second")

    assert second == first
    assert (second / "model_index.json").stat().st_ino == marker
    assert _du(view_root_for(built.tree)) == before


def test_a_view_dies_with_the_snapshot_it_belongs_to(tmp_path: Path) -> None:
    """Otherwise it is disk nothing can name.

    `delete_ref_bytes` reclaims a ref by removing its snapshot tree. A view is
    keyed by that tree and reachable only through it, so a view that survived
    would be bytes no ref accounts for and no sweep collects.
    """

    built = fixture.build(tmp_path)
    third_party_dir(built.tree, why="test: view to be reclaimed")
    view = view_root_for(built.tree)
    assert view.is_dir() and _du(view) > 0

    disk_gc.delete_ref_bytes("acme/model", built.tree, tmp_path)

    assert not built.tree.exists()
    assert not view.exists()


def test_disk_gc_sizes_a_projected_tree_from_its_manifest(
    tmp_path: Path,
) -> None:
    """A walk of a projected tree answers ~0 for a model of any size.

    Every caller of `tree_bytes` is asking how much disk a ref holds, and
    eviction acts on the answer. Answering ~0 does not fail loudly — the pod
    simply believes a resident model frees nothing, and fills up.
    """

    built = fixture.build(tmp_path)
    model = sum(entry.size_bytes for entry in built.manifest.files)

    assert _du(built.tree) < 4096  # what a walk would have reported
    assert disk_gc.tree_bytes(built.tree) == model

    third_party_dir(built.tree / "unet", why="test: priced copy counts")
    component = sum(
        entry.size_bytes
        for entry in built.manifest.files
        if entry.path.startswith("unet/")
    )
    assert disk_gc.tree_bytes(built.tree) == model + component


def test_a_path_the_manifest_does_not_cover_is_a_refusal(tmp_path: Path) -> None:
    """Refusing to guess, rather than handing back an empty directory.

    An empty directory is what `from_pretrained` reports as "no such model",
    which sends the investigation to the catalog instead of to the store.
    """

    built = fixture.build(tmp_path)
    with pytest.raises(projection.UnresolvedProjection) as refusal:
        third_party_dir(built.tree / "no-such-component", why="test: absent")
    assert "no-such-component" in str(refusal.value)


def test_one_file_is_a_file_and_not_a_directory_holding_it(
    tmp_path: Path,
) -> None:
    """`from_single_file` and `GGUFReader` name a FILE.

    The directory arm builds under a scratch and renames; a single file must
    not go through it, or the seam hands back a directory where the caller
    asked for a checkpoint.
    """

    built = fixture.build(tmp_path)
    rel = "unet/diffusion_pytorch_model.safetensors"
    made = third_party_dir(built.tree / rel, why="test: from_single_file")

    assert made.is_file()
    assert made.read_bytes() == (built.source / rel).read_bytes()


def test_the_seam_is_the_only_way_the_hatch_is_reached(tmp_path: Path) -> None:
    """A state assertion about the census, not an intent.

    This pins the fact that the `author-slot-directory` hatch row has exactly
    one first-party occupant, so a second one is a decision someone makes on
    purpose.
    """

    source = Path(materialized_view.__file__).read_text()
    assert source.count("# mixed-cas-hatch: author-slot-directory") == 1
    assert "cas.materialize(" in source
