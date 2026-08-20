"""Reading a PROJECTED tree: who may, who must refuse, and what they say.

# pgw#1513: the eager bridge must refuse a projected tree, by name.

# The incident, and why it looked like a short write for two days

Two endpoints (sd15, sdxl), two volumes, 20x apart in size, identical
`SafetensorError: header too large`, hub source bytes verified intact. Every
short-write theory was dead on arrival once the write paths were read: the CAS
hashes and size-checks every object before committing it, tensorfs'
materializer re-hashes each object AND the whole file and refuses a short read,
and `project_snapshot` builds in a scratch directory and renames.

What actually happens is that a projected tree's tensor containers are ~128 B
TFSSTUB1 POINTER STUBS — the weights live in the CAS — and the stock
safetensors reader that `from_pretrained` uses knows nothing about stubs. It
reads the stub's first eight bytes as a header length and raises. The stub is a
FIXED SIZE regardless of the model behind it, which is exactly why a 3.4 GB and
a 68 GB checkpoint failed byte-identically, and that coincidence is what a
genuine truncation could never produce.

`ctx.load` is supposed to route a projected tree to the pgw#1380 streaming
engine, whose whole job is reading those stubs. The eager `from_pretrained`
bridge is documented for "a tree with no chunk store behind it — a bare
download, a local run, a fixture". Reaching the bridge WITH a projected tree
means the engine declined it, and the bridge then reports a corrupt checkpoint
for a checkpoint that is intact.

So the bridge refuses instead, naming the stub and its two sizes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gen_worker._vendor.tensorfs.project import stub_bytes
from gen_worker.serving.context import (
    DeployBinding,
    LoadContext,
    ProjectedTreeNotStreamable,
    _projection_artifacts,
)

# The two shapes from the field, so the fixed-size-stub signature is asserted
# rather than described.
_SD15_BYTES = 3_400_000_000
_SDXL_BYTES = 68_000_000_000


def _stubbed_tree(root: Path, named_bytes: int) -> Path:
    """A tree whose tensor container is a pointer, as `_project_entry` writes it."""
    tree = root / "snapshots" / ("sha256:" + "b7" * 32)
    (tree / "unet").mkdir(parents=True)
    (tree / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(
        stub_bytes("a" * 64, named_bytes)
    )
    (tree / "model_index.json").write_text('{"_class_name": "X"}')
    return tree


class _Pipeline:
    """A loader that would happily be called — so the refusal is the guard's,
    not an accident of the fixture."""

    called: list[Any] = []

    @classmethod
    def from_pretrained(cls, path: Any, **kwargs: Any) -> "_Pipeline":
        cls.called.append(path)
        from safetensors import safe_open

        # Exactly what diffusers does: the stock reader, on the container.
        with safe_open(
            str(Path(path) / "unet" / "diffusion_pytorch_model.safetensors"),
            framework="pt",
        ):
            pass
        return cls()


@pytest.mark.parametrize("named", [_SD15_BYTES, _SDXL_BYTES])
def test_the_eager_bridge_REFUSES_a_projected_tree_by_name(
    tmp_path: Path, named: int
) -> None:
    """Both field shapes, and the refusal must carry the numbers."""
    tree = _stubbed_tree(tmp_path, named)
    _Pipeline.called.clear()

    ctx: LoadContext[Any] = LoadContext(
        binding=DeployBinding(checkpoint_ref="acme/m@1", checkpoint_dir=tree),
        engine=None,  # the engine declined — the state the field pods were in
    )

    with pytest.raises(ProjectedTreeNotStreamable) as caught:
        ctx.load(_Pipeline)

    assert _Pipeline.called == [], (
        "the bridge must refuse BEFORE calling the loader — reaching "
        "from_pretrained is what produced `header too large`")
    message = str(caught.value)
    assert "PROJECTED" in message
    assert "unet/diffusion_pytorch_model.safetensors" in message, (
        f"the refusal must name the member: {message}")
    assert f"{named:,}" in message, (
        f"the refusal must name the bytes the stub STANDS FOR: {message}")
    assert "header too large" in message, (
        "the refusal must name the error it is replacing, or the next reader "
        f"searching that string will not find this: {message}")


def test_the_stub_signature_is_FIXED_SIZE_across_a_20x_model(tmp_path: Path) -> None:
    """The coincidence that killed every short-write theory, asserted.

    A truncation is proportional to what was being written; a stub is not. Two
    models 20x apart produce stubs within a byte or two of each other, which is
    why the two field errors were identical.
    """
    small = _projection_artifacts(_stubbed_tree(tmp_path / "a", _SD15_BYTES))
    large = _projection_artifacts(_stubbed_tree(tmp_path / "b", _SDXL_BYTES))

    assert len(small) == 1 and len(large) == 1
    (_, small_on_disk, small_named) = small[0]
    (_, large_on_disk, large_named) = large[0]

    assert large_named // small_named >= 19, "fixture is not a 20x spread"
    assert abs(large_on_disk - small_on_disk) <= 8, (
        f"stub sizes {small_on_disk} vs {large_on_disk} should be ~equal; that "
        "near-equality IS the signature that ruled out a short write")
    assert small_on_disk < 200 and large_on_disk < 200


def test_a_MATERIALIZED_tree_still_takes_the_eager_bridge(tmp_path: Path) -> None:
    """The guard must not fire on the trees the bridge legitimately serves.

    A bare download / fixture / local run has no chunk store and no stubs, and
    it is exactly what the eager bridge exists for. If this test ever fails,
    the guard has started refusing the local development path.
    """
    tree = tmp_path / "snapshots" / "plain"
    (tree / "unet").mkdir(parents=True)
    # Real (tiny) safetensors bytes, not a pointer.
    import torch
    from safetensors.torch import save_file

    save_file(
        {"w": torch.zeros(4)},
        str(tree / "unet" / "diffusion_pytorch_model.safetensors"),
    )
    (tree / "model_index.json").write_text('{"_class_name": "X"}')

    assert _projection_artifacts(tree) == []
    _Pipeline.called.clear()
    ctx: LoadContext[Any] = LoadContext(
        binding=DeployBinding(checkpoint_ref="acme/m@1", checkpoint_dir=tree),
        engine=None,
    )
    ctx.load(_Pipeline)
    assert _Pipeline.called == [tree], "the eager bridge must still run here"


def test_a_tree_with_a_MISSING_PIN_is_not_answered_as_resident(tmp_path: Path) -> None:
    """The one state that is byte-perfect and still unservable.

    `resolve_projection` recovers a tree's manifest through a `snapshot:<key>`
    ref keyed on the tree's own directory name. Without it the streaming
    engine cannot bind, `ctx.load` falls to the eager bridge, and the bridge
    reads a stub. Verification alone does not catch this: every byte is
    correct.

    So `announce_resident` refuses, and the refusal sends the ref through
    `ensure_local`, which re-pins WITHOUT moving bytes.
    """
    import asyncio

    from gen_worker.models import projection
    from gen_worker.models.refs import WireRef
    from gen_worker.models.store import ModelStore
    from gen_worker.pb import worker_scheduler_pb2 as pb
    from gen_worker._vendor.tensorfs.project import stub_bytes

    base = tmp_path / "cas"
    (base / "refs").mkdir(parents=True)
    (base / "objects").mkdir(parents=True)
    tree = base / "snapshots" / ("sha256:" + "c3" * 32)
    (tree / "unet").mkdir(parents=True)
    (tree / "unet" / "x.safetensors").write_bytes(stub_bytes("a" * 64, 3_400_000_000))

    # Byte-perfect and stubbed, but nothing pinned it.
    assert projection.stub_at_any(tree) is True
    assert projection.resolve_projection(tree) is None

    async def noop(_msg: Any) -> None:
        return None

    store = ModelStore(noop, cache_dir=base)
    snap = pb.Snapshot(digest="sha256:" + "c3" * 32)
    snap.files.add(path="unet/x.safetensors", size_bytes=3_400_000_000,
                   digest="sha256:" + "a" * 64)
    ref = WireRef("acme/unpinned")

    async def run() -> bool:
        return await store.announce_resident(ref, snap)

    answered = asyncio.run(run())
    assert answered is False, (
        "a tree no engine can bind to must not be answered as resident — "
        "that is the state that reaches the eager bridge and reports "
        "`header too large` about an intact checkpoint")


def test_a_tree_whose_OBJECTS_WERE_COLLECTED_says_so_and_not_malformed(
    tmp_path: Path,
) -> None:
    """se#790's state D, measured on a real 5.6 GB `@composed-v3` tree.

    A tree's manifest pin is the ONLY GC root its objects have. Drop the pin,
    run a GC, and every object is deleted while the tree stands: the tensor
    containers are still stubs and the NON-tensor files — `model_index.json`
    among them — become dangling symlinks. `skeleton.build` then reports
    "carries no model_index.json" about a tree that has one, which is a
    reader-level fact rendered as a claim about the checkpoint.

    The refusal must name the real condition, and must not be confused with
    the cheap missing-pin case: these bytes are GONE and must be re-fetched.
    """
    base = tmp_path / "cas"
    (base / "refs").mkdir(parents=True)
    objects = base / "objects"
    objects.mkdir(parents=True)
    tree = base / "snapshots" / ("sha256:" + "d4" * 32)
    tree.mkdir(parents=True)

    # A projected non-tensor entry whose object has been collected.
    (tree / "model_index.json").symlink_to(
        Path("..") / ".." / "objects" / "sha256" / "de" / "ad" / ("de" * 32)
    )
    assert (tree / "model_index.json").is_symlink()
    assert not (tree / "model_index.json").exists(), "fixture must be dangling"

    _Pipeline.called.clear()
    ctx: LoadContext[Any] = LoadContext(
        binding=DeployBinding(checkpoint_ref="acme/m@1", checkpoint_dir=tree),
        engine=None,
    )
    with pytest.raises(ProjectedTreeNotStreamable) as caught:
        ctx.load(_Pipeline)

    assert _Pipeline.called == []
    message = str(caught.value)
    assert "COLLECTED" in message, f"the real condition must be named: {message}"
    assert "model_index.json" in message
    assert "RE-FETCHED" in message, (
        "this must not be confused with the cheap re-pin case — the bytes are "
        f"gone: {message}")
    assert "carries no model_index.json" in message, (
        "the refusal must name the FALSE message it is pre-empting, so the "
        f"next reader searching that string lands here: {message}")


def test_collected_entries_puts_model_index_json_FIRST(tmp_path: Path) -> None:
    """se#790's refinement, and the reason it is not cosmetic.

    `model_index.json` is the entry whose absence produces the false "carries
    no model_index.json", so it is the one a reader most needs to see in the
    truncated list the refusal shows. Plain alphabetical order drops it out of
    that window on any tree with early-alphabet components — measured on a real
    `@composed-v3` tree, where it landed 2nd of 3 by luck of `dit/` sorting
    first. Everything after it keeps sorted order so the list stays diffable.
    """
    from gen_worker.models import projection

    for rel in ("model_index.json", "aaa/config.json", "dit/config.json",
                "text_encoder/config.json"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(Path("..") / "objects" / "sha256" / "de" / "ad" / ("de" * 32))

    got = projection.collected_entries(tmp_path)
    assert got[0] == "model_index.json", (
        f"the entry that causes the false refusal must lead the list: {got}")
    assert got[1:] == sorted(got[1:]), f"the tail must stay sorted: {got}"
    assert "model_index.json" in projection.collected_refusal(tmp_path, got)[:400], (
        "and it must survive into the truncated (shown) window of the message")


# ---------------------------------------------------------------------------
# pgw#1514: the STREAMING reader. Everything above is the eager bridge; this is
# the other caller, and it reads the index BEFORE anything that guard can see.
# ---------------------------------------------------------------------------


class _KeepsComponents:
    """A pipeline class that would build happily — so a refusal below is the
    guard's, never an accident of the fixture."""

    def __init__(self, **components: Any) -> None:  # pragma: no cover
        for name, value in components.items():
            setattr(self, name, value)


def _collected_tree(root: Path, *, entries: tuple[str, ...]) -> Path:
    """A projected tree whose objects have been collected: every entry is a
    dangling link into `objects/`, exactly as one GC pass after a lost pin
    leaves it."""
    base = root / "cas"
    (base / "refs").mkdir(parents=True)
    (base / "objects").mkdir(parents=True)
    tree = base / "snapshots" / ("sha256:" + "c0" * 32)
    tree.mkdir(parents=True)
    for rel in entries:
        path = tree / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        depth = len(Path(rel).parts)
        up = Path(*([".."] * (depth + 1)))
        path.symlink_to(up / "objects" / "sha256" / "de" / "ad" / ("de" * 32))
        assert path.is_symlink() and not path.exists(), rel
    return tree


def test_skeleton_build_does_not_call_a_collected_tree_index_less(
    tmp_path: Path,
) -> None:
    """THE pgw#1514 REGRESSION. `Path.is_file()` follows the link, so before
    this fix a dangling `model_index.json` and an absent one were the same
    False — and `skeleton.build` reported the store's condition as the
    checkpoint's shape."""
    from gen_worker.serving.streaming import skeleton as sk

    tree = _collected_tree(
        tmp_path, entries=("model_index.json", "dit/config.json", "vae/config.json")
    )
    with pytest.raises(sk.SkeletonError) as caught:
        sk.build(_KeepsComponents, tree)

    message = str(caught.value)
    assert "COLLECTED" in message, message
    assert "RE-FETCHED" in message, message
    assert not message.startswith(f"{tree} carries no model_index.json"), (
        "the false refusal is exactly what this issue removed")


def test_skeleton_build_walks_the_WHOLE_tree_not_just_the_index(
    tmp_path: Path,
) -> None:
    """The index merely DIES FIRST. 14 entries dangle in the measured state, so
    guarding one file would relocate the wrong message to the next component's
    config rather than remove it. Here the index is FINE and a component config
    is collected — the pre-fix code reached `_build_on_meta` and failed there
    with a message about a missing config."""
    from gen_worker.serving.streaming import skeleton as sk

    tree = _collected_tree(tmp_path, entries=("dit/config.json",))
    (tree / "model_index.json").write_text(
        '{"_class_name": "X", "dit": ["anima.components", "AnimaDiTComponent"]}'
    )

    with pytest.raises(sk.SkeletonError) as caught:
        sk.build(_KeepsComponents, tree)
    assert "COLLECTED" in str(caught.value), str(caught.value)
    assert "dit/config.json" in str(caught.value)


def test_a_tree_that_GENUINELY_has_no_index_still_says_so(tmp_path: Path) -> None:
    """The original wording SURVIVES and is now true whenever it is reached. A
    fix that swallowed the genuine case would trade one wrong message for
    another, which is the mistake this issue is about."""
    from gen_worker.serving.streaming import skeleton as sk

    bare = tmp_path / "bare"
    (bare / "unet").mkdir(parents=True)
    (bare / "unet" / "config.json").write_text("{}")

    with pytest.raises(sk.SkeletonError) as caught:
        sk.build(_KeepsComponents, bare)
    assert "carries no model_index.json" in str(caught.value)
    assert "COLLECTED" not in str(caught.value)
