"""The DERIVE half meets projected trees too, and it now says so.

# se#817, the incident this closes

On a rented CPU pod, `gen-worker download tensorhub/anima@composed-v3`
completed verified — 99 objects, 5,640,447,348 bytes, integrity-gated — the hub
manifest was sane, and `gen-worker lock` then died with::

    SafetensorError: Error while deserializing header: header too large

on `text_encoder/qwen_3_06b_base.safetensors`, a container the manifest sizes at
1,192,135,096 bytes: exactly right for Qwen3-0.6B at bf16. Nothing upstream was
wrong. `download` PROJECTS its snapshot (`cozy_snapshot.ensure_snapshot` ->
`project_snapshot`), `cli/workspace.resolve_checkpoint` hands `lock` that tree,
and `TraceLoadContext.load` handed it to the author's own `from_pretrained` —
which `safe_open`s the container and reads `b"TFSSTUB1"` as a little-endian u64
header length.

`serving.context.LoadContext.load` has guarded this since pgw#1513. The trace
half never did, so the one path that runs the AUTHOR'S loader against a real
hub tree was the one path with nothing to say about the tree's real shape.

# Why a DIAGNOSIS and not a refusal

A pre-check cannot tell a loader that reads stubs correctly (tier 1 via
`models.tensor_source`, tier 3 via `models.materialized_view`) from one that
does not, and refusing would break the correct one. So the diagnosis rides the
FAILURE — and only a failure. Every tree that carries real bytes raises what it
raised before, unchanged, which is the third case below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import projection_fixture as pf
from gen_worker.release.trace_context import (
    ProjectedTreeAtTrace,
    TraceLoadContext,
)


class _RawLoader:
    """An author loader that opens a container with the stock reader.

    This is anima's shape, and pgw#1578 measured it to be the fleet's only one.
    It is spelled out here rather than imported so the test states the defect
    instead of depending on an endpoint repo that is being fixed.
    """

    #: Set by whichever instance ran last, so a passing case can assert the
    #: loader really reached the bytes rather than being skipped.
    read: list[str] = []

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "_RawLoader":
        from safetensors import safe_open

        target = next(iter(sorted(Path(path).rglob("*.safetensors"))))
        with safe_open(str(target), framework="pt", device="cpu") as handle:
            cls.read = list(handle.keys())
        return cls()


class _NativeLoader:
    """The same author loader, cut over to tier 1 of pgw#1303's ladder."""

    read: list[str] = []

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "_NativeLoader":
        from gen_worker.models.tensor_source import open_tensor_source

        target = next(iter(sorted(Path(path).rglob("*.safetensors"))))
        with open_tensor_source(target, why="the pgw#1609 test's author loader") as h:
            cls.read = list(h.keys())
        return cls()


class _BrokenLoader:
    """An ordinary author bug, so the diagnosis can be shown NOT to fire."""

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "_BrokenLoader":
        raise ValueError("the author's own bug, on real bytes")


def _ctx(tree: Path) -> TraceLoadContext:
    return TraceLoadContext(lane=None, checkpoint_dir=tree)


def test_raw_loader_on_a_projected_tree_gets_the_stub_diagnosis(
    tmp_path: Path,
) -> None:
    """The se#817 failure, reproduced through the real derive seam."""

    fixture = pf.build(tmp_path)
    # The article, asserted rather than assumed: this tree really does hold
    # stubs and not weights.
    stubs = list(pf.iter_stubs(fixture.tree))
    assert stubs, "the fixture did not project — the rest proves nothing"
    # THE signature: a stub's size is fixed at ~128 B (it varies only by the
    # digit count of the size it names), so every container in a projected
    # tree is the same tiny size no matter how large the model behind it. That
    # coincidence is what a genuine truncation can never produce.
    assert all(120 <= path.stat().st_size <= 140 for path in stubs)
    assert len({path.stat().st_size for path in stubs}) <= 2

    with pytest.raises(ProjectedTreeAtTrace) as caught:
        _ctx(fixture.tree).load(_RawLoader)

    message = str(caught.value)
    # The author's real failure survives verbatim, as the first line and as
    # the __cause__ — a diagnosis that HID it would be the pgw#1308 mistake
    # arriving from the other side.
    assert "header too large" in message
    assert type(caught.value.__cause__).__name__ == "SafetensorError"
    # ...and the new information: what the tree actually is, and the ladder.
    assert "TFSSTUB1 pointer stubs" in message
    # Both sizes on the line, which is what identifies a stub on sight: tiny
    # on disk, huge in what it names.
    smallest = min(path.stat().st_size for path in stubs)
    assert f"{smallest} B on disk" in message
    assert " B)" in message and "names " in message
    assert "open_tensor_source" in message
    assert "third_party_dir" in message
    assert "se#817" in message


def test_a_stub_aware_loader_is_not_touched(tmp_path: Path) -> None:
    """The fix's acceptance: tier 1 reads the same tree and the trace passes.

    This is the case a pre-check refusal would have broken, which is why the
    diagnosis rides the failure instead of preceding the call.
    """

    fixture = pf.build(tmp_path)
    loaded = _ctx(fixture.tree).load(_NativeLoader)
    assert isinstance(loaded, _NativeLoader)
    assert _NativeLoader.read, "the loader returned without reading any tensor"


def test_third_party_dir_makes_the_same_container_openable(tmp_path: Path) -> None:
    """Tier 3, for the loader that will not be cut over.

    anima's text encoder and VAE go through DiffSynth's `ModelPool`, which
    calls `hash_model_file(path)` — a `safe_open` — BEFORE it looks at
    `ModelConfig.state_dict`. No argument skips that read, so the container has
    to become a real file. The seam is checked here because se#817's fix rests
    on it end to end, not just on tier 1.
    """

    from safetensors import safe_open

    from gen_worker.models.materialized_view import third_party_dir

    fixture = pf.build(tmp_path)
    stub = next(iter(pf.iter_stubs(fixture.tree)))
    rel = stub.relative_to(fixture.tree).as_posix()

    with pytest.raises(Exception, match="header too large"):
        with safe_open(str(stub), framework="pt", device="cpu"):
            pass

    real = Path(third_party_dir(stub, why="the pgw#1609 test's third party"))
    assert real != stub
    assert real.stat().st_size == len(pf.bytes_at(fixture.tree, rel))
    with safe_open(str(real), framework="pt", device="cpu") as handle:
        assert list(handle.keys())
    # Byte-for-byte the manifest's file, not merely a file that parses.
    assert real.read_bytes() == pf.bytes_at(fixture.tree, rel)


def test_an_ordinary_author_bug_is_unchanged(tmp_path: Path) -> None:
    """A tree of REAL bytes raises exactly what it always raised.

    The diagnosis is keyed on the tree holding stubs, never on the exception's
    text, so it cannot dress an unrelated failure in a stub story.
    """

    source = tmp_path / "plain"
    pf.write_model(source)

    with pytest.raises(ValueError, match="the author's own bug"):
        _ctx(source).load(_BrokenLoader)


def test_a_real_tree_that_fails_the_way_se817_did_is_still_not_claimed(
    tmp_path: Path,
) -> None:
    """A genuinely truncated container on a REAL tree keeps its own verdict.

    The signature that identified se#817 was that the failure was identical for
    a 1 GB container and a 68 GB one, because a stub is a fixed size. A real
    short write is a different fact and must keep reading as one.
    """

    source = tmp_path / "truncated"
    pf.write_model(source)
    # The loader opens the FIRST container in sorted order; truncate that one.
    shard = next(iter(sorted(source.rglob("*.safetensors"))))
    shard.write_bytes(shard.read_bytes()[:16])

    with pytest.raises(Exception) as caught:
        _ctx(source).load(_RawLoader)
    assert not isinstance(caught.value, ProjectedTreeAtTrace)


def test_the_census_counts_every_stub_and_truncates_the_list(
    tmp_path: Path,
) -> None:
    """Three named, the rest counted — the shape every long refusal here uses."""

    fixture = pf.build(tmp_path)
    stubs = list(pf.iter_stubs(fixture.tree))
    assert len(stubs) >= 3

    with pytest.raises(ProjectedTreeAtTrace) as caught:
        _ctx(fixture.tree).load(_RawLoader)
    message = str(caught.value)
    assert f"{len(stubs)} of its tensor containers" in message
    extra = len(stubs) - 3
    if extra > 0:
        assert f"(+{extra} more)" in message


def test_the_diagnosis_never_replaces_a_failure_with_its_own(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """If diagnosing raises, the author's exception wins. Falsified, not assumed."""

    from gen_worker.models import projection

    fixture = pf.build(tmp_path)

    def explode(_path: Any) -> Any:
        raise OSError("the diagnosis itself is broken")

    monkeypatch.setattr(projection, "stub_at", explode)
    with pytest.raises(Exception) as caught:
        _ctx(fixture.tree).load(_RawLoader)
    assert not isinstance(caught.value, ProjectedTreeAtTrace)
    assert "header too large" in str(caught.value)
