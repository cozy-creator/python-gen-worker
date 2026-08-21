"""The DERIVE half meets projected trees too, and it now says so."""

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

    read: list[str] = []

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "_RawLoader":
        from safetensors import safe_open

        target = next(iter(sorted(Path(path).rglob("*.safetensors"))))
        with safe_open(str(target), framework="pt", device="cpu") as handle:
            cls.read = list(handle.keys())
        return cls()


class _NativeLoader:

    read: list[str] = []

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "_NativeLoader":
        from gen_worker.models.tensor_source import open_tensor_source

        target = next(iter(sorted(Path(path).rglob("*.safetensors"))))
        with open_tensor_source(target, why="the pgw#1609 test's author loader") as h:
            cls.read = list(h.keys())
        return cls()


class _BrokenLoader:

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "_BrokenLoader":
        raise ValueError("the author's own bug, on real bytes")


def _ctx(tree: Path) -> TraceLoadContext:
    return TraceLoadContext(lane=None, checkpoint_dir=tree)


def test_raw_loader_on_a_projected_tree_gets_the_stub_diagnosis(
    tmp_path: Path,
) -> None:

    fixture = pf.build(tmp_path)
    stubs = list(pf.iter_stubs(fixture.tree))
    assert stubs, "the fixture did not project — the rest proves nothing"
    assert all(120 <= path.stat().st_size <= 140 for path in stubs)
    assert len({path.stat().st_size for path in stubs}) <= 2

    with pytest.raises(ProjectedTreeAtTrace) as caught:
        _ctx(fixture.tree).load(_RawLoader)

    message = str(caught.value)
    assert "header too large" in message
    assert type(caught.value.__cause__).__name__ == "SafetensorError"
    assert "TFSSTUB1 pointer stubs" in message
    smallest = min(path.stat().st_size for path in stubs)
    assert f"{smallest} B on disk" in message
    assert " B)" in message and "names " in message
    assert "open_tensor_source" in message
    assert "third_party_dir" in message
    assert "se#817" in message


def test_a_stub_aware_loader_is_not_touched(tmp_path: Path) -> None:
    """The fix's acceptance: tier 1 reads the same tree and the trace passes."""

    fixture = pf.build(tmp_path)
    loaded = _ctx(fixture.tree).load(_NativeLoader)
    assert isinstance(loaded, _NativeLoader)
    assert _NativeLoader.read, "the loader returned without reading any tensor"


def test_third_party_dir_makes_the_same_container_openable(tmp_path: Path) -> None:
    """Tier 3, for the loader that will not be cut over."""

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
    assert real.read_bytes() == pf.bytes_at(fixture.tree, rel)


def test_an_ordinary_author_bug_is_unchanged(tmp_path: Path) -> None:
    """A tree of REAL bytes raises exactly what it always raised."""

    source = tmp_path / "plain"
    pf.write_model(source)

    with pytest.raises(ValueError, match="the author's own bug"):
        _ctx(source).load(_BrokenLoader)


def test_a_real_tree_that_fails_the_way_se817_did_is_still_not_claimed(
    tmp_path: Path,
) -> None:
    """A genuinely truncated container on a REAL tree keeps its own verdict."""

    source = tmp_path / "truncated"
    pf.write_model(source)
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
    """If diagnosing raises, the author's exception wins."""

    from gen_worker.models import projection

    fixture = pf.build(tmp_path)

    def explode(_path: Any) -> Any:
        raise OSError("the diagnosis itself is broken")

    monkeypatch.setattr(projection, "stub_at", explode)
    with pytest.raises(Exception) as caught:
        _ctx(fixture.tree).load(_RawLoader)
    assert not isinstance(caught.value, ProjectedTreeAtTrace)
    assert "header too large" in str(caught.value)
