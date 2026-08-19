"""The mint's INPUT: where a serialized ExportedProgram is found, and refused.

# pgw#1462 part 2: the image bakes the release's graph blobs and the pod's own
# CAS is a different directory, so the miner's content-address fallthrough has
# never reached them.

The defect these fence is a SILENT one, which is why it survived: a cache miss
reports nothing, so a tier looking in the wrong directory and a release with no
blobs at all produce the same observable — an eager pod.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gen_worker._vendor.tensorfs import LocalCAS
from gen_worker.models.cache_paths import BAKED_PROGRAM_CAS_DIR, baked_program_cas_dir
from gen_worker.serving.mint_store import (
    ProgramBlobUnreachable,
    TieredGraphStore,
    worker_store,
)

PROGRAM = b"\x80\x05serialized-exported-program-bytes" * 64


def _seed(root: Path, blob: bytes = PROGRAM) -> str:
    """Put ``blob`` in a real LocalCAS and return its ``sha256:`` address."""
    cas = LocalCAS(root)
    ref = cas.put_bytes(blob)
    digest = f"sha256:{hashlib.sha256(blob).hexdigest()}"
    assert str(ref) in (digest, digest.split(":", 1)[1]), f"unexpected ref {ref}"
    return digest


def _local_tier(root: Path) -> object:
    """A stand-in for LocalGraphStore: the mint reads only its `.cas`."""

    class _Store:
        cas = LocalCAS(root)

    return _Store()


def test_a_baked_blob_is_unreachable_without_the_image_tier(tmp_path: Path) -> None:
    """THE RED ARM, and it is the whole defect in four lines.

    The blob is present, on this machine, in the directory the builder bakes
    to — and the pod cannot see it, because the pod's CAS is somewhere else.
    """
    image_cas = tmp_path / "app" / ".tensorhub" / "derive-cas"
    digest = _seed(image_cas)
    pod_cas = tmp_path / "tensorhub-cache" / "cas"

    without = TieredGraphStore(_local_tier(pod_cas), upstream=None, baked=None)
    with pytest.raises(ProgramBlobUnreachable) as refusal:
        without.fetch_program(digest, tmp_path / "out" / "p.pt2")
    # The message has to name the ordinary cause, because the ordinary cause is
    # a build that used a committed lock and therefore baked nothing.
    assert "COMMITTED endpoint.lock" in str(refusal.value)

    # …and the GREEN arm is the same store with the tier wired.
    with_tier = TieredGraphStore(
        _local_tier(pod_cas), upstream=None, baked=LocalCAS(image_cas)
    )
    got = with_tier.fetch_program(digest, tmp_path / "out" / "p.pt2")
    assert Path(got).read_bytes() == PROGRAM


def test_the_pods_own_cas_still_wins_and_needs_no_image(tmp_path: Path) -> None:
    """The new tier is additive: a pod that already holds the blob never looks."""
    pod_cas = tmp_path / "cas"
    digest = _seed(pod_cas)
    store = TieredGraphStore(_local_tier(pod_cas), upstream=None, baked=None)
    assert Path(store.fetch_program(digest, tmp_path / "p.pt2")).read_bytes() == PROGRAM


def test_corrupted_baked_bytes_are_refused_not_compiled(tmp_path: Path) -> None:
    """`contains` is presence, NOT integrity — it says so itself.

    These bytes go straight into a compiler, so the store's own scrub runs
    first. Corrupting the object IN PLACE is the case `contains` documents as
    answering True by design, which is exactly the one that would otherwise
    reach inductor.
    """
    image_cas = tmp_path / "derive-cas"
    digest = _seed(image_cas)
    cas = LocalCAS(image_cas)
    target = Path(cas.object_path(digest))
    target.chmod(0o644)
    target.write_bytes(b"not the graph the release stamped" * 64)

    store = TieredGraphStore(
        _local_tier(tmp_path / "pod"), upstream=None, baked=LocalCAS(image_cas)
    )
    with pytest.raises(ProgramBlobUnreachable) as refusal:
        store.fetch_program(digest, tmp_path / "p.pt2")
    assert "integrity scrub" in str(refusal.value)

    # NEGATIVE CONTROL: without the corruption the same call succeeds, so the
    # assertion above is about the scrub and not about a broken fixture.
    clean = tmp_path / "clean-cas"
    clean_digest = _seed(clean)
    ok = TieredGraphStore(
        _local_tier(tmp_path / "pod2"), upstream=None, baked=LocalCAS(clean)
    )
    assert Path(ok.fetch_program(clean_digest, tmp_path / "ok.pt2")).exists()


def test_an_upstream_that_can_serve_one_still_wins(tmp_path: Path) -> None:
    """The tier order is a COST decision; every tier is content-addressed.

    A hub that grows `fetch_program` must not be shadowed by a baked blob, and
    a baked blob must not be shadowed by a hub that has none.
    """
    calls: list[str] = []

    class _Upstream:
        def fetch_program(self, digest: str, destination: Path) -> Path:
            calls.append(digest)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(PROGRAM)
            return destination

    image_cas = tmp_path / "derive-cas"
    digest = _seed(image_cas)
    store = TieredGraphStore(
        _local_tier(tmp_path / "pod"), upstream=_Upstream(), baked=LocalCAS(image_cas)
    )
    store.fetch_program(digest, tmp_path / "p.pt2")
    assert calls == [digest]


def test_a_missing_bake_is_absence_and_never_a_created_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image built from a committed lock bakes nothing. That is ORDINARY.

    It must not read as a failure, and it must not leave an empty CAS behind
    that a later reader would treat as an authoritative empty store.
    """
    absent = tmp_path / "no-such-bake"
    monkeypatch.setenv("BAKED_PROGRAM_CAS_ROOT", str(absent))
    assert baked_program_cas_dir() is None
    assert not absent.exists()

    # worker_store resolves it and simply carries no baked tier.
    store = worker_store(tmp_path / "cas")
    assert store.baked is None


def test_the_baked_path_matches_what_the_builder_writes() -> None:
    """ONE spelling, across two repos, and a drift here is a silent miss.

    tensorhub's `internal/builder/image/derive_bake.go` declares
    `DeriveCASImagePath = "/app/.tensorhub/derive-cas"`. Nothing can import a Go
    constant, so the agreement is pinned here instead of assumed — changing
    either side without the other reintroduces exactly the defect this module
    exists for, and it would report nothing.
    """
    assert BAKED_PROGRAM_CAS_DIR == "/app/.tensorhub/derive-cas"
