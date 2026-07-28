"""pgw#751: adopt-on-arm over a WARM local cache must succeed when the
delivered cell carries same-cache-key members whose bytes differ from the
locally-compiled ones.

Triton/inductor cache paths are cache-KEY-addressed; the bytes are NOT the
identity (pgw#699/#711 respec: three mints of one cell key are never
byte-identical). Live evidence: 7 of 13 adoption attempts in the 0.2.14
burst failed ``adopt_failed:cache_collision`` because the pod had already
compiled something before delivery — any warm pod could never install a
cell. Semantics: LOCAL WINS (the local member may be mmapped/served by
this process; torch's consumption is keyed, so serving is identical) and
the merge stays additive. A structural conflict (directory where a file is
expected) still refuses typed.

Red-verified: on the pre-fix tree the byte-divergent adopt raises
``AdoptError(cache_collision)`` and both tests here fail."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gen_worker import compile_cache as cc


def _tree(root: Path, files: dict[str, bytes]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    root = root.resolve()
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_adopt_over_warm_cache_with_divergent_same_key_member(
    tmp_path: Path,
) -> None:
    """The pgw#751 red test: byte-divergent same-key member -> adoption
    SUCCEEDS deterministically, local bytes win, new members merge in."""
    cache_dir = tmp_path / "cache"
    live = _tree(cache_dir / "compile-cache", {
        "triton/2D45K2G4/kernel.json": b"LOCAL-bytes",
        "inductor/aa/entry.py": b"local-entry",
    })
    delivered = _tree(tmp_path / "delivered", {
        "triton/2D45K2G4/kernel.json": b"PRODUCER-bytes",  # same key, new bytes
        "inductor/aa/entry.py": b"local-entry",            # identical member
        "inductor/bb/new.py": b"new-member",               # additive member
    })
    meta = cc.artifact_metadata(
        family="sd15", shapes=[(768, 768)], targets=["transformer"])
    artifact = cc.pack(delivered, tmp_path / "cell.tar.gz", meta)

    cc.seed_artifact(artifact, "sd15", cache_dir)  # must NOT raise

    after = _snapshot(live)
    assert after["triton/2D45K2G4/kernel.json"] == b"LOCAL-bytes"  # local wins
    assert after["inductor/aa/entry.py"] == b"local-entry"
    assert after["inductor/bb/new.py"] == b"new-member"  # still additive


def test_adopt_is_idempotent_over_the_divergent_member(tmp_path: Path) -> None:
    """Replaying the same delivery converges: local bytes stay, nothing
    flip-flops."""
    cache_dir = tmp_path / "cache"
    live = _tree(cache_dir / "compile-cache", {
        "triton/k/member.json": b"LOCAL",
    })
    delivered = _tree(tmp_path / "delivered", {
        "triton/k/member.json": b"REMOTE",
        "inductor/x/f.py": b"x",
    })
    meta = cc.artifact_metadata(
        family="sd15", shapes=[(768, 768)], targets=["transformer"])
    artifact = cc.pack(delivered, tmp_path / "cell.tar.gz", meta)
    cc.seed_artifact(artifact, "sd15", cache_dir)
    first = _snapshot(live)
    cc.seed_artifact(artifact, "sd15", cache_dir)
    assert _snapshot(live) == first
    assert first["triton/k/member.json"] == b"LOCAL"


def test_structural_conflict_still_refuses_typed(tmp_path: Path) -> None:
    """A directory squatting a delivered file path is a real conflict —
    keep the typed refusal."""
    cache_dir = tmp_path / "cache"
    live = cache_dir / "compile-cache"
    (live / "triton" / "k" / "member.json").mkdir(parents=True)  # dir, not file
    delivered = _tree(tmp_path / "delivered", {
        "triton/k/member.json": b"REMOTE",
    })
    meta = cc.artifact_metadata(
        family="sd15", shapes=[(768, 768)], targets=["transformer"])
    artifact = cc.pack(delivered, tmp_path / "cell.tar.gz", meta)
    with pytest.raises(cc.AdoptError) as exc:
        cc.seed_artifact(artifact, "sd15", cache_dir)
    assert exc.value.reason == "cache_collision"
    assert "structural" in str(exc.value)
