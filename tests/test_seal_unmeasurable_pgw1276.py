"""pgw#1276: an UNMEASURABLE loader map must not score as "unchanged".

`assert_seal_unchanged`'s library half is the only surface that can see a
native library substituted after boot — the seal's own toolchain axis comes
from the DISK manifest, which cannot see an LD_PRELOADed object. The live
comparison reads `/proc/self/maps`. When that map cannot be read, the check
measured nothing; folding that into the passing verdict disarms the tripwire
silently. These tests drive the REAL establish -> assert_seal_unchanged path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import pytest

from gen_worker import env_seal

_LIBS: Tuple[Tuple[str, bytes], ...] = (
    ("libtorch_cuda.so", b"torch-image-bytes"),
    ("libcudart.so.12", b"cudart-image-bytes"),
)


@pytest.fixture(autouse=True)
def _isolate_seal_globals() -> Iterator[None]:
    boot, snapshot = env_seal._BOOT_READBACK, env_seal._LIB_SNAPSHOT
    try:
        yield
    finally:
        env_seal._BOOT_READBACK = boot
        env_seal._LIB_SNAPSHOT = snapshot


def _toolchain_dir(tmp_path: Path, name: str,
                   libs: Tuple[Tuple[str, bytes], ...] = _LIBS) -> Path:
    root = tmp_path / name
    root.mkdir()
    for base, content in libs:
        (root / base).write_bytes(content)
    return root


def _maps(tmp_path: Path, name: str, mapped: Tuple[Path, ...]) -> Path:
    path = tmp_path / name
    lines = [
        f"7f{i:010x}000-7f{i:010x}fff r-xp 00000000 08:01 {i + 1} {lib}"
        for i, lib in enumerate(mapped)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _arm(monkeypatch: pytest.MonkeyPatch, disk: Path, maps: Path) -> None:
    """Point the seal at a controlled env: `disk` is what the env ships,
    `maps` is what the loader mapped. The boot read-back is adopted first
    (the manifest freezes here), so the next call runs the library check."""
    monkeypatch.setattr(env_seal, "_TOOLCHAIN_LIB_DIRS_OVERRIDE", (disk,))
    monkeypatch.setattr(env_seal, "_LIB_SNAPSHOT", None)
    monkeypatch.setattr(env_seal, "_MAPS_PATH", maps)
    monkeypatch.setattr(env_seal, "_BOOT_READBACK", None)
    env_seal.assert_seal_unchanged("adopt")  # first call adopts, checks nothing
    assert dict(env_seal.frozen_library_digests()), "manifest must be armed"


def test_unreadable_loader_map_refuses_instead_of_scoring_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The pgw#1276 red test: the env ships libraries to check and the map
    that says what is loaded cannot be read. Nothing was measured, so the
    verdict cannot be "unchanged"."""
    disk = _toolchain_dir(tmp_path, "disk")
    _arm(monkeypatch, disk, _maps(tmp_path, "maps", tuple(disk.iterdir())))
    monkeypatch.setattr(env_seal, "_MAPS_PATH", tmp_path / "no-such-maps")

    with pytest.raises(env_seal.EnvSealError) as excinfo:
        env_seal.assert_seal_unchanged("mint")
    message = str(excinfo.value)
    assert "no-such-maps" in message
    assert "mint" in message
    assert "2" in message  # the libraries left unchecked are counted


def test_readable_map_matching_the_manifest_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The green control: the same arming, a map that CAN be read, contents
    that match — no refusal. Unmeasurable and unchanged are different."""
    disk = _toolchain_dir(tmp_path, "disk")
    _arm(monkeypatch, disk, _maps(tmp_path, "maps", tuple(disk.iterdir())))
    env_seal.assert_seal_unchanged("mint")


def test_shipped_but_never_mapped_library_is_not_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The deliberate skip stays: a library the loader never mapped cannot
    have been substituted in this process — no fact to measure, as opposed
    to a fact we failed to measure."""
    disk = _toolchain_dir(tmp_path, "disk")
    _arm(monkeypatch, disk, _maps(tmp_path, "maps",
                                  (disk / "libtorch_cuda.so",)))
    env_seal.assert_seal_unchanged("mint")


def test_mapped_library_whose_file_cannot_be_read_is_named_unverified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A mapped library the process cannot stat digests to `<unreadable>` on
    BOTH sides once the manifest hits the same file — equal strings that are
    not a comparison. The refusal says unverified, not "substituted"."""
    disk = _toolchain_dir(tmp_path, "disk")
    _arm(monkeypatch, disk, _maps(tmp_path, "maps", tuple(disk.iterdir())))
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.setattr(env_seal, "_MAPS_PATH", _maps(
        tmp_path, "maps2",
        (disk / "libtorch_cuda.so", gone / "libcudart.so.12")))

    with pytest.raises(env_seal.EnvSealError) as excinfo:
        env_seal.assert_seal_unchanged("mint")
    message = str(excinfo.value)
    assert "libcudart.so.12" in message
    assert "content unverified" in message
    assert "substituted" not in message


def test_substituted_mapped_library_still_refuses_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The arm itself works under this fixture: swap the bytes of a mapped
    library and the same call names it. Without this the unreadable-map test
    could pass against a check that was never armed."""
    disk = _toolchain_dir(tmp_path, "disk")
    _arm(monkeypatch, disk, _maps(tmp_path, "maps", tuple(disk.iterdir())))
    preload = tmp_path / "preload"
    preload.mkdir()
    (preload / "libcudart.so.12").write_bytes(b"SUBSTITUTED-cudart")
    monkeypatch.setattr(env_seal, "_MAPS_PATH", _maps(
        tmp_path, "maps2",
        (disk / "libtorch_cuda.so", preload / "libcudart.so.12")))

    with pytest.raises(env_seal.EnvSealError, match="libcudart.so.12"):
        env_seal.assert_seal_unchanged("mint")
