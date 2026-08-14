"""pgw#745: the loaded-lib identity manifest must enumerate USERSPACE
TOOLCHAIN libs only — driver-side objects are NEVER identity. The
host-mounted driver userspace half (libcuda.so.<driver-version>,
libnvidia-*) varies per machine and driver rollout, invisible to the image
digest; sealing it fractures cell keys per driver cohort (live:
libcuda.so.580.126.16 vs .580.159.04 split an L4 fleet — every worker kept
self-minting).

Post-pgw#749 the identity manifest is enumerated from the python env ON
DISK (`toolchain_library_digests`), where the host driver can never appear
at all; the maps-based probe (`loaded_library_digests`) remains the LIVE
integrity surface and must exclude driver objects too."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import pytest

torch = pytest.importorskip("torch")

from gen_worker import env_seal


@pytest.fixture(autouse=True)
def _fresh_snapshot(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(env_seal, "_LIB_SNAPSHOT", None)
    monkeypatch.setattr(env_seal, "_BOOT_READBACK", None)
    yield


def _toolchain_dir(
    tmp_path: Path, name: str,
    libs: Tuple[Tuple[str, bytes], ...] = (
        ("libtorch_cuda.so", b"torch-image-bytes"),
        ("libcudart.so.12", b"cudart-image-bytes"),
    ),
) -> Path:
    root = tmp_path / name
    root.mkdir()
    for base, content in libs:
        (root / base).write_bytes(content)
    return root


def _maps_for(tmp_path: Path, name: str, libs: Tuple[Tuple[str, bytes], ...]) -> Path:
    """A fake /proc/self/maps + backing files for the LIVE mapped set."""
    root = tmp_path / name
    root.mkdir()
    lines = []
    for i, (base, content) in enumerate(libs):
        path = root / base
        path.write_bytes(content)
        lines.append(
            f"7f{i:010x}000-7f{i:010x}fff r-xp 00000000 08:01 {i + 1} {path}")
    maps = root / "maps"
    maps.write_text("\n".join(lines) + "\n")
    return maps


def _seal_with_dirs(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict:
    monkeypatch.setattr(env_seal, "_TOOLCHAIN_LIB_DIRS_OVERRIDE", (root,))
    monkeypatch.setattr(env_seal, "_LIB_SNAPSHOT", None)
    return env_seal.effective_seal()


def test_seal_identical_across_driver_cohorts_when_userspace_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The pgw#745 live shape: same image (same toolchain content on disk),
    two RunPod machines with different host drivers — the seal (and hence
    the cell key's env_seal axis) must be IDENTICAL. Driver identity stays
    a recorded-only metadata axis, and even a driver lib FILE
    smuggled into a toolchain dir is excluded by name."""
    driver_a = (("libcuda.so.580.126.16", b"driver-cohort-a"),)
    driver_b = (("libcuda.so.580.159.04", b"driver-cohort-b"),)
    base = (
        ("libtorch_cuda.so", b"torch-image-bytes"),
        ("libcudart.so.12", b"cudart-image-bytes"),
    )
    seal_a = _seal_with_dirs(
        monkeypatch, _toolchain_dir(tmp_path, "a", base + driver_a))
    seal_b = _seal_with_dirs(
        monkeypatch, _toolchain_dir(tmp_path, "b", base + driver_b))
    assert seal_a["loaded_libs"] == seal_b["loaded_libs"]
    assert env_seal.seal_digest(seal_a) == env_seal.seal_digest(seal_b)


def test_driver_libs_never_enter_either_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Sweep (pgw#745 task 3): every host-driver surface is excluded by
    name from BOTH the disk identity manifest and the live mapped probe;
    the image-shipped toolchain set is still enumerated."""
    driver_libs: Tuple[Tuple[str, bytes], ...] = (
        ("libcuda.so.580.126.16", b"driver"),
        ("libnvidia-ml.so.1", b"host-nvml"),
        ("libnvidia-ptxjitcompiler.so.580.126.16", b"host-ptxjit"),
        ("libcudadebugger.so.1", b"host-debugger"),
    )
    userspace: Tuple[Tuple[str, bytes], ...] = (
        ("libtorch_cuda.so", b"torch-image-bytes"),
        ("libcudart.so.12", b"cudart-image-bytes"),
    )
    root = _toolchain_dir(tmp_path, "disk", userspace + driver_libs)
    monkeypatch.setattr(env_seal, "_TOOLCHAIN_LIB_DIRS_OVERRIDE", (root,))
    disk = dict(env_seal.toolchain_library_digests())
    maps = _maps_for(tmp_path, "maps", userspace + driver_libs)
    monkeypatch.setattr(env_seal, "_MAPS_PATH", maps)
    live = dict(env_seal.loaded_library_digests())
    for libs in (disk, live):
        assert "libtorch_cuda.so" in libs
        assert "libcudart.so.12" in libs
        assert not any(base.startswith("libcuda.so") for base in libs), libs
        assert not any(base.startswith("libnvidia-") for base in libs), libs
        assert not any(
            base.startswith("libcudadebugger") for base in libs), libs


def test_userspace_substitution_still_rekeys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The pgw#719 hole stays closed: substituting an IMAGE lib's CONTENT
    (libcudart here) must still change the seal."""
    seal_a = _seal_with_dirs(monkeypatch, _toolchain_dir(tmp_path, "a"))
    seal_b = _seal_with_dirs(monkeypatch, _toolchain_dir(tmp_path, "b", (
        ("libtorch_cuda.so", b"torch-image-bytes"),
        ("libcudart.so.12", b"SUBSTITUTED-cudart"),
    )))
    assert seal_a["loaded_libs"] != seal_b["loaded_libs"]
    assert env_seal.seal_digest(seal_a) != env_seal.seal_digest(seal_b)
