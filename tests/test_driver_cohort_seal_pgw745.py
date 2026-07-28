"""pgw#745: the loaded-lib manifest must enumerate USERSPACE TOOLCHAIN libs
only — driver-side objects are NEVER identity (gw#577). The host-mounted
driver userspace half (libcuda.so.<driver-version>, libnvidia-*) varies per
machine and driver rollout, invisible to the image digest; sealing it
fractures cell keys per driver cohort (live: libcuda.so.580.126.16 vs
.580.159.04 split an L4 fleet — every worker kept self-minting).

Red-verified: on the pre-fix tree the two cohort seals differ (libcuda in
the manifest) and the identity tests here fail."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import pytest

torch = pytest.importorskip("torch")

from gen_worker import env_seal


@pytest.fixture(autouse=True)
def _fresh_snapshot(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(env_seal, "_LIB_SNAPSHOT", None)
    monkeypatch.setattr(env_seal, "_BOOT_SEAL", None)
    yield


def _cohort(
    tmp_path: Path, name: str, driver: str, driver_bytes: bytes,
    cudart_bytes: bytes = b"cudart-image-bytes",
    extra: Tuple[Tuple[str, bytes], ...] = (),
) -> Path:
    """A fake /proc/self/maps + backing lib files for one machine cohort:
    identical image-shipped userspace libs, one host-mounted driver."""
    root = tmp_path / name
    root.mkdir()
    libs = [
        ("libtorch_cuda.so", b"torch-image-bytes"),
        ("libcudart.so.12", cudart_bytes),
        (driver, driver_bytes),
        *extra,
    ]
    lines = []
    for i, (base, content) in enumerate(libs):
        path = root / base
        path.write_bytes(content)
        lines.append(
            f"7f{i:010x}000-7f{i:010x}fff r-xp 00000000 08:01 {i + 1} {path}")
    maps = root / "maps"
    maps.write_text("\n".join(lines) + "\n")
    return maps


def _seal_for(monkeypatch: pytest.MonkeyPatch, maps: Path) -> dict:
    monkeypatch.setattr(env_seal, "_MAPS_PATH", maps)
    monkeypatch.setattr(env_seal, "_LIB_SNAPSHOT", None)
    return env_seal.effective_seal()


def test_seal_identical_across_driver_cohorts_when_userspace_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The pgw#745 live shape: same image, two RunPod machines, different
    host driver — the seal (and hence the cell key's env_seal axis) must be
    IDENTICAL. The driver stays a recorded-only axis (gw#577)."""
    seal_a = _seal_for(monkeypatch, _cohort(
        tmp_path, "a", "libcuda.so.580.126.16", b"driver-cohort-a"))
    seal_b = _seal_for(monkeypatch, _cohort(
        tmp_path, "b", "libcuda.so.580.159.04", b"driver-cohort-b"))
    assert seal_a["loaded_libs"] == seal_b["loaded_libs"]
    assert env_seal.seal_digest(seal_a) == env_seal.seal_digest(seal_b)


def test_driver_libs_never_enter_the_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Sweep (pgw#745 task 3): every host-driver surface is excluded by
    name; the image-shipped toolchain set is still enumerated."""
    maps = _cohort(
        tmp_path, "sweep", "libcuda.so.580.126.16", b"driver", extra=(
            ("libnvidia-ml.so.1", b"host-nvml"),
            ("libnvidia-ptxjitcompiler.so.580.126.16", b"host-ptxjit"),
            ("libcudadebugger.so.1", b"host-debugger"),
        ))
    monkeypatch.setattr(env_seal, "_MAPS_PATH", maps)
    libs = dict(env_seal.loaded_library_digests())
    assert "libtorch_cuda.so" in libs
    assert "libcudart.so.12" in libs
    assert not any(base.startswith("libcuda.so") for base in libs), libs
    assert not any(base.startswith("libnvidia-") for base in libs), libs
    assert not any(base.startswith("libcudadebugger") for base in libs), libs


def test_userspace_substitution_still_rekeys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The pgw#719 hole stays closed: substituting an IMAGE lib (libcudart
    here — the LD_PRELOAD class) must still change the seal."""
    seal_a = _seal_for(monkeypatch, _cohort(
        tmp_path, "a", "libcuda.so.580.126.16", b"driver-a"))
    seal_b = _seal_for(monkeypatch, _cohort(
        tmp_path, "b", "libcuda.so.580.126.16", b"driver-a",
        cudart_bytes=b"SUBSTITUTED-cudart"))
    assert seal_a["loaded_libs"] != seal_b["loaded_libs"]
    assert env_seal.seal_digest(seal_a) != env_seal.seal_digest(seal_b)
