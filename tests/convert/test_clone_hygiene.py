"""gw#462 clone scratch hygiene + disk preflight (real filesystem, real flocks).

The J24 qwen postmortem: a 20GB conversion pod ENOSPC-died mid-download with
no preflight, and failed-clone workdirs accumulated forever.
"""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

import pytest

from gen_worker.convert.clone import (
    CloneDiskSpaceError,
    _clone_workdir,
    _preflight_disk,
    _sweep_stale_workdirs,
    run_clone,
)


class _Plan:
    def __init__(self, sizes: list[int]) -> None:
        self._sizes = sizes

    def bank_files(self):
        return [(f"f{i}", s, f"cid{i}") for i, s in enumerate(self._sizes)]


# ---------------------------------------------------------------------------
# Disk preflight
# ---------------------------------------------------------------------------

def test_preflight_rejects_oversized_source_with_actionable_message(tmp_path: Path) -> None:
    # 10 PiB source cannot fit any real test filesystem.
    with pytest.raises(CloneDiskSpaceError, match=r"need ~.* GiB free .*have .* GiB"):
        _preflight_disk(tmp_path, _Plan([10 * 1024**5]))


def test_preflight_passes_tiny_source(tmp_path: Path) -> None:
    _preflight_disk(tmp_path, _Plan([1024]))  # must not raise


def test_preflight_skips_when_plan_unavailable(tmp_path: Path) -> None:
    _preflight_disk(tmp_path, None)  # fail-open: download surfaces its own error


def test_preflight_headroom_is_tunable(tmp_path: Path, monkeypatch) -> None:
    free = os.statvfs(tmp_path).f_bavail * os.statvfs(tmp_path).f_frsize
    # A source about half the free space passes at 1x headroom but must fail
    # at an absurd multiplier.
    source = max(free // 2, 1)
    monkeypatch.setenv("COZY_CONVERT_DISK_HEADROOM", "1000000")
    with pytest.raises(CloneDiskSpaceError):
        _preflight_disk(tmp_path, _Plan([source]))
    monkeypatch.setenv("COZY_CONVERT_DISK_HEADROOM", "0.000001")
    _preflight_disk(tmp_path, _Plan([source]))


# ---------------------------------------------------------------------------
# Stale-scratch sweep
# ---------------------------------------------------------------------------

def _mkdir_aged(base: Path, name: str, age_s: float) -> Path:
    d = base / name
    d.mkdir(parents=True)
    (d / "junk.bin").write_bytes(b"x" * 8)
    stamp = time.time() - age_s
    os.utime(d, (stamp, stamp))
    return d


def test_sweep_removes_stale_unlocked_keeps_live_and_fresh(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COZY_CONVERT_SCRATCH_TTL_S", "60")
    stale = _mkdir_aged(tmp_path, "clone-deadbeef00000001", age_s=3600)
    fresh = _mkdir_aged(tmp_path, "clone-deadbeef00000002", age_s=1)
    live = _mkdir_aged(tmp_path, "clone-deadbeef00000003", age_s=3600)
    mine = _mkdir_aged(tmp_path, "clone-deadbeef00000004", age_s=3600)

    # A concurrent clone holds `live`'s flock (real lock, same protocol).
    live_lock = os.open(tmp_path / f".{live.name}.lock", os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(live_lock, fcntl.LOCK_EX)
    try:
        _sweep_stale_workdirs(tmp_path, keep=mine)
    finally:
        os.close(live_lock)

    assert not stale.exists(), "stale unlocked scratch must be swept"
    assert fresh.exists(), "fresh scratch is inside the TTL"
    assert live.exists(), "flock-held scratch belongs to a live clone"
    assert mine.exists(), "the caller's own workdir is never swept"


def test_sweep_survives_missing_base(tmp_path: Path) -> None:
    _sweep_stale_workdirs(tmp_path / "does-not-exist")  # must not raise


# ---------------------------------------------------------------------------
# Workdir cleanup after EVERY job (success AND failure)
# ---------------------------------------------------------------------------

class _Ctx:
    _file_api_base_url = "http://127.0.0.1:1"
    _worker_capability_token = "tok"
    owner = "acme"


def test_failed_clone_removes_workdir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path))
    monkeypatch.delenv("COZY_CONVERT_RETAIN_WORKDIR", raising=False)
    with pytest.raises(ValueError, match="unsupported clone provider"):
        run_clone(_Ctx(), provider="bogus", destination_repo="acme/x")
    workdir = _clone_workdir("bogus", "", "acme/x")  # re-derives the keyed path
    # _clone_workdir recreates the dir; the failed run must have removed its
    # contents-bearing predecessor, so the fresh dir is empty.
    assert list(workdir.iterdir()) == []
    workdir.rmdir()
    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == []


def test_failed_clone_retains_workdir_when_opted_in(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path))
    monkeypatch.setenv("COZY_CONVERT_RETAIN_WORKDIR", "1")
    with pytest.raises(ValueError, match="unsupported clone provider"):
        run_clone(_Ctx(), provider="bogus", destination_repo="acme/x")
    dirs = [p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("clone-")]
    assert len(dirs) == 1, "opt-in retention must keep the failed workdir"
