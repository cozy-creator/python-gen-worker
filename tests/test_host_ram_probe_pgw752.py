from __future__ import annotations

from pathlib import Path

from gen_worker.models.memory import probe_host_ram

_GIB = 1024 ** 3


def _cgroup(tmp_path: Path, *, limit: int, current: int, **stat: int) -> tuple[Path, Path]:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "memory.max").write_text(str(limit))
    (root / "memory.current").write_text(str(current))
    (root / "memory.stat").write_text(
        "".join(f"{k} {v}\n" for k, v in stat.items()))
    proc = tmp_path / "self_cgroup"
    proc.write_text("0::/\n")
    return root, proc


def test_active_snapshot_cache_is_available_to_the_next_load(tmp_path) -> None:
    """The wan-2.2 pod's shape: 251GiB cap, ~10GiB anonymous, the rest clean snapshot cache on the ACTIVE file LRU because it was just written."""
    root, proc = _cgroup(
        tmp_path,
        limit=251 * _GIB,
        current=190 * _GIB,
        anon=10 * _GIB,
        file=180 * _GIB,
        active_file=178 * _GIB,
        inactive_file=2 * _GIB,
        shmem=0,
        file_dirty=0,
        file_writeback=0,
    )
    ram = probe_host_ram(root=root, proc_self_cgroup=proc)

    assert ram.reclaimable_file_gb == 180.0
    assert ram.cgroup_limit_gb == 251.0
    cgroup_available = ram.cgroup_limit_gb - 10.0
    assert cgroup_available >= 64.3 + 8.0
    assert ram.available_gb <= cgroup_available


def test_unreclaimable_pages_stay_charged(tmp_path) -> None:
    """Anonymous memory, tmpfs and not-yet-written-back pages are NOT room — crediting them would trade a false refusal for an OOM kill."""
    root, proc = _cgroup(
        tmp_path,
        limit=100 * _GIB,
        current=90 * _GIB,
        anon=40 * _GIB,
        file=50 * _GIB,
        active_file=30 * _GIB,
        inactive_file=20 * _GIB,
        shmem=10 * _GIB,
        file_dirty=5 * _GIB,
        file_writeback=1 * _GIB,
    )
    ram = probe_host_ram(root=root, proc_self_cgroup=proc)

    assert ram.reclaimable_file_gb == 34.0
    assert ram.cgroup_limit_gb == 100.0
    assert ram.cgroup_limit_gb - (90.0 - ram.reclaimable_file_gb) == 44.0


def test_cgroup_v1_totals_are_read(tmp_path) -> None:
    """v1 hosts report the same facts under total_* keys."""
    root = tmp_path / "cgroup"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "memory.limit_in_bytes").write_text(str(64 * _GIB))
    (root / "memory" / "memory.usage_in_bytes").write_text(str(40 * _GIB))
    (root / "memory" / "memory.stat").write_text(
        f"total_active_file {20 * _GIB}\n"
        f"total_inactive_file {5 * _GIB}\n"
        f"total_shmem {1 * _GIB}\n"
    )
    proc = tmp_path / "self_cgroup"
    proc.write_text("")

    ram = probe_host_ram(root=root, proc_self_cgroup=proc)
    assert ram.reclaimable_file_gb == 24.0
