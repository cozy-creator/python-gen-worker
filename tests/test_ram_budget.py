"""Cgroup-aware host-RAM budget derivation (th#721) — CPU-only, deterministic.

RunPod GPU pods land on lottery-RAM hosts (31GB vs 62GB for the same GPU) and
the container is cgroup-limited below /proc/meminfo; a probe that trusts
psutil alone over-reports and the kernel SIGKILLs at the cgroup ceiling.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gen_worker.models import memory as memory_mod
from gen_worker.models.memory import (
    HostRam,
    cgroup_memory_current_bytes,
    cgroup_memory_limit_bytes,
    log_ram_budget_once,
    probe_host_ram,
)

_GiB = 1024 ** 3


def _fake_cgroup(tmp_path: Path, *, rel: str = "", files: dict[str, str]) -> tuple[Path, Path]:
    """Build a fake cgroup tree + /proc/self/cgroup pointing at ``rel``."""
    root = tmp_path / "cgroup"
    root.mkdir()
    for name, content in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text(f"0::/{rel}\n" if rel else "0::/\n")
    return root, proc


def test_v2_limit_at_self_cgroup(tmp_path: Path) -> None:
    root, proc = _fake_cgroup(
        tmp_path, rel="kubepods/pod1",
        files={"kubepods/pod1/memory.max": str(31 * _GiB)},
    )
    assert cgroup_memory_limit_bytes(root, proc) == 31 * _GiB


def test_v2_tightest_ancestor_limit_wins(tmp_path: Path) -> None:
    root, proc = _fake_cgroup(
        tmp_path, rel="kubepods/pod1",
        files={
            "kubepods/memory.max": str(16 * _GiB),
            "kubepods/pod1/memory.max": str(31 * _GiB),
        },
    )
    assert cgroup_memory_limit_bytes(root, proc) == 16 * _GiB


def test_v2_max_means_uncapped(tmp_path: Path) -> None:
    root, proc = _fake_cgroup(tmp_path, files={"memory.max": "max"})
    assert cgroup_memory_limit_bytes(root, proc) is None


def test_v1_fallback_limit(tmp_path: Path) -> None:
    root, proc = _fake_cgroup(
        tmp_path, files={"memory/memory.limit_in_bytes": str(31 * _GiB)},
    )
    assert cgroup_memory_limit_bytes(root, proc) == 31 * _GiB


def test_v1_sentinel_means_uncapped(tmp_path: Path) -> None:
    root, proc = _fake_cgroup(
        tmp_path, files={"memory/memory.limit_in_bytes": str(0x7FFFFFFFFFFFF000)},
    )
    assert cgroup_memory_limit_bytes(root, proc) is None


def test_missing_cgroup_files_mean_uncapped(tmp_path: Path) -> None:
    root, proc = _fake_cgroup(tmp_path, files={})
    assert cgroup_memory_limit_bytes(root, proc) is None
    assert cgroup_memory_current_bytes(root, proc) is None


def test_current_reads_deepest_counter(tmp_path: Path) -> None:
    root, proc = _fake_cgroup(
        tmp_path, rel="pod1",
        files={
            "memory.current": str(50 * _GiB),
            "pod1/memory.current": str(20 * _GiB),
        },
    )
    assert cgroup_memory_current_bytes(root, proc) == 20 * _GiB


def test_probe_caps_meminfo_at_cgroup_limit(tmp_path: Path) -> None:
    """RunPod shape: host meminfo says lots, cgroup caps the container."""
    root, proc = _fake_cgroup(
        tmp_path,
        files={"memory.max": str(4 * _GiB), "memory.current": str(1 * _GiB)},
    )
    ram = probe_host_ram(root=root, proc_self_cgroup=proc)
    assert isinstance(ram, HostRam)
    assert ram.source == "cgroup"
    assert ram.cgroup_limit_gb == pytest.approx(4.0)
    assert ram.total_gb == pytest.approx(4.0)  # real meminfo total >> 4GiB
    assert ram.available_gb == pytest.approx(3.0)  # limit - current
    assert ram.meminfo_total_gb > 4.0


def test_probe_counts_inactive_file_as_reclaimable_headroom(tmp_path: Path) -> None:
    """#543: model reads may fill a cgroup's inactive file page cache.

    ``memory.current`` includes that cache, but the kernel can reclaim pages on
    the inactive file LRU for the next tensor allocation.  Admission therefore
    uses cgroup working set, not raw usage.
    """
    root, proc = _fake_cgroup(
        tmp_path,
        files={
            "memory.max": str(4 * _GiB),
            "memory.current": str(3 * _GiB),
            "memory.stat": f"anon {_GiB}\ninactive_file {2 * _GiB}\n",
        },
    )
    ram = probe_host_ram(root=root, proc_self_cgroup=proc)
    assert ram.available_gb == pytest.approx(3.0)


def test_probe_keeps_active_file_in_cgroup_working_set(tmp_path: Path) -> None:
    """Only the kernel's inactive file list is treated as ready headroom."""
    root, proc = _fake_cgroup(
        tmp_path,
        files={
            "memory.max": str(4 * _GiB),
            "memory.current": str(3 * _GiB),
            "memory.stat": (
                f"inactive_file {_GiB}\nactive_file {_GiB}\n"
            ),
        },
    )
    ram = probe_host_ram(root=root, proc_self_cgroup=proc)
    assert ram.available_gb == pytest.approx(2.0)


def test_probe_uses_v1_hierarchical_inactive_file(tmp_path: Path) -> None:
    root, proc = _fake_cgroup(
        tmp_path,
        files={
            "memory/memory.limit_in_bytes": str(4 * _GiB),
            "memory/memory.usage_in_bytes": str(3 * _GiB),
            "memory/memory.stat": f"total_inactive_file {2 * _GiB}\n",
        },
    )
    ram = probe_host_ram(root=root, proc_self_cgroup=proc)
    assert ram.available_gb == pytest.approx(3.0)


def test_probe_without_cgroup_is_meminfo_passthrough(tmp_path: Path) -> None:
    root, proc = _fake_cgroup(tmp_path, files={})
    ram = probe_host_ram(root=root, proc_self_cgroup=proc)
    assert ram.source == "meminfo"
    assert ram.cgroup_limit_gb is None
    assert ram.total_gb == ram.meminfo_total_gb > 0
    assert ram.available_gb == ram.meminfo_available_gb > 0


def test_log_ram_budget_once_is_once(monkeypatch, caplog) -> None:
    monkeypatch.setattr(memory_mod, "_ram_budget_logged", False)
    with caplog.at_level(logging.INFO, logger="gen_worker.models.memory"):
        log_ram_budget_once(floor_gb=8.0)
        log_ram_budget_once(floor_gb=8.0)
    lines = [r.message for r in caplog.records if "RAM_BUDGET=" in r.message]
    assert len(lines) == 1
    assert "source=" in lines[0]
    assert "floor_gb=8.0" in lines[0]


def test_log_names_cgroup_constraint(monkeypatch, caplog) -> None:
    monkeypatch.setattr(memory_mod, "_ram_budget_logged", False)
    monkeypatch.setattr(memory_mod, "probe_host_ram", lambda **_: HostRam(
        total_gb=31.0, available_gb=24.0,
        meminfo_total_gb=62.0, meminfo_available_gb=50.0,
        cgroup_limit_gb=31.0, source="cgroup",
    ))
    with caplog.at_level(logging.INFO, logger="gen_worker.models.memory"):
        log_ram_budget_once(floor_gb=6.2)
    [rec] = [r for r in caplog.records if "RAM_BUDGET=" in r.message]
    assert rec.levelno == logging.WARNING
    assert "RAM_BUDGET=24.8GiB" in rec.message
    assert "source=cgroup" in rec.message
    assert "cgroup_limit_gb=31.0" in rec.message
    assert "meminfo_total_gb=62.0" in rec.message
    assert "spill to disk" in rec.message
