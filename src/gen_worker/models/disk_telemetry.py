"""Measured disk telemetry (pgw#610 / th#962).

statvfs on the REAL mount points the worker uses (CAS root on the container
disk; the attached endpoint volume when mounted; a shared NFS mount when one
exists), plus per-tier "safely reclaimable" bytes: ref-index entries that are
inactive AND not in the current desired set — exactly the disk-GC LRU's
eligible set, i.e. evictions the hub may cause without touching live work.

Everything here is O(mounts + refs) over in-memory state and a couple of
statvfs/stat syscalls — never a tree rescan.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import msgspec

# Mirror proto StorageTier values (worker_scheduler.proto).
TIER_CONTAINER = 1
TIER_VOLUME = 2
TIER_NFS = 3

# Free/used are floored to this quantum so statvfs jitter does not churn the
# edge-triggered StateDelta or the capacity generation.
DISK_QUANTUM_BYTES = 64 * 1024 * 1024


class MountSpec(msgspec.Struct, frozen=True, kw_only=True):
    tier: int
    path: str


class TierUsage(msgspec.Struct, frozen=True, kw_only=True):
    tier: int
    mount_path: str
    total_bytes: int
    free_bytes: int
    used_bytes: int
    reclaimable_bytes: int


def _statvfs_totals(path: str) -> Optional[Tuple[int, int]]:
    """(total, free) bytes for the filesystem holding ``path``; walks up to
    the nearest existing parent (the cache dir may not exist yet)."""
    p = Path(path)
    for candidate in (p, *p.parents):
        try:
            st = os.statvfs(candidate)
        except OSError:
            continue
        frsize = int(st.f_frsize or st.f_bsize or 0)
        if frsize <= 0:
            return None
        return int(st.f_blocks) * frsize, int(st.f_bavail) * frsize
    return None


def _device_of(path: str) -> Optional[int]:
    p = Path(path)
    for candidate in (p, *p.parents):
        try:
            return int(os.stat(candidate).st_dev)
        except OSError:
            continue
    return None


def measure_tiers(
    mounts: Sequence[MountSpec],
    reclaimable_entries: Sequence[Tuple[str, int]],
) -> List[TierUsage]:
    """Measure each mount and attribute reclaimable (path, bytes) entries to
    the mount whose device holds them (first mount wins ties/unknowns)."""
    devices: List[Optional[int]] = [_device_of(m.path) for m in mounts]
    reclaimable = [0] * len(mounts)
    for entry_path, entry_bytes in reclaimable_entries:
        if entry_bytes <= 0 or not mounts:
            continue
        dev = _device_of(entry_path)
        target = 0
        if dev is not None:
            for i, mount_dev in enumerate(devices):
                if mount_dev == dev:
                    target = i
                    break
        reclaimable[target] += int(entry_bytes)

    out: List[TierUsage] = []
    for i, mount in enumerate(mounts):
        totals = _statvfs_totals(mount.path)
        if totals is None:
            continue
        total, free = totals
        free_q = (free // DISK_QUANTUM_BYTES) * DISK_QUANTUM_BYTES
        out.append(TierUsage(
            tier=mount.tier,
            mount_path=mount.path,
            total_bytes=total,
            free_bytes=free_q,
            used_bytes=max(0, total - free_q),
            reclaimable_bytes=reclaimable[i],
        ))
    return out


__all__ = [
    "DISK_QUANTUM_BYTES",
    "MountSpec",
    "TierUsage",
    "TIER_CONTAINER",
    "TIER_NFS",
    "TIER_VOLUME",
    "measure_tiers",
]
