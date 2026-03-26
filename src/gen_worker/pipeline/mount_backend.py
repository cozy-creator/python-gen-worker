from __future__ import annotations

import os
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MountBackend:
    mountpoint: str
    fstype: str
    source: str

    @property
    def is_nfs(self) -> bool:
        return (self.fstype or "").lower() in ("nfs", "nfs4")


def _unescape_mountinfo_path(s: str) -> str:
    # /proc/*/mountinfo uses octal escapes for special chars, e.g. "\040" for space.
    out = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 3 < len(s) and s[i + 1 : i + 4].isdigit():
            try:
                out.append(chr(int(s[i + 1 : i + 4], 8)))
                i += 4
                continue
            except Exception:
                pass
        out.append(s[i])
        i += 1
    return "".join(out)


def _parse_mountinfo_line(line: str) -> Optional[MountBackend]:
    line = (line or "").strip()
    if not line:
        return None

    # See `man proc` ("mountinfo").
    # <id> <parent> <major:minor> <root> <mountpoint> <opts> ... - <fstype> <source> <superopts>
    try:
        left, right = line.split(" - ", 1)
    except ValueError:
        return None

    left_parts = left.split()
    right_parts = right.split()
    if len(left_parts) < 6 or len(right_parts) < 2:
        return None

    mountpoint = _unescape_mountinfo_path(left_parts[4])
    fstype = right_parts[0]
    source = _unescape_mountinfo_path(right_parts[1])
    return MountBackend(mountpoint=mountpoint, fstype=fstype, source=source)


def mount_backend_for_path(path: str | Path, *, mountinfo_text: Optional[str] = None) -> Optional[MountBackend]:
    """
    Best-effort map a path to its mount backend by parsing /proc/self/mountinfo.

    - Returns the most specific (longest mountpoint) entry that contains the path.
    - Never raises (returns None on parse/read errors).
    """
    try:
        p = Path(path).resolve()
    except Exception:
        try:
            p = Path(os.fspath(path))
        except Exception:
            return None

    try:
        text = mountinfo_text
        if text is None:
            text = Path("/proc/self/mountinfo").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    best: Optional[MountBackend] = None
    best_len = -1
    for raw in (text or "").splitlines():
        ent = _parse_mountinfo_line(raw)
        if ent is None:
            continue
        try:
            mp = Path(ent.mountpoint).resolve()
        except Exception:
            mp = Path(ent.mountpoint)

        # Exact match or prefix match.
        try:
            if mp == p or mp in p.parents:
                l = len(str(mp))
                if l > best_len:
                    best = ent
                    best_len = l
        except Exception:
            continue

    return best


def is_nfs_path(path: str | Path, *, mountinfo_text: Optional[str] = None) -> bool:
    mb = mount_backend_for_path(path, mountinfo_text=mountinfo_text)
    return bool(mb and mb.is_nfs)


def volume_key_for_path(path: str | Path, *, mountinfo_text: Optional[str] = None) -> Optional[str]:
    """
    Return a stable, non-sensitive identifier for the mount containing `path`.

    We intentionally do NOT return the raw mount source (may contain internal IPs).
    Instead, we hash (fstype, source, mountpoint) and use the hex digest.
    """
    mb = mount_backend_for_path(path, mountinfo_text=mountinfo_text)
    if mb is None:
        return None
    raw = f"{mb.fstype}|{mb.source}|{mb.mountpoint}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()
