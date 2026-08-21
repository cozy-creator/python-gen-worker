"""Per-key single-flight for compile work, on LOCAL DISK ONLY."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

LEDGER_DIR = ".work-ledger"

_REMOTE_FSTYPES = frozenset({
    "nfs", "nfs4", "cifs", "smb3", "fuse", "fuse.sshfs", "fuseblk",
    "lustre", "gpfs", "ceph", "9p", "afs", "glusterfs",
})

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class LedgerError(RuntimeError):
    """The ledger could not be used on this filesystem or path."""


class Busy(RuntimeError):
    """Another process holds this key's lease right now."""


def assert_local(root: Path) -> None:
    """Refuse a ledger root whose filesystem does not honor ``flock``."""
    fstype = _fstype(root)
    if fstype is None:
        raise LedgerError(
            f"cannot identify the filesystem under {root}; the compile work "
            f"ledger is local-disk-only (network flock is unproven, e2e#1910) "
            f"and will not run where it cannot prove that."
        )
    if fstype in _REMOTE_FSTYPES:
        raise LedgerError(
            f"{root} is on {fstype}; the compile work ledger is "
            f"local-disk-only. Point --graph-store at local disk."
        )


def _fstype(path: Path) -> Optional[str]:
    try:
        entries = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    target = str(Path(path).resolve())
    best: tuple[int, Optional[str]] = (-1, None)
    for line in entries.splitlines():
        _, _, rest = line.partition(" - ")
        fields = line.split()
        if len(fields) < 5 or not rest:
            continue
        mountpoint = fields[4]
        fstype = rest.split()[0]
        if target == mountpoint or target.startswith(
            mountpoint.rstrip("/") + "/"
        ):
            if len(mountpoint) > best[0]:
                best = (len(mountpoint), fstype)
    return best[1]


def lease_path(root: Path, key: str) -> Path:
    """Where one key's lease file lives."""
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
    hint = _SAFE.sub("-", key)[:48].strip("-")
    return Path(root) / LEDGER_DIR / f"{hint}-{digest}.lease"


@contextmanager
def lease(root: Path, key: str, *, blocking: bool = False) -> Iterator[None]:
    """Hold the single-flight lease for ``key`` for the body's duration."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    assert_local(root)
    path = lease_path(root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(fd, flags)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise Busy(key) from exc
            raise
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n{key}\n".encode("utf-8"))
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


__all__ = ["Busy", "LEDGER_DIR", "LedgerError", "assert_local", "lease", "lease_path"]
