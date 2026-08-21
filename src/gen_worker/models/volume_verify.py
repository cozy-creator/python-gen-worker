"""Mandatory, fused, file-parallel snapshot verification."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from gen_worker._vendor.tensorfs import CASRef, DigestMismatch

from . import projection
from .cozy_snapshot import _norm_rel_path

__all__ = [
    "VerifyReport",
    "VerifyTarget",
    "snapshot_verify_targets",
    "split_projection_targets",
    "verify_files",
    "verify_projection",
]

_log = logging.getLogger(__name__)

@dataclass(frozen=True)
class VerifyTarget:
    """One file to verify."""

    path: Path
    ref: str
    size: int = 0
    label: str = ""


@dataclass
class VerifyReport:
    """The verdict AND its denominators."""

    examined: int = 0
    hashed: int = 0
    bytes_hashed: int = 0
    bad: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    expected: int = 0
    projected: int = 0

    @property
    def ok(self) -> bool:
        if self.bad or self.findings:
            return False
        if self.expected and self.examined != self.expected:
            return False
        return self.expected > 0 or self.examined > 0 or self.expected == 0


def _max_workers(n: int) -> int:
    cores = os.cpu_count() or 4
    return max(1, min(n, cores, 8))


def verify_files(
    targets: Sequence[VerifyTarget],
    *,
    parallel: Optional[int] = None,
) -> VerifyReport:
    """Verify every target."""
    rep = VerifyReport(expected=len(targets))
    if not targets:
        return rep

    lock = threading.Lock()

    def _one(t: VerifyTarget) -> None:
        label = t.label or t.ref
        try:
            parsed = CASRef.parse(t.ref)
        except ValueError as exc:
            with lock:
                rep.examined += 1
                rep.bad.append(label)
                rep.findings.append(f"{t.path.name}: unreadable digest {t.ref!r}: {exc}")
            return
        try:
            if not t.path.exists():
                raise DigestMismatch("missing")
            actual = t.path.stat().st_size
            if t.size and actual != t.size:
                raise DigestMismatch(f"size mismatch (declared {t.size}, on disk {actual})")
            digest = hashlib.sha256()
            with t.path.open("rb") as handle:
                while block := handle.read(1 << 20):
                    digest.update(block)
            got = digest.hexdigest()
            if got != parsed.digest:
                raise DigestMismatch(
                    f"sha256 of bytes is {got[:16]}…, manifest says {parsed.digest[:16]}…"
                )
            with lock:
                rep.examined += 1
                rep.hashed += 1
                rep.bytes_hashed += actual
        except (OSError, DigestMismatch, ValueError) as exc:
            _log.warning("volume_verify_fail path=%s: %s", t.path, exc)
            with lock:
                rep.examined += 1
                rep.bad.append(label)
                rep.findings.append(f"{t.path.name}: {exc}")

    workers = parallel or _max_workers(len(targets))
    if workers <= 1:
        for t in targets:
            _one(t)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_one, targets))

    _log.info(
        "volume_verify examined=%d/%d hashed=%d bytes=%d bad=%d",
        rep.examined, rep.expected, rep.hashed, rep.bytes_hashed, len(rep.bad),
    )
    return rep


def split_projection_targets(
    targets: Sequence[VerifyTarget],
) -> Tuple[List[VerifyTarget], List[VerifyTarget]]:
    """Split targets into ``(projection artifacts, files holding real bytes)``."""

    projected: List[VerifyTarget] = []
    material: List[VerifyTarget] = []
    for t in targets:
        if projection.is_projection_artifact(t.path):
            projected.append(t)
        else:
            material.append(t)
    return projected, material


def verify_projection(targets: Sequence[VerifyTarget]) -> VerifyReport:
    """Verify projection artifacts against the manifest, WITHOUT hashing."""

    rep = VerifyReport(expected=len(targets))
    for t in targets:
        label = t.label or t.ref
        rep.examined += 1
        try:
            want = CASRef.parse(t.ref)
        except ValueError as exc:
            rep.bad.append(label)
            rep.findings.append(f"{t.path.name}: unreadable digest {t.ref!r}: {exc}")
            continue
        fault = projection.projection_fault(
            t.path, digest=want.digest, size=t.size
        )
        if fault is None:
            rep.projected += 1
        else:
            rep.bad.append(label)
            rep.findings.append(f"{t.path.name}: {fault}")
    _log.info(
        "projection_verify examined=%d/%d projected=%d bad=%d",
        rep.examined, rep.expected, rep.projected, len(rep.bad),
    )
    return rep


def snapshot_verify_targets(
    files: Sequence[Any], root: Path,
) -> Tuple[List[VerifyTarget], List[str]]:
    """Build the verification target list for a materialized snapshot."""

    targets: List[VerifyTarget] = []
    skipped: List[str] = []
    for f in files:
        path_attr = getattr(f, "path", "") or ""
        try:
            dst = root / _norm_rel_path(path_attr)
        except ValueError:
            skipped.append(path_attr)
            continue
        ref = (getattr(f, "digest", "") or "").strip().lower()
        if not ref:
            skipped.append(path_attr)
            continue
        targets.append(
            VerifyTarget(
                path=dst,
                ref=ref,
                size=int(getattr(f, "size_bytes", 0) or 0),
                label=ref,
            )
        )
    return targets, skipped
