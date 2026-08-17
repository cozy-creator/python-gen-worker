"""Mandatory, fused, file-parallel snapshot verification.

This is a LOAD-BEARING SECURITY CONTROL, not an optimization. A materialized
snapshot can come from a volume shared across releases and pods, so "the bytes
on this volume are the bytes the manifest names" is the only thing standing
between one release and another's weights. It is therefore:

*   **MANDATORY** — never sampled, never opt-in, never skipped because the tree
    "looks fine". A file the manifest covers is hashed.
*   **PREFIX-DISPATCHED** — the algorithm comes from the digest, never from the
    call site. Under manifest v2 ``f.blake3`` is EMPTY and the digest lives in
    ``f.digest`` as ``sha256:<hex>``, so a call site that picks the algorithm
    itself verifies nothing while still reporting a clean tree — the same
    false-clean shape as reading ``manifest["files"]`` when the key is
    ``entries``.
*   **FAIL-CLOSED** — a mismatch names the blob to quarantine so
    re-materialization re-downloads instead of re-linking the same bad bytes.
    Upstream reports stay untrusted hints.
*   **FILE-PARALLEL** — hashing is CPU-bound and releases the GIL inside
    ``hashlib``, so N files hash on N cores. At ~2 GB/s per core, 8
    cores put a 40 GiB tree at a few seconds.
*   **COPY-SPECIFIC** — every materialized file is hashed. A digest identifies
    expected content, not the mutable path being checked, so one good copy can
    never authorize another same-sized copy.

Every report carries the DENOMINATOR it examined. A verdict computed from zero
files is a broken reader, not a clean tree, and ``ok`` is false in that case
whenever files were expected.
"""

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
    """One file to verify. ``ref`` is ALGORITHM-TAGGED — an untagged ref is an
    unreadable digest, i.e. CORRUPT, not "nothing to check". ``size``
    of 0 means the manifest declared none."""

    path: Path
    ref: str
    size: int = 0
    # label is what the caller wants back in `bad` — normally the digest, so a
    # quarantine can delete the blob rather than just the materialized copy.
    label: str = ""


@dataclass
class VerifyReport:
    """The verdict AND its denominators."""

    examined: int = 0
    hashed: int = 0
    bytes_hashed: int = 0
    bad: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    # expected is what the caller believed it was checking. When it disagrees
    # with `examined`, the reader is wrong and the verdict is not trustworthy.
    expected: int = 0
    # Projection artifacts checked structurally rather than by hashing. They
    # count as REAL work for the vacuity guard: a projected tree legitimately
    # hashes nothing, and scoring that as "read nothing at all" is what turned
    # boot verification into an infinite re-download (pgw#1308).
    projected: int = 0

    @property
    def ok(self) -> bool:
        if self.bad or self.findings:
            return False
        if self.expected and self.examined != self.expected:
            return False
        # A pass over zero files is only meaningful if zero were expected.
        return self.expected > 0 or self.examined > 0 or self.expected == 0


def _max_workers(n: int) -> int:
    cores = os.cpu_count() or 4
    # Hashing is memory-bandwidth bound well before it is core bound, and this
    # runs inside a pod's cgroup alongside the model, so do not take the box.
    return max(1, min(n, cores, 8))


def verify_files(
    targets: Sequence[VerifyTarget],
    *,
    parallel: Optional[int] = None,
) -> VerifyReport:
    """Verify every target. Hashes in parallel; never samples."""
    rep = VerifyReport(expected=len(targets))
    if not targets:
        return rep

    lock = threading.Lock()

    def _one(t: VerifyTarget) -> None:
        label = t.label or t.ref
        try:
            parsed = CASRef.parse(t.ref)
        except ValueError as exc:
            # An unreadable digest is CORRUPT, not "nothing to check". Treating
            # it as a mere finding would let a malformed manifest entry pass
            # the tree — the same degrade-to-skip that made the v2 snapshot
            # report clean without being hashed.
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
    """Split targets into ``(projection artifacts, files holding real bytes)``.

    A projection artifact is a TFSSTUB1 pointer stub or a symlink into the CAS
    ``objects/``. Neither holds the bytes its manifest entry names, so hashing
    it at its path is not a weaker check — it is a check of the wrong thing,
    and it fails 100% of the time.
    """

    projected: List[VerifyTarget] = []
    material: List[VerifyTarget] = []
    for t in targets:
        if projection.is_projection_artifact(t.path):
            projected.append(t)
        else:
            material.append(t)
    return projected, material


def verify_projection(targets: Sequence[VerifyTarget]) -> VerifyReport:
    """Verify projection artifacts against the manifest, WITHOUT hashing.

    This is deliberately not a weakening of the mandatory-hash rule above; it
    is the same rule applied where the bytes actually are. A projected tree's
    bytes live in CAS objects that were hashed at ADMISSION (``LocalCAS``
    commits an object only after its content hashes to its own name), on a
    store that is always pod-local — a shared volume is a fill SOURCE whose
    fills cross that same admission check, never the store itself. So the
    only things a boot can still get wrong are structural, and they are
    exactly what is checked here:

    * a stub must name its entry's digest and size — the whole content of a
      stub, so this is complete, not partial;
    * a symlink must resolve to the object its entry names, and that object
      must be present.

    Re-hashing the objects instead would re-read every resident model on every
    boot to re-derive a fact the store already refused to store wrongly.

    What a correct artifact IS lives in :func:`projection.projection_fault`,
    which the snapshot chokepoint's convergence check reads too. This function
    owns the report shape and the wire-digest parse; it does not own a second
    opinion about the rule.
    """

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
    """Build the verification target list for a materialized snapshot.

    ``files`` are protobuf SnapshotFile messages (duck-typed here so this module
    stays free of the transport). Returns ``(targets, skipped_paths)``.

    The digest is read from ``f.digest`` — algorithm-tagged — and from nowhere
    else. There is deliberately no legacy ``f.blake3`` fallback: it is EMPTY on
    every v2 entry, and reading it first is how a whole tree gets a clean verdict
    without being hashed. Files the manifest gives no digest for are
    returned as SKIPPED so the caller must account for them explicitly instead
    of losing them into a pass.
    """

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
