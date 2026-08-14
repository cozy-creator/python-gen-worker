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
*   **MEMOIZED per (root, digest)** — content-addressed bytes cannot change
    under a digest, so re-verifying the same blob in the same root within a
    process is pure waste. Memoization records only PASSES; a failure is always
    re-checked because it triggers deletion and refetch.

Every report carries the DENOMINATOR it examined. A verdict computed from zero
files is a broken reader, not a clean tree, and ``ok`` is false in that case
whenever files were expected.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .chunk_cas import DigestMismatch, hash_file, parse_cas_ref
from .cozy_snapshot import _is_part_file, _is_parts_manifest, _norm_rel_path

__all__ = [
    "VerifyReport",
    "VerifyTarget",
    "snapshot_verify_targets",
    "verify_files",
    "clear_memo",
]

_log = logging.getLogger(__name__)

# (blobs_root, algo:hex) -> True. Passes only.
_memo: Dict[Tuple[str, str], bool] = {}
_memo_lock = threading.Lock()


def clear_memo() -> None:
    with _memo_lock:
        _memo.clear()


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
    memo_hits: int = 0
    bytes_hashed: int = 0
    bad: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    # expected is what the caller believed it was checking. When it disagrees
    # with `examined`, the reader is wrong and the verdict is not trustworthy.
    expected: int = 0

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
    blobs_root: str = "",
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
            algo, want = parse_cas_ref(t.ref)
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
        memo_key = (blobs_root, f"{algo}:{want}")
        try:
            if not t.path.exists():
                raise DigestMismatch("missing")
            actual = t.path.stat().st_size
            if t.size and actual != t.size:
                raise DigestMismatch(f"size mismatch (declared {t.size}, on disk {actual})")
            with _memo_lock:
                hit = _memo.get(memo_key, False)
            if hit:
                with lock:
                    rep.examined += 1
                    rep.memo_hits += 1
                return
            got = hash_file(t.path, algo)
            if got.lower() != want:
                raise DigestMismatch(
                    f"{algo} of bytes is {got[:16]}…, manifest says {want[:16]}…"
                )
            with _memo_lock:
                _memo[memo_key] = True
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
        "volume_verify examined=%d/%d hashed=%d memo=%d bytes=%d bad=%d",
        rep.examined, rep.expected, rep.hashed, rep.memo_hits, rep.bytes_hashed, len(rep.bad),
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
        if _is_parts_manifest(path_attr) or _is_part_file(path_attr):
            continue  # not materialized: parts live only in blobs/
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
