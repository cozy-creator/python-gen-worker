"""Installed-distribution RECORD manifests — the declared content facts.

A wheel's ``dist-info/RECORD`` already carries, per installed file, the
**sha256 the installer verified at install time** and the size it wrote. That
is the same fact ``sha256(content)`` recomputes, so a boot that needs the
content identity of ``libtorch_cuda.so`` can READ it instead of re-hashing
3.96 GB of shipped toolchain (measured: 10.6-17.3 s per process, 68 % of a
33.7 s cold boot through ``env_seal.establish``).

Two consumers, ONE derivation:

* :func:`digest_for` — per-FILE identity for ``env_seal``'s native-library
  manifest (the seal's ``loaded_libs`` fact and the live substitution check).
* :func:`record_texts` — per-DISTRIBUTION RECORD text for
  ``compile_cache.toolchain_digest``'s binary half, which reads exactly these
  files.

**The digest is byte-identical to the hash it replaces.** RECORD stores
``sha256=<urlsafe-b64 of the 32 raw bytes>``; decoding to hex and truncating
to 16 chars reproduces ``hashlib.sha256(content).hexdigest()[:16]`` exactly. So
this is a pure COST move: no seal value changes, no compiled graph re-keys.

**What RECORD trust is, precisely:**

* The installer verified the hash when it wrote the file. It is NOT re-verified
  at load time, so RECORD trust is a claim about install, extended forward by
  the two guards :func:`digest_for` applies before honouring it — the file's
  size still matches RECORD, and the file has not been written since the
  RECORD that describes it was written (``mtime_ns <= RECORD's mtime_ns``).
* CAUGHT, and hashed instead of trusted: any file absent from every RECORD
  (system libs, an ``LD_PRELOAD`` object, a lib from a non-wheel install), any
  size change, any in-place rewrite that leaves a newer mtime, any dist whose
  RECORD does not describe itself.
* NOT CAUGHT: an in-place rewrite that preserves the byte SIZE *and* restores
  the original ``mtime_ns``. A full-hash boot would have seen that and moved
  the seal. This is the one place RECORD trust is weaker than hashing, and it is
  the same forge the memo and the ``_lib_digest`` lru_cache (both keyed on
  ``(path, mtime_ns, size)``) are open to — an actor able to do it already holds
  write access to site-packages, i.e. to ``env_seal.py`` itself.
"""

from __future__ import annotations

import base64
import binascii
import csv
import functools
import importlib.metadata
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

#: How much of the sha256 hex our digests carry (``env_seal`` convention).
DIGEST_HEX_LEN = 16

_SHA256_FIELD = "sha256="

#: Only NATIVE artifacts are indexed per-file: the seal's manifest is the
#: shipped ``.so`` set, and indexing all 27k rows of a torch env costs memory
#: for rows nothing looks up.
_NATIVE_MARKER = ".so"


@dataclass(frozen=True)
class RecordedFile:
    """One RECORD row, resolved to an absolute path.

    ``digest`` is already in ``env_seal`` form (16 hex chars); ``record_mtime_ns``
    is the mtime of the RECORD that made the claim, which is what bounds how
    far forward the claim may be trusted.
    """

    path: str
    digest: str
    size: int
    dist: str
    record_mtime_ns: int


def _decode_sha256(field: str) -> str:
    """RECORD's ``sha256=<b64>`` -> the first :data:`DIGEST_HEX_LEN` hex chars,
    or ``""`` when the row carries no usable sha256 (legacy md5 rows, the
    hashless self-row, a truncated field)."""
    if not field.startswith(_SHA256_FIELD):
        return ""
    raw = field[len(_SHA256_FIELD):]
    try:
        decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    except (binascii.Error, ValueError):
        return ""
    if len(decoded) != 32:
        return ""
    return decoded.hex()[:DIGEST_HEX_LEN]


def _record_rows(text: str) -> List[List[str]]:
    return [row for row in csv.reader(text.splitlines()) if len(row) == 3]


class _Scan(NamedTuple):
    texts: Dict[str, str]
    native: Dict[str, RecordedFile]


@functools.lru_cache(maxsize=1)
def _scan() -> _Scan:
    """ONE walk of the installed distributions, feeding both readers.

    Measured on the reference env: 99 distributions, 2.8 MB of RECORD text,
    ~0.26 s cold and free thereafter — against the 6.3-17.3 s of SHA-256 it
    replaces. Paths are joined, never ``realpath``d, at scan time: resolving
    27k rows costs 1.5 s while resolving the ~36 the seal asks about costs
    nothing (:func:`digest_for` resolves the QUERY instead).
    """
    texts: Dict[str, str] = {}
    native: Dict[str, RecordedFile] = {}
    try:
        dists = list(importlib.metadata.distributions())
    except Exception:  # pragma: no cover - a broken site-packages
        logger.debug("dist_records: distribution walk failed", exc_info=True)
        return _Scan(texts, native)
    for dist in dists:
        try:
            name = str(dist.metadata.get("Name") or "").lower()
            text = dist.read_text("RECORD")
        except Exception:
            continue
        if not name:
            continue
        # A distribution with NO RECORD is still recorded, as the empty text
        # its reader has always seen (`compile_cache` hashed `read_text(...)
        # or ""`): dropping the row here would silently move the `toolchain`
        # key axis for any env that has one.
        texts[name] = text or ""
        if not text:
            continue
        rows = _record_rows(text)
        record_mtime = _record_mtime_ns(dist, rows)
        if not record_mtime:
            continue  # an unanchored claim: those files get hashed
        for rel, hash_field, size_field in rows:
            if _NATIVE_MARKER not in os.path.basename(rel):
                continue
            digest = _decode_sha256(hash_field)
            if not digest:
                continue
            try:
                size = int(size_field)
                path = os.path.normpath(str(dist.locate_file(rel)))
            except (OSError, ValueError):
                continue
            native[path] = RecordedFile(
                path=path, digest=digest, size=size, dist=name,
                record_mtime_ns=record_mtime)
    return _Scan(texts, native)


def record_texts() -> Mapping[str, str]:
    """``{distribution name (lowercased): RECORD text}`` for every installed
    distribution that ships one — ``compile_cache.toolchain_digest``'s
    per-PACKAGE binary half, reading the same scan the seal's
    per-FILE half reads."""
    return _scan().texts


def _record_mtime_ns(dist: importlib.metadata.Distribution,
                     rows: List[List[str]]) -> int:
    """mtime of the RECORD file itself, located from its OWN row (installers
    list ``<dist-info>/RECORD`` hashless). 0 when the file does not describe
    itself — no anchor, so :func:`digest_for` refuses to trust that dist."""
    for row in rows:
        if not row[0].endswith(".dist-info/RECORD"):
            continue
        try:
            return os.stat(str(dist.locate_file(row[0]))).st_mtime_ns
        except OSError:
            return 0
    return 0


def native_index() -> Mapping[str, RecordedFile]:
    """``{absolute path: RecordedFile}`` for every native shared object any
    installed distribution RECORDs — the seal's per-FILE half."""
    return _scan().native


def digest_for(path: str, mtime_ns: int, size: int) -> Optional[str]:
    """The RECORDed content digest of ``path``, or ``None`` when no
    distribution's RECORD covers it under both guards (see the module
    docstring). ``None`` is the instruction to HASH the file — never a reason
    to skip it."""
    index = native_index()
    entry = index.get(os.path.normpath(path))
    if entry is None:
        entry = index.get(os.path.realpath(path))
    if entry is None:
        return None
    if entry.size != size:
        return None  # rewritten to a different length: RECORD is stale
    if mtime_ns > entry.record_mtime_ns:
        return None  # written after the claim was made: RECORD is stale
    return entry.digest


def coverage() -> Tuple[int, int]:
    """(distributions seen, native files indexed) — the number
    a boot reports so a fleet-wide coverage regression is visible in the phase
    row instead of arriving as a slow boot nobody explains."""
    return len(record_texts()), len(native_index())


def reset_cache() -> None:
    """Forget the scan (tests that install a distribution mid-process)."""
    _scan.cache_clear()


__all__ = [
    "DIGEST_HEX_LEN",
    "RecordedFile",
    "coverage",
    "digest_for",
    "native_index",
    "record_texts",
    "reset_cache",
]
