"""Installed-distribution RECORD manifests — the declared content facts."""

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

DIGEST_HEX_LEN = 16

_SHA256_FIELD = "sha256="

_NATIVE_MARKER = ".so"


@dataclass(frozen=True)
class RecordedFile:
    """One RECORD row, resolved to an absolute path."""

    path: str
    digest: str
    size: int
    dist: str
    record_mtime_ns: int


def _decode_sha256(field: str) -> str:
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
        texts[name] = text or ""
        if not text:
            continue
        rows = _record_rows(text)
        record_mtime = _record_mtime_ns(dist, rows)
        if not record_mtime:
            continue
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
    """``{distribution name (lowercased): RECORD text}`` for every installed distribution that ships one — ``compile_cache.toolchain_digest``'s per-PACKAGE binary half, reading the same scan the seal's pe..."""
    return _scan().texts


def _record_mtime_ns(dist: importlib.metadata.Distribution,
                     rows: List[List[str]]) -> int:
    for row in rows:
        if not row[0].endswith(".dist-info/RECORD"):
            continue
        try:
            return os.stat(str(dist.locate_file(row[0]))).st_mtime_ns
        except OSError:
            return 0
    return 0


def native_index() -> Mapping[str, RecordedFile]:
    """``{absolute path: RecordedFile}`` for every native shared object any installed distribution RECORDs — the seal's per-FILE half."""
    return _scan().native


def digest_for(path: str, mtime_ns: int, size: int) -> Optional[str]:
    """The RECORDed content digest of ``path``, or ``None`` when no distribution's RECORD covers it under both guards (see the module docstring)."""
    index = native_index()
    entry = index.get(os.path.normpath(path))
    if entry is None:
        entry = index.get(os.path.realpath(path))
    if entry is None:
        return None
    if entry.size != size:
        return None
    if mtime_ns > entry.record_mtime_ns:
        return None
    return entry.digest


def coverage() -> Tuple[int, int]:
    """(distributions seen, native files indexed) — the number a boot reports so a fleet-wide coverage regression is visible in the phase row instead of arriving as a slow boot nobody explains."""
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
