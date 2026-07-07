"""Safetensors-only enforcement for override-supplied weight files.

Issue #18: when an invoker overrides a binding via the orchestrator-stamped
``resolved_models`` map, the downloaded snapshot may not have gone through the
build-time validator (which already rejects pickle weights for binding-default
refs). The orchestrator validates pre-dispatch in gen-orchestrator #358, but
this module is the worker-side belt-and-braces gate.

Policy: pickle weight files (``.bin`` / ``.pt`` / ``.ckpt``) are never loaded.
A snapshot whose primary weight is one of those formats is rejected with
``UnsafeFileFormat`` — a terminal error (no retry: pickle never becomes safe
between attempts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


# Weight-file extensions we explicitly accept: loading them does not
# execute arbitrary Python code.
_SAFE_WEIGHT_EXTS: frozenset[str] = frozenset({".safetensors"})

# Pickle-bearing weight-file extensions we refuse to load. Loading any of
# these can execute arbitrary Python code from the snapshot — that's the
# CVE class this gate exists to block.
_UNSAFE_WEIGHT_EXTS: frozenset[str] = frozenset({".bin", ".pt", ".ckpt"})


class UnsafeFileFormat(Exception):
    """Raised when an override download produces a snapshot whose primary
    weight file is a pickle format (.bin / .pt / .ckpt) and there is no
    safetensors sibling.

    Terminal: this is a property of the snapshot's contents, not of the
    download attempt. Retrying won't change the file extensions.
    """


def _walk_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    if root.is_file():
        yield root
        return
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def assert_safe_weight_format(snapshot_dir: Path, *, ref: str = "") -> None:
    """Inspect ``snapshot_dir`` for weight files.

    Accepts (no-op) when:
      - the directory contains at least one ``.safetensors`` file (even if
        pickle siblings exist — some HF repos ship both for back-compat), or
      - the directory contains no weight files at all (the snapshot may be
        a metadata-only repo; the downstream loader will produce a more
        specific error).

    Raises ``UnsafeFileFormat`` when the only weight files present are
    pickle formats (``.bin`` / ``.pt`` / ``.ckpt``).
    """
    root = Path(snapshot_dir)
    safe_hits: list[Path] = []
    unsafe_hits: list[Path] = []
    for f in _walk_files(root):
        ext = f.suffix.lower()
        if ext in _SAFE_WEIGHT_EXTS:
            safe_hits.append(f)
        elif ext in _UNSAFE_WEIGHT_EXTS:
            unsafe_hits.append(f)

    if safe_hits:
        # At least one safe-format weight is present. Acceptable even if
        # pickle siblings exist (HF back-compat repos often do).
        return

    if not unsafe_hits:
        # No weights at all — let the loader produce its own error.
        return

    # Only pickle weights present. Refuse with a terminal error.
    # Pick the largest unsafe file for the error message so the operator
    # sees the file size and can correlate it with the loader's expected
    # primary weight.
    biggest: Optional[Path] = None
    biggest_size = -1
    for f in unsafe_hits:
        try:
            sz = f.stat().st_size
        except Exception:
            sz = 0
        if sz > biggest_size:
            biggest_size = sz
            biggest = f
    named = (biggest or unsafe_hits[0]).as_posix()
    detail = f"refusing to load {named}; safetensors only"
    if ref:
        detail = f"{detail} (override ref={ref!r})"
    raise UnsafeFileFormat(detail)
