"""Canonical JSON-config digests for content-keyed sharing."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CANONICAL_JSON_MAX_BYTES = 256 * 1024

_PROVENANCE_KEYS = frozenset({
    "_name_or_path", "transformers_version", "_diffusers_version",
})


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in _PROVENANCE_KEYS or v is None:
                continue
            if k == "torch_dtype":
                k = "dtype"
            out[k] = _normalize(v)
        for k in sorted((k for k, v in out.items() if isinstance(v, dict)), key=str):
            child = out[k]
            kept = {}
            for kk, vv in child.items():
                if isinstance(vv, (dict, list)):
                    kept[kk] = vv
                elif kk in out and not isinstance(out[kk], (dict, list)):
                    if out[kk] == vv:
                        continue
                    kept[kk] = vv
                elif kk in out:
                    kept[kk] = vv
                else:
                    out[kk] = vv
            out[k] = kept
        return out
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def _digest_of(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return "cj:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _transformers_folded(path: Path) -> Any:
    import transformers

    cfg = transformers.AutoConfig.from_pretrained(
        str(path.parent), local_files_only=True, trust_remote_code=False,
    )
    return cfg.to_diff_dict()


def canonical_json_digest(path: Path) -> str:
    """Canonical digest ("cj:"-prefixed, never collides with a raw blake3) of one small JSON sidecar; "" when the file cannot be canonicalized (caller keeps the raw manifest digest — conservative no-share)."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    if len(raw) > CANONICAL_JSON_MAX_BYTES:
        return ""
    try:
        parsed = json.loads(raw)
    except ValueError:
        return ""
    if path.name == "config.json":
        try:
            parsed = _transformers_folded(path)
        except Exception:
            pass
    return _digest_of(_normalize(parsed))


__all__ = ["canonical_json_digest", "CANONICAL_JSON_MAX_BYTES"]
