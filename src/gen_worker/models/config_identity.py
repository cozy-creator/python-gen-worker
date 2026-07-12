"""Canonical JSON-config digests for content-keyed sharing (gw#479).

Split-vendor repo pairs ship byte-identical component WEIGHTS whose tiny JSON
sidecars differ only in save-era serialization: provenance stamps
(`_name_or_path`, `transformers_version`, `_diffusers_version`), explicit
class-default fields vs omitted ones, `null` vs absent, and the transformers
4.56 `torch_dtype` -> `dtype` key rename (live evidence: Qwen-Image vs
Qwen-Image-Edit-2511 text encoders, saved by 4.53 vs 4.57). Content identity
must track what the loader CONSTRUCTS, not save-time vocabulary — so weights
keep their manifest blake3 and small JSON sidecars are hashed canonically:

- ``config.json`` folds through the INSTALLED transformers' AutoConfig
  ``to_diff_dict()`` (explicit defaults drop out) when it parses; anything
  else (diffusers configs, tokenizer/processor configs) gets structural
  canonicalization only.
- structural canonicalization: recursively drop provenance keys and nulls,
  rename ``torch_dtype`` -> ``dtype``, then a sorted-key compact dump.

Content keys are PROCESS-LOCAL identity (the in-memory shared-component
cache), so folding through whatever transformers version is installed is
safe by construction — both lanes in one process fold identically. Any
parse failure falls back conservatively (raw manifest digest = no sharing).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# JSON files above this are data (tokenizer.json vocab/merges), not configs:
# they keep their manifest digest.
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
        # Composite-config scalar dedupe (qwen live evidence): transformers
        # mirrors token-id scalars between a composite VL config and its
        # sub-configs, and WHERE they serialize moved across versions — 4.53
        # wrote them in text_config (sometimes both), 4.57 at top level
        # only. Same materialized values, different paths. Canonical form:
        # a child scalar duplicating the parent drops; a child-only scalar
        # HOISTS to the parent; a child value CONFLICTING with the parent
        # stays put (real content, keys must separate).
        for k in sorted((k for k, v in out.items() if isinstance(v, dict)), key=str):
            child = out[k]
            kept = {}
            for kk, vv in child.items():
                if isinstance(vv, (dict, list)):
                    kept[kk] = vv
                elif kk in out and not isinstance(out[kk], (dict, list)):
                    if out[kk] == vv:
                        continue      # duplicate of parent
                    kept[kk] = vv     # conflict: keep both sides
                elif kk in out:
                    kept[kk] = vv     # parent holds a container under this name
                else:
                    out[kk] = vv      # hoist child-only scalar to the parent
            out[k] = kept
        return out
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def _digest_of(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return "cj:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _transformers_folded(path: Path) -> Any:
    """AutoConfig round-trip for a transformers ``config.json``: class
    defaults fold out of the serialized form, save-era vocabulary
    normalizes. None when this isn't a transformers config."""
    import transformers

    cfg = transformers.AutoConfig.from_pretrained(
        str(path.parent), local_files_only=True, trust_remote_code=False,
    )
    return cfg.to_diff_dict()


def canonical_json_digest(path: Path) -> str:
    """Canonical digest ("cj:"-prefixed, never collides with a raw blake3)
    of one small JSON sidecar; "" when the file cannot be canonicalized
    (caller keeps the raw manifest digest — conservative no-share)."""
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
            pass  # diffusers/other config: structural canonicalization only
    return _digest_of(_normalize(parsed))


__all__ = ["canonical_json_digest", "CANONICAL_JSON_MAX_BYTES"]
