"""Range-read the header / __metadata__ block of a safetensors file on HF.

Used by the HF classifier to detect kohya-trained native LoRAs without
downloading the full safetensors bytes. The safetensors format is:
    [u64 LE: header_len N] [N bytes of UTF-8 JSON header] [tensor data...]

The JSON header carries an optional `__metadata__` dict whose keys are
arbitrary strings — kohya writes ``ss_network_module``, ``ss_base_model_version``,
etc. We pull only the first 8 + ~64 KB to extract this.
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Mapping, Optional


_log = logging.getLogger(__name__)

# Safetensors specifies max header at 100 MB — but real headers are <1 MB for
# even huge models. We cap at 16 MB to be safe; well above any realistic
# kohya-style metadata block.
_MAX_HEADER_BYTES = 16 * 1024 * 1024

# How much we initially fetch to cover the 8-byte length prefix + most headers
# in one round-trip. Larger than typical kohya metadata (~5 KB).
_INITIAL_FETCH_BYTES = 256 * 1024  # 256 KB


def read_safetensors_header_metadata_from_hf(
    repo_id: str,
    filename: str,
    *,
    revision: str = "main",
    token: Optional[str] = None,
) -> Optional[Mapping[str, str]]:
    """Read the __metadata__ dict from a safetensors file on HuggingFace.

    Returns the dict (string keys/values) or None if the file has no
    __metadata__ block (or on any fetch / parse error — this is best-effort
    classification, not load-bearing).
    """
    try:
        from huggingface_hub import hf_hub_url
    except Exception:
        _log.debug("huggingface_hub not available for safetensors header read")
        return None
    try:
        import requests  # noqa: F401  (used via hf_hub_url path; we use it below)
    except Exception:
        _log.debug("requests not available for safetensors header read")
        return None

    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    return _read_safetensors_header_metadata_from_url(url, token=token)


def _read_safetensors_header_metadata_from_url(
    url: str,
    *,
    token: Optional[str] = None,
) -> Optional[Mapping[str, str]]:
    try:
        import requests
    except Exception:
        return None

    headers: dict[str, str] = {
        "Range": f"bytes=0-{_INITIAL_FETCH_BYTES - 1}",
        "User-Agent": "gen-worker/safetensors-header-peek",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
    except Exception as exc:
        _log.debug("safetensors header range-fetch failed: %s", exc)
        return None
    if resp.status_code not in (200, 206):
        _log.debug("safetensors header range-fetch HTTP %d", resp.status_code)
        return None
    body = resp.content
    if len(body) < 8:
        return None
    (header_len,) = struct.unpack("<Q", body[:8])
    if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
        return None

    if 8 + header_len > len(body):
        # Need a second fetch for the rest of the header.
        end = 8 + header_len - 1
        headers["Range"] = f"bytes=0-{end}"
        try:
            resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        except Exception as exc:
            _log.debug("safetensors header second range-fetch failed: %s", exc)
            return None
        if resp.status_code not in (200, 206):
            return None
        body = resp.content
        if len(body) < 8 + header_len:
            return None

    try:
        header_json = json.loads(body[8 : 8 + header_len].decode("utf-8"))
    except Exception as exc:
        _log.debug("safetensors header JSON parse failed: %s", exc)
        return None

    md = header_json.get("__metadata__") if isinstance(header_json, dict) else None
    if not isinstance(md, dict):
        return None
    # Coerce all values to str — safetensors metadata is technically string-only
    # but old writers sometimes embed numbers.
    return {str(k): str(v) for k, v in md.items()}


__all__ = [
    "read_safetensors_header_metadata_from_hf",
]
