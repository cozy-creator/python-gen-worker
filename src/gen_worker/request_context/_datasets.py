"""Dataset snapshot materialization against the tensorhub datasets API.

Free functions used by ``_PublisherMixin.resolve_dataset``: look up a dataset
row by (owner, name), fetch its parquet materialize manifest (presigned shard
URLs, th#642 wire format), and stream each shard to disk with digest
verification + bounded retries.
"""
from __future__ import annotations

import logging
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..api.errors import AuthError

logger = logging.getLogger(__name__)

_DOWNLOAD_RETRIES = 3
_DOWNLOAD_BACKOFF_S = 1.0
_CHUNK_BYTES = 1024 * 1024


def lookup_dataset_id(base: str, token: str, owner: str, name: str) -> str:
    """GET /api/v1/datasets?owner= → dataset_id of the row named ``name``."""
    import requests

    url = f"{base}/api/v1/datasets?owner={urllib.parse.quote(owner, safe='')}"
    headers = {"Authorization": f"Bearer {token}", "X-Cozy-Owner": owner}
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code in (401, 403):
        raise AuthError(f"dataset lookup unauthorized ({resp.status_code})")
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"dataset lookup failed ({resp.status_code}): {resp.text[:256]}")
    items = resp.json().get("items") or []
    for it in items:
        if str(it.get("name") or "").lower() == name.lower():
            dataset_id = str(it.get("dataset_id") or "")
            if dataset_id:
                return dataset_id
    raise RuntimeError(f"dataset not found for owner={owner} name={name}")


def fetch_materialize_manifest(
    base: str, token: str, owner: str, dataset_id: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """GET /datasets/:id/materialize?format=parquet&include_urls=true.

    Returns (snapshot_id, entries); entries carry
    {path, url?, size_bytes?, checksum?, inline_text?, blob_digest?}.
    """
    import requests

    url = (
        f"{base}/api/v1/datasets/{urllib.parse.quote(dataset_id, safe='')}"
        "/materialize?format=parquet&include_urls=true"
    )
    headers = {"Authorization": f"Bearer {token}", "X-Cozy-Owner": owner}
    resp = requests.get(url, headers=headers, timeout=120)
    if resp.status_code in (401, 403):
        raise AuthError(f"dataset materialize unauthorized ({resp.status_code})")
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(
            f"dataset materialize failed ({resp.status_code}): {resp.text[:256]}"
        )
    data = resp.json() if resp.text else {}
    entries = data.get("entries") or []
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(
            f"dataset materialize returned no entries for dataset_id={dataset_id}"
        )
    return str(data.get("snapshot_id") or ""), entries


def _expected_digest(entry: Dict[str, Any]) -> str:
    """Normalized 'blake3:<hex>' from the entry checksum, or ''."""
    raw = str(entry.get("checksum") or "").strip().lower()
    if not raw:
        return ""
    return raw if ":" in raw else f"blake3:{raw}"


def _download_url_streamed(url: str, dest: Path, *, expected_digest: str,
                           expected_size: Optional[int]) -> None:
    """Stream ``url`` → ``dest`` (1MiB chunks), verifying digest/size.

    Writes to ``dest.tmp`` then renames, so a partial download can never be
    mistaken for a complete shard.
    """
    import blake3
    import requests

    tmp = dest.with_name(dest.name + ".tmp")
    hasher = blake3.blake3() if expected_digest.startswith("blake3:") else None
    total = 0
    with requests.get(url, stream=True, timeout=300) as resp:
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f"shard download failed ({resp.status_code}) url={url[:128]}")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=_CHUNK_BYTES):
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)
                if hasher is not None:
                    hasher.update(chunk)
    if expected_size is not None and total != int(expected_size):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"shard size mismatch: got {total}, want {expected_size}")
    if hasher is not None:
        got = f"blake3:{hasher.hexdigest()}"
        if got != expected_digest:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"shard digest mismatch: got {got}, want {expected_digest}")
    tmp.replace(dest)


def download_entries(
    entries: List[Dict[str, Any]],
    target_root: Path,
    *,
    fetch_blob: Optional[Callable[[str, Path], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> None:
    """Materialize every manifest entry under ``target_root``.

    Presigned ``url`` entries stream to disk with digest verification and up
    to ``_DOWNLOAD_RETRIES`` attempts; ``inline_text`` entries are written
    directly; entries with only a ``blob_digest`` fall back to ``fetch_blob``
    (the repo-CAS by-digest reader).
    """
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if cancelled is not None and cancelled():
            raise RuntimeError("dataset materialization cancelled")
        rel_path = str(entry.get("path") or "").strip().lstrip("/")
        if not rel_path or ".." in rel_path.split("/"):
            continue
        dest = target_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected_digest = _expected_digest(entry)
        expected_size = entry.get("size_bytes")
        expected_size = int(expected_size) if expected_size is not None else None
        if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
            continue  # already materialized

        inline = entry.get("inline_text")
        if isinstance(inline, str) and inline and not entry.get("url"):
            dest.write_text(inline, encoding="utf-8")
            continue

        url = str(entry.get("url") or "").strip()
        blob_digest = str(entry.get("blob_digest") or "").strip()
        if not url and not blob_digest:
            raise RuntimeError(f"dataset entry {rel_path!r} has neither url nor blob_digest")

        last_exc: Optional[Exception] = None
        for attempt in range(_DOWNLOAD_RETRIES):
            try:
                if url:
                    _download_url_streamed(
                        url, dest,
                        expected_digest=expected_digest, expected_size=expected_size,
                    )
                else:
                    assert fetch_blob is not None
                    fetch_blob(blob_digest, dest)
                last_exc = None
                break
            except AuthError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "dataset shard %s download attempt %d/%d failed: %s",
                    rel_path, attempt + 1, _DOWNLOAD_RETRIES, exc,
                )
                time.sleep(_DOWNLOAD_BACKOFF_S * (attempt + 1))
        if last_exc is not None:
            raise RuntimeError(
                f"dataset shard {rel_path!r} failed after {_DOWNLOAD_RETRIES} attempts: {last_exc}"
            ) from last_exc
