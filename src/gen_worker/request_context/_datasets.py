"""Dataset snapshot materialization against the tensorhub datasets API."""
from __future__ import annotations

import hashlib
import logging
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..api.errors import AuthError, SnapshotBuildFailedError
from ..bounded_stream import free_space_bound
from ..hub_error import hub_error_of
from ..hubio.fetch import fetch_once
from ..stall import SilenceWindow
import requests

logger = logging.getLogger(__name__)


class DatasetRefNotFound(RuntimeError):
    """No dataset exists at the requested address."""


_DOWNLOAD_RETRIES = 3
_DOWNLOAD_BACKOFF_S = 1.0
_CHUNK_BYTES = 1024 * 1024

_MATERIALIZE_REQUEST_TIMEOUT_S = 120.0
_MATERIALIZE_SILENCE_WINDOW_S = 3.0 * _MATERIALIZE_REQUEST_TIMEOUT_S
_MATERIALIZE_WAIT_HINT_S = 30
_POLL_BACKOFF_START_S = 1.0
_POLL_BACKOFF_CAP_S = 30.0
_POLL_SLEEP_MAX_S = 60.0
_CANCEL_POLL_SLICE_S = 0.5


def lookup_dataset_id(base: str, token: str, tenant: str, name: str) -> str:
    """GET /api/v1/datasets?tenant= → dataset_id of the row named ``name``."""

    url = f"{base}/api/v1/datasets?tenant={urllib.parse.quote(tenant, safe='')}"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code in (401, 403):
        raise AuthError(
            f"dataset lookup unauthorized ({resp.status_code}): "
            f"{hub_error_of(resp).detail() or resp.text[:200]}"
        )
    if resp.status_code == 404:
        raise DatasetRefNotFound(f"no such tenant dataset list: tenant={tenant}")
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"dataset lookup failed ({resp.status_code}): {resp.text[:256]}")
    items = resp.json().get("items") or []
    for it in items:
        if str(it.get("name") or "").lower() == name.lower():
            dataset_id = str(it.get("dataset_id") or "")
            if dataset_id:
                return dataset_id
    raise DatasetRefNotFound(f"dataset not found for tenant={tenant} name={name}")


def _check_cancelled(cancelled: Optional[Callable[[], bool]]) -> None:
    if cancelled is not None and cancelled():
        raise RuntimeError("dataset materialization cancelled")


def _sleep_cancellable(seconds: float, cancelled: Optional[Callable[[], bool]]) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        _check_cancelled(cancelled)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, _CANCEL_POLL_SLICE_S))


def _retry_after_s(resp: Any, data: Dict[str, Any]) -> float:
    for raw in (data.get("retry_after"), resp.headers.get("Retry-After")):
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val
    return 0.0


def _snapshot_build_failed(resp: Any) -> Optional[SnapshotBuildFailedError]:
    body = resp.text or ""
    if "snapshot_build_failed" not in body:
        return None
    error_code = ""
    try:
        data = resp.json()
        if isinstance(data, dict):
            error_code = str(data.get("error_code") or "")
    except Exception:
        pass
    detail = f": {error_code}" if error_code else ""
    return SnapshotBuildFailedError(
        f"dataset snapshot build failed hub-side{detail} "
        f"(http {resp.status_code}); a new materialize request re-enqueues the build",
        error_code=error_code,
    )


def fetch_materialize_manifest(
    base: str,
    token: str,
    dataset_id: str,
    *,
    hub_silence_window_s: float = _MATERIALIZE_SILENCE_WINDOW_S,
    cancelled: Optional[Callable[[], bool]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """GET /datasets/:id/materialize?format=files&include_urls=true."""

    url = (
        f"{base}/api/v1/datasets/{urllib.parse.quote(dataset_id, safe='')}"
        "/materialize?format=files&include_urls=true"
        f"&wait={_MATERIALIZE_WAIT_HINT_S}"
    )
    headers = {"Authorization": f"Bearer {token}"}
    contact = SilenceWindow(hub_silence_window_s)
    backoff = _POLL_BACKOFF_START_S

    def _wait_or_hub_silent(sleep_s: float, why: str) -> None:
        if contact.stalled():
            raise RuntimeError(
                f"dataset materialize: the hub said nothing definite for "
                f"{contact.silent_for():.0f}s (window "
                f"{contact.window_s:.0f}s) for dataset_id={dataset_id} "
                f"(last state: {why})"
            )
        _sleep_cancellable(min(sleep_s, _POLL_SLEEP_MAX_S), cancelled)

    while True:
        _check_cancelled(cancelled)
        try:
            resp = requests.get(
                url, headers=headers, timeout=_MATERIALIZE_REQUEST_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            logger.warning(
                "dataset %s materialize request failed (%s); retrying", dataset_id, exc,
            )
            _wait_or_hub_silent(backoff, f"transport error: {exc}")
            backoff = min(backoff * 2.0, _POLL_BACKOFF_CAP_S)
            continue

        if resp.status_code in (401, 403):
            raise AuthError(
                f"dataset materialize unauthorized ({resp.status_code}): "
                f"{hub_error_of(resp).detail() or resp.text[:200]}"
            )

        if resp.status_code == 202:
            contact.touch()
            try:
                data = resp.json() if resp.text else {}
            except ValueError:
                data = {}
            if not isinstance(data, dict):
                data = {}
            retry_after = _retry_after_s(resp, data)
            sleep_s = retry_after if retry_after > 0 else backoff
            logger.info(
                "dataset %s snapshot building (state_version=%s); polling again in %.1fs",
                dataset_id, data.get("state_version"), min(sleep_s, _POLL_SLEEP_MAX_S),
            )
            _wait_or_hub_silent(sleep_s, "202 building")
            backoff = min(backoff * 2.0, _POLL_BACKOFF_CAP_S)
            continue

        if resp.status_code < 200 or resp.status_code >= 300:
            failed = _snapshot_build_failed(resp)
            if failed is not None:
                raise failed
            if resp.status_code in (502, 503, 504):
                logger.warning(
                    "dataset %s materialize got %d; retrying", dataset_id, resp.status_code,
                )
                _wait_or_hub_silent(backoff, f"http {resp.status_code}")
                backoff = min(backoff * 2.0, _POLL_BACKOFF_CAP_S)
                continue
            if resp.status_code == 404:
                raise DatasetRefNotFound(
                    f"dataset_id={dataset_id} does not exist"
                )
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


_DIGEST_HASHERS: Dict[str, Callable[[], Any]] = {
    "sha256": hashlib.sha256,
}


def _expected_digest(entry: Dict[str, Any]) -> str:
    raw = str(entry.get("checksum") or "").strip().lower()
    if not raw:
        raise RuntimeError("dataset entry has no checksum — refusing to download it unverified")
    algo, sep, _ = raw.partition(":")
    if not sep:
        raise RuntimeError(
            f"dataset entry checksum {raw!r} is untagged — 64 hex chars name no algorithm"
        )
    if algo not in _DIGEST_HASHERS:
        raise RuntimeError(f"unsupported digest algorithm {algo!r} in checksum {raw!r}")
    return raw


def _download_url_streamed(url: str, dest: Path, *, expected_digest: str,
                           expected_size: Optional[int]) -> None:

    algo = expected_digest.partition(":")[0]
    if algo not in _DIGEST_HASHERS:
        raise RuntimeError(f"unsupported digest algorithm {algo!r}")
    declared = int(expected_size) if expected_size is not None else 0
    fetch_once(
        url, dest,
        expected_digest=expected_digest,
        expected_size=declared,
        cap_bytes=0 if declared > 0 else free_space_bound(dest.parent),
        resume=False,
    )


def download_entries(
    entries: List[Dict[str, Any]],
    target_root: Path,
    *,
    fetch_blob: Optional[Callable[[str, Path], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> None:
    """Materialize every manifest entry under ``target_root``."""
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
        expected_size = entry.get("size_bytes")
        expected_size = int(expected_size) if expected_size is not None else None
        if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
            continue

        inline = entry.get("inline_text")
        if isinstance(inline, str) and inline and not entry.get("url"):
            dest.write_text(inline, encoding="utf-8")
            continue

        url = str(entry.get("url") or "").strip()
        blob_digest = str(entry.get("blob_digest") or "").strip()
        if not url and not blob_digest:
            raise RuntimeError(f"dataset entry {rel_path!r} has neither url nor blob_digest")
        expected_digest = _expected_digest(entry) if url else ""

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
