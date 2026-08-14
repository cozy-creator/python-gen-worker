"""Dataset snapshot materialization against the tensorhub datasets API.

Free functions used by ``_PublisherMixin.resolve_dataset``: look up a dataset
row by (tenant, name), fetch its blob manifest (a rows.jsonl-style entry index with
presigned URLs / inline text / raw CAS blob digests) — polling 202 until the
async snapshot build is ready (DATASET-V2 contract) — and stream each entry to
disk with sha256 digest verification + bounded retries.
"""
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
    """No dataset exists at the requested address.

    Internal marker only: these helpers see an opaque id and cannot know
    whether the caller or the platform chose it. ``resolve_dataset`` — the
    boundary that DOES know — converts this into the typed request error.
    """


_DOWNLOAD_RETRIES = 3
_DOWNLOAD_BACKOFF_S = 1.0
_CHUNK_BYTES = 1024 * 1024

# DATASET-V2 202 contract.
#
# The old `_MATERIALIZE_BUDGET_S = 30 min` gave up
# on a materialization purely because a stopwatch expired. What the hub
# actually reports is a DEFINITE state per poll — `202 building`, `503
# snapshot_build_failed` (typed, terminal), or the manifest — and every poll
# also re-enqueues the (unique) build job, so a lost build self-heals rather
# than hanging. A 202 is therefore evidence the build is live and the loop
# waits; only silence — no answer at all, or a 5xx that says nothing —
# accumulates the window below, derived from the poll's own request timeout
# times headroom.
_MATERIALIZE_REQUEST_TIMEOUT_S = 120.0
_MATERIALIZE_SILENCE_WINDOW_S = 3.0 * _MATERIALIZE_REQUEST_TIMEOUT_S
_MATERIALIZE_WAIT_HINT_S = 30  # ?wait long-poll hint (server caps ~30s)
_POLL_BACKOFF_START_S = 1.0
_POLL_BACKOFF_CAP_S = 30.0
_POLL_SLEEP_MAX_S = 60.0  # sanity cap even on a huge server retry_after
_CANCEL_POLL_SLICE_S = 0.5


def lookup_dataset_id(base: str, token: str, tenant: str, name: str) -> str:
    """GET /api/v1/datasets?tenant= → dataset_id of the row named ``name``.

    Only used for owner/name refs (local/dev). Production refs arrive as bare
    dataset UUIDs and skip the lookup — a grant-scoped read_dataset capability
    token cannot list datasets at all.
    """

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
    """Sleep in small slices so a cancel lands promptly mid-poll."""
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        _check_cancelled(cancelled)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, _CANCEL_POLL_SLICE_S))


def _retry_after_s(resp: Any, data: Dict[str, Any]) -> float:
    """Server-suggested wait from the 202 body's retry_after (seconds),
    falling back to the Retry-After header; 0 when absent/garbage."""
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
    """Typed snapshot_build_failed from a non-2xx body, else None."""
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
    """GET /datasets/:id/materialize?format=files&include_urls=true.

    Authorizes by dataset id + read_dataset grant (or tenant read perm).
    Returns (snapshot_id, entries); entries carry
    {path, url?, size_bytes?, checksum?, inline_text?, blob_digest?}.

    DATASET-V2 async contract: the hub may answer
    202 ``{status: building, state_version, retry_after}`` while the snapshot
    builds in the background. We long-poll (``?wait=``, ignored by pre-v2
    hubs) and retry with backoff, honoring ``retry_after``.

    There is NO total budget. A 202 is the hub definitively saying
    the build is live, and each poll re-enqueues the unique build job, so the
    loop waits as long as the hub keeps answering. A typed
    ``snapshot_build_failed`` raises ``SnapshotBuildFailedError`` — that is
    the terminal outcome. Only silence gives up: transport errors and 5xx
    say nothing definite, and ``hub_silence_window_s`` bounds how long the
    loop tolerates hearing nothing at all.
    """

    url = (
        f"{base}/api/v1/datasets/{urllib.parse.quote(dataset_id, safe='')}"
        "/materialize?format=files&include_urls=true"
        f"&wait={_MATERIALIZE_WAIT_HINT_S}"
    )
    headers = {"Authorization": f"Bearer {token}"}
    contact = SilenceWindow(hub_silence_window_s)
    backoff = _POLL_BACKOFF_START_S

    def _wait_or_hub_silent(sleep_s: float, why: str) -> None:
        """Sleep before the next poll, unless the hub has told us nothing
        definite for the whole silence window — the only give-up."""
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
            # A definite answer: the build is live hub-side.
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


#: Digest algorithms this reader can verify.
#:
#: No ``blake3``: the dataset-CAS blake3 namespace is deleted and the hub
#: cannot build a blake3 dataset key at all, so a ``blake3:`` checksum names
#: bytes that do not exist. Keeping the hasher would let this reader verify a
#: download that could never have been served, and make a live blake3 entry
#: look supported.
_DIGEST_HASHERS: Dict[str, Callable[[], Any]] = {
    "sha256": hashlib.sha256,
}


def _expected_digest(entry: Dict[str, Any]) -> str:
    """The entry's ALGORITHM-TAGGED checksum, or raise.

    A bare 64-hex is never defaulted to an algorithm: both candidate digests
    are 32 bytes, so the length names nothing and a guess verifies nothing when
    it loses. An untagged or absent checksum is a refusal — there is no
    spelling of "download it unverified".
    """
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
    """Stream ``url`` → ``dest`` (1MiB chunks), verifying digest/size.

    Writes to a UNIQUE temp file in ``dest``'s directory then renames, so a
    partial download can never be mistaken for a complete shard.

    This site is RACY. ``resolve_dataset`` materializes into a pod-wide
    content-keyed cache (``/tmp/gen_worker_datasets/<owner>/<name>/<snapshot>``),
    so two requests in one container asking for one dataset arrive at the SAME
    ``dest`` — and a temp name derived from it lands them on the same temp file.
    The bytes are identical, so the destruction is in the LIFECYCLE, not the
    content: one writer's failure path runs ``tmp.unlink()`` on the other's
    in-flight download, and the victim then fails its own ``replace`` after
    paying for every byte. A unique name per writer costs one ``mkstemp`` and
    removes the shared object entirely; the rename stays atomic, and two
    winners simply publish the same verified bytes.

    the size comparison used to sit after the loop, where the bytes
    are already on disk and the only thing it can report is how far past the
    declaration the shard went. The declared size is known before the first
    byte, so it caps the stream; an entry that declares none falls back to the
    destination filesystem, which is what an unbounded shard exhausts.
    """

    # ONE verified-download discipline (`hubio.fetch.fetch_once`)
    # — writer-unique `.part` staging, the byte cap inside the stream loop,
    # tag-dispatched digest verification, durable atomic rename. This
    # wrapper keeps the dataset policy: `_expected_digest` upstream already
    # refused absent/untagged/unsupported checksums, and a shard that
    # declares no size is bounded by the destination filesystem — the
    # resource an unbounded shard actually exhausts.
    algo = expected_digest.partition(":")[0]
    if algo not in _DIGEST_HASHERS:  # unreachable after _expected_digest; keep the refusal local too
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
        expected_size = entry.get("size_bytes")
        expected_size = int(expected_size) if expected_size is not None else None
        # Safe as a resume check because a downloaded file is renamed into
        # place ONLY after its digest matched — dest existing means verified.
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
        # Demanded only on the URL path: inline entries carry their bytes and
        # `fetch_blob` verifies against the digest that addresses them.
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
