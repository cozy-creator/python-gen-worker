"""Tensorhub commit client — the ONE publish path (tensorhub #515).

The write API is HF `create_commit`-shaped:

  POST /api/v1/repos/{tenant}/{name}/commits
      {operations: [{type:"add", path, blake3, size_bytes}, ...],
       tags: [{tag, default_flavor?}], mode: "merge"|"replace",
       flavor/dtype/file_layout/file_type, metadata, provenance, repo spec}
  → {revision_id, uploads: [{path, exists, upload_id, part_urls, part_size,
                             complete_url, ...}]}

Then per non-dedup'd upload: PUT the parts, POST …/uploads/{id}/complete
with the ETags, and finally POST …/commits/{revision_id}/finalize (no body;
202 → poll). One commit == one checkpoint == one flavor.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import requests
import socket
from blake3 import blake3
from ..api.errors import AuthError
from ..request_context._helpers import _parse_owner_repo
from ..stall import SilenceWindow
from gen_worker.models.refs import flavor_token as _token

logger = logging.getLogger(__name__)

_RETRY_BASE_DELAY_S = 1.0
_RETRY_MAX_DELAY_S = 30.0

# Connect timeout split from read (mirrors gen_worker._upload_transport and
# the gw#456 download-side floor): a dead host fails in seconds instead of
# consuming the whole read budget.
_CONNECT_TIMEOUT_S = 15.0

# gw#462: bounded re-uploads of ONE file whose staged bytes the hub lost
# (409 staging_object_missing from /complete, th#699). Each attempt re-opens
# the upload via POST .../commits/<rev>/uploads and re-PUTs just that file.
_REUPLOAD_ATTEMPTS = 2

# tensorhub's /complete verifies the whole object synchronously (streams it
# back from R2 and hashes it) before responding, holding a per-upload lock
# for the duration. For large single files this can outlast whatever timeout
# sits in front of tensorhub -- the client sees a 5xx/timeout on an attempt
# that is still running server-side, retries, and races the first attempt
# into 409 upload_complete_in_progress. Found live mirroring a ~6.94GB SDXL
# checkpoint: the default 120s request timeout expired while the server was
# still hashing, the retry got 409, and _upload_one raised immediately --
# aborting the whole commit even though the first attempt was about to
# succeed (e2e tracker #110).
_COMPLETE_TIMEOUT_S = 600.0
_COMPLETE_IN_PROGRESS_POLL_S = 5.0

# A severed /complete connection is NOT fatal either: middleboxes on the
# worker->hub path (NAT idle eviction, tunnel circuit caps) kill the idle
# ~5-minute verify of multi-GB shards, so the client sees a network error
# while the server may finish (sess.Finalized fast path answers the re-POST)
# or may have aborted (a re-POST restarts the verify). Re-POST patiently —
# each attempt can legitimately take a full verify. Found live twice on the
# flux2-klein-4b clone (te#44 J9 runs 7+8: RemoteDisconnected at ~4m50s).
_COMPLETE_NETWORK_RETRY_DELAY_S = 15.0

# gw#666 (th#1166 finding D): the old `_COMPLETE_NETWORK_MAX_WAIT_S = 1800.0`
# abandoned the publish at 30 minutes of wall time — throwing away a commit
# whose bytes were already uploaded, because a stopwatch expired while the
# hub was still verifying them.
#
# The hub tells us which case we are in. A `409 upload_complete_in_progress`
# is a DEFINITE answer: the hub is up and something holds the completion lock
# (which the hub sets NX with a TTL and never renews, so the 409 cannot
# outlive a dead holder). While those keep arriving the verify is advancing
# and there is nothing to give up on. What is NOT definite is silence — a
# severed connection or an edge-masked 5xx tells us nothing — so only those
# accumulate the window below. It is derived from the call cadence: two full
# verify-length attempts' worth of hearing nothing definite.
#
# pgw#743 RESIZED it from 2 to 6 verify-lengths on measured evidence. Two
# 58-minute clones died here: the chaos hub's container was being rebuilt, its
# tunnel served an HTML 503 for LONGER THAN THE 20-MINUTE WINDOW, and the
# publish was declared fatal with 53 GiB of already-paid download in hand. The
# window is a statement about how long the channel may plausibly be gone, and
# a container rebuild is tens of minutes, not two. The arithmetic also favours
# waiting: an hour parked on the CPU rig that runs these jobs costs about what
# re-downloading costs, and unlike the re-download it cannot fail again the
# same way. Waiting stays observable — the loop beats liveness every pass.
_COMPLETE_SILENCE_WINDOW_S = 6.0 * _COMPLETE_TIMEOUT_S

_SESSION: Optional[requests.Session] = None


def _http_session() -> requests.Session:
    """Shared session with TCP keepalives so NAT/conntrack middleboxes don't
    evict the idle flow while the server verifies (no response bytes for
    minutes)."""

    global _SESSION
    if _SESSION is not None:
        return _SESSION
    socket_options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    for name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 30), ("TCP_KEEPCNT", 20)):
        if hasattr(socket, name):
            socket_options.append((socket.IPPROTO_TCP, getattr(socket, name), value))

    class _KeepaliveAdapter(requests.adapters.HTTPAdapter):
        def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
            kwargs["socket_options"] = socket_options
            super().init_poolmanager(*args, **kwargs)

    session = requests.Session()
    adapter = _KeepaliveAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    _SESSION = session
    return session


class HubPublishError(RuntimeError):
    """Terminal failure talking to tensorhub's commit API."""


class BankedBlobGoneError(HubPublishError):
    """A commit referenced a banked CAS blob (no local bytes) that the hub
    no longer has — the bank lied (GC race). Callers fall back to a full
    download (th#592 download-skip is fail-open)."""


class _StagingLostError(HubPublishError):
    """/complete reported 409 staging_object_missing: the staged bytes are
    gone server-side and retrying complete can never succeed. Internal —
    _upload_one converts it into a re-open + re-upload of just that file."""


def _retry_after_s(resp: requests.Response) -> Optional[float]:
    try:
        value = float(str(resp.headers.get("Retry-After") or "").strip())
    except Exception:
        return None
    return min(value, _RETRY_MAX_DELAY_S) if value > 0 else None


def _error_code_of(resp: requests.Response) -> str:
    """Best-effort extraction of the structured `error.code` field
    (docs/api-conventions.md: `{"error": {"code": ..., ...}}`); "" if the
    body isn't that shape."""
    try:
        body = resp.json() if resp.text else {}
    except Exception:
        return ""
    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    if not isinstance(err, dict):
        return ""
    return str(err.get("code") or "")


# pgw#738/#743: how long _send_with_retries tolerates hearing nothing
# DEFINITE from the hub (network errors, 5xx, 429, proxy-shaped 404s) before
# giving up. A hub restart is seconds and a tunnel re-dial is minutes; the old
# 5-attempt cap (~2 min) classified both FATAL and threw away paid GPU work
# at the finish line (two 58-min clones, #743). Silence-bounded per gw#666.
_SEND_SILENCE_WINDOW_S = 600.0


def _send_with_retries(
    what: str,
    send: Callable[[], requests.Response],
    *,
    silence_window_s: Optional[float] = None,
    definite: Optional[Callable[[requests.Response], bool]] = None,
) -> requests.Response:
    """Origin-discriminating, silence-bounded retry (pgw#715/#738/#743).

    Only a DEFINITE hub answer ends the loop: 2xx/3xx, or a 4xx that is
    really the hub speaking (not a proxy's offline page). Indefinite answers
    — network errors, 429, 5xx, and ANY proxy-shaped status (ngrok with no
    healthy backend while the hub restarts answers 404 AND 503 with the same
    HTML page; pgw#743) — retry with backoff until nothing definite has been
    heard for ``silence_window_s``, then surface the last state. Unknown
    exception types never leak raw: they become a typed ``HubPublishError``.
    Returns the last response for non-retryable statuses — callers keep their
    own status handling.
    """
    from .. import activity as _activity
    from ..http_origin import is_definite_hub_answer

    if silence_window_s is None:
        silence_window_s = _SEND_SILENCE_WINDOW_S
    contact = SilenceWindow(silence_window_s)
    delay = _RETRY_BASE_DELAY_S
    attempt = 0
    last_resp: Optional[requests.Response] = None
    last_exc: Optional[BaseException] = None
    while True:
        attempt += 1
        try:
            resp = send()
        except requests.RequestException as exc:
            last_resp, last_exc = None, exc
        except Exception as exc:  # noqa: BLE001 - typed terminal, never a raw leak
            raise HubPublishError(
                f"{what} failed ({type(exc).__name__}): {exc}") from exc
        else:
            code = int(resp.status_code)
            # `definite` lets a caller supply its OWN definiteness rule for a
            # route whose refusals are not envelope-shaped (th#1303's v2
            # completion answers with a projection carrying an explicit
            # `retryable` flag). The default stays the shape heuristic.
            is_definite = definite(resp) if definite is not None else is_definite_hub_answer(resp)
            if code != 429 and is_definite:
                return resp  # definite hub answer
            last_resp, last_exc = resp, None
            delay = _retry_after_s(resp) or delay
        # pgw#743: this loop can now legitimately run for the better part of
        # an hour riding out a hub rebuild, and a publisher that goes silent
        # for an hour is exactly the dead-job signature the watchdogs kill on
        # (pgw#738). Waiting IS work — say so on the liveness channel.
        _activity.note_progress()
        if contact.stalled():
            if last_resp is not None:
                return last_resp
            raise HubPublishError(
                f"{what} failed (network, no definite hub answer for "
                f"{contact.silent_for():.0f}s): {last_exc}") from last_exc
        logger.warning(
            "%s retrying (attempt %d, %s; no definite answer for %.0fs of %.0fs)",
            what, attempt,
            (f"status={last_resp.status_code}" if last_resp is not None
             else f"err={type(last_exc).__name__}"),
            contact.silent_for(), contact.window_s)
        time.sleep(delay + random.uniform(0, delay * 0.1))
        delay = min(delay * 2, _RETRY_MAX_DELAY_S)


def blake3_file(path: Path, *, chunk: int = 8 * 1024 * 1024) -> str:

    h = blake3()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


@dataclass
class CommitFile:
    """One file to add: repo path + local bytes.

    ``local_path=None`` is a *by-reference* add (th#592 download-skip):
    blake3 + size_bytes are pre-known from the bank and the blob must
    already exist in CAS — there are no bytes to upload."""

    path: str
    local_path: Optional[Path] = None
    size_bytes: int = 0
    blake3: str = ""

    def resolve(self) -> "CommitFile":
        if self.local_path is None:
            if not self.blake3:
                raise HubPublishError(
                    f"by-reference commit file {self.path!r} needs a blake3")
            return self
        if not self.size_bytes:
            self.size_bytes = int(Path(self.local_path).stat().st_size)
        if not self.blake3:
            self.blake3 = blake3_file(Path(self.local_path))
        return self


@dataclass
class CommitResult:
    revision_id: str
    uploaded: int
    deduped: int
    total_bytes: int
    # Content-addressed checkpoint id minted at finalize (tensorhub derives it
    # from the snapshot manifest); THE id for tree/lineage queries. The
    # revision_id above is the upload-session id, not queryable post-finalize.
    checkpoint_id: str = ""
    response: dict[str, Any] = field(default_factory=dict)


class HubClient:
    """Thin client over tensorhub's `/commits` API for one destination repo."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        owner: str,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.token = str(token or "").strip()
        self.owner = str(owner or "").strip()
        self.timeout_s = timeout_s
        if not self.base_url or not self.token:
            raise HubPublishError("missing tensorhub base URL or capability token")

    @classmethod
    def from_ctx(cls, ctx: Any) -> "HubClient":
        """Build from a gen_worker RequestContext (cap-token identity)."""
        base = str(getattr(ctx, "_file_api_base_url", "") or "").strip()
        token = str(getattr(ctx, "_worker_capability_token", "") or "").strip()
        owner = str(getattr(ctx, "owner", "") or getattr(ctx, "_owner", "") or "").strip()
        return cls(base_url=base, token=token, owner=owner)

    # ---- internals ----

    def _headers(self) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        if self.owner:
            h["X-Cozy-Owner"] = self.owner
        return h

    def _repo_path(self, destination_repo: str) -> str:
        owner, _, name = str(destination_repo).partition("/")
        if not owner or not name:
            raise HubPublishError(f"destination_repo must be owner/repo, got {destination_repo!r}")
        return (
            f"/api/v1/repos/{urllib.parse.quote(owner, safe='')}/"
            f"{urllib.parse.quote(name, safe='')}"
        )

    def _post(self, path: str, payload: Optional[dict] = None, *, timeout: float | None = None) -> requests.Response:
        return _send_with_retries(f"POST {path}", lambda: _http_session().post(
            f"{self.base_url}{path}",
            headers=self._headers(),
            data=json.dumps(payload) if payload is not None else None,
            timeout=(_CONNECT_TIMEOUT_S, timeout or self.timeout_s),
        ))

    @staticmethod
    def _json(resp: requests.Response) -> dict[str, Any]:
        try:
            out = resp.json() if resp.text else {}
        except Exception:
            out = {}
        return out if isinstance(out, dict) else {}

    def _post_complete(self, complete_path: str, payload: dict) -> requests.Response:
        """POST .../complete with a generous timeout (large single files
        verify synchronously server-side, see _COMPLETE_TIMEOUT_S), then poll
        through a 409 upload_complete_in_progress race instead of treating it
        as fatal: /complete is idempotent once finalized (tensorhub's
        sess.Finalized fast path returns the same success payload), so
        re-POSTing catches up to whatever the in-flight attempt decides.
        Network-severed attempts get the same treatment, bounded by SILENCE
        rather than a clock (gw#666): every 409 in-progress answer is proof
        the hub is up and verifying, so the loop waits as long as the hub
        keeps saying so, and gives up only after _COMPLETE_SILENCE_WINDOW_S
        with no definite answer at all."""
        contact = SilenceWindow(_COMPLETE_SILENCE_WINDOW_S)
        while True:
            try:
                resp = self._post(complete_path, payload, timeout=_COMPLETE_TIMEOUT_S)
            except HubPublishError:
                if contact.stalled():
                    raise
                logger.warning(
                    "POST %s network-severed; re-POSTing (idempotent complete; "
                    "no definite answer for %.0fs of %.0fs)",
                    complete_path, contact.silent_for(), contact.window_s)
                time.sleep(_COMPLETE_NETWORK_RETRY_DELAY_S)
                continue
            if resp.status_code == 409 and _error_code_of(resp) == "upload_complete_in_progress":
                # The hub is alive AND a verify holds the lock: that is
                # progress, not a reason to give up. Wait it out.
                contact.touch()
                time.sleep(_COMPLETE_IN_PROGRESS_POLL_S)
                continue
            if resp.status_code >= 500:
                # gw#565: an edge/tunnel in front of tensorhub (ngrok) times
                # out DURING the synchronous verify and answers 5xx while the
                # server is still working. A returned 5xx is the same case as
                # a severed connection — it says nothing definite, so it does
                # NOT refresh the window; the sess.Finalized fast path answers
                # the catch-up POST if the verify did land.
                if contact.stalled():
                    return resp
                logger.warning(
                    "POST %s returned %d (edge-masked verify?); re-POSTing "
                    "(idempotent complete; no definite answer for %.0fs of %.0fs)",
                    complete_path, resp.status_code,
                    contact.silent_for(), contact.window_s)
                time.sleep(_COMPLETE_NETWORK_RETRY_DELAY_S)
                continue
            return resp

    def _reopen_upload(self, repo_path: str, revision_id: str, path: str) -> dict[str, Any]:
        """Mint a fresh presigned upload for one stashed add whose staged
        bytes were lost (th#699). Returns the same entry shape as the
        create-commit `uploads` array (may be a dedup hit: `exists: true`)."""
        resp = self._post(
            f"{repo_path}/commits/{urllib.parse.quote(revision_id, safe='')}/uploads",
            {"path": path},
        )
        if resp.status_code < 200 or resp.status_code >= 300:
            raise HubPublishError(
                f"upload re-open failed ({resp.status_code}) for {path!r}: {resp.text[:500]}")
        return self._json(resp)

    def _upload_one(self, repo_path: str, revision_id: str, entry: Mapping[str, Any],
                    local_path: Path,
                    part_progress: Optional[Callable[[int, int, int], None]] = None) -> None:
        """Upload one file, surviving server-side staging loss: on
        409 staging_object_missing from /complete, re-open the upload and
        re-send just this file (bounded — the rest of the commit is unaffected)."""
        path = str(entry.get("path") or "")
        for attempt in range(_REUPLOAD_ATTEMPTS + 1):
            try:
                self._upload_entry_once(repo_path, revision_id, entry, local_path,
                                        part_progress=part_progress)
                return
            except _StagingLostError as exc:
                if attempt == _REUPLOAD_ATTEMPTS:
                    raise HubPublishError(
                        f"upload for {path!r} failed: staged bytes lost server-side "
                        f"{attempt + 1} time(s) (last: {exc})") from exc
                logger.warning(
                    "staged bytes for %r lost server-side; re-opening upload "
                    "(re-upload %d/%d)", path, attempt + 1, _REUPLOAD_ATTEMPTS)
                entry = self._reopen_upload(repo_path, revision_id, path)
                if entry.get("exists"):
                    return  # landed in CAS meanwhile — server recorded the dedup

    def _upload_entry_once(self, repo_path: str, revision_id: str, entry: Mapping[str, Any],
                           local_path: Path,
                           part_progress: Optional[Callable[[int, int, int], None]] = None) -> None:
        upload_id = str(entry.get("upload_id") or "").strip()
        if not upload_id:
            raise HubPublishError(f"commit upload entry missing upload_id for {entry.get('path')!r}")
        complete_path = (
            f"{repo_path}/commits/{urllib.parse.quote(revision_id, safe='')}"
            f"/uploads/{urllib.parse.quote(upload_id, safe='')}/complete"
        )

        # SDK transfer-grant path (R2): the server returns a scoped temporary
        # credential instead of presigned multipart part URLs. Upload the
        # object directly with the S3 SDK, then complete with the transfer
        # block (same wire shape gen_worker.presigned_upload uses for media).
        grant_raw = entry.get("transfer_grant")
        if isinstance(grant_raw, Mapping):
            from gen_worker.s3_transfer import S3TransferGrant, upload_file_with_grant

            grant = S3TransferGrant.from_mapping(grant_raw)
            size_bytes = int(entry.get("size_bytes") or local_path.stat().st_size)
            # pgw#738: the SDK lane was upload-progress-BLIND — a multi-GB
            # publish produced zero signals for its whole transfer, so a live
            # publish was indistinguishable from a dead one (and got killed
            # on the dead-signature). Forward boto3's byte callback.
            sdk_progress = None
            if part_progress is not None:
                def sdk_progress(done_flag: int, _total: int, bytes_up: int) -> None:
                    part_progress(int(done_flag), 1, int(bytes_up))
            result = upload_file_with_grant(
                file_path=local_path,
                grant=grant,
                blake3_hex=str(entry.get("blake3") or ""),
                size_bytes=size_bytes,
                on_progress=sdk_progress,
            )
            resp = self._post_complete(complete_path, {"transfer": {
                "mode": "s3_sdk",
                "bucket": result.bucket,
                "key": result.key,
                "size_bytes": result.size_bytes,
                "blake3": result.blake3,
                "etag": result.etag,
            }})
            self._check_complete(resp, str(entry.get("path") or ""))
            if part_progress is not None:
                part_progress(1, 1, int(size_bytes))
            return

        part_urls = list(entry.get("part_urls") or [])
        part_size = int(entry.get("part_size") or 0)
        if not part_urls or part_size <= 0:
            raise HubPublishError(f"commit upload entry missing presign data for {entry.get('path')!r}")
        parts: list[dict[str, Any]] = []
        bytes_up = 0
        with open(local_path, "rb") as f:
            for i, url in enumerate(part_urls):
                buf = f.read(part_size)
                if not buf and i > 0:
                    break
                def _put(u: str = url, b: bytes = buf) -> requests.Response:
                    return _http_session().put(
                        u, data=b, timeout=(_CONNECT_TIMEOUT_S, self.timeout_s * 5))

                resp = _send_with_retries(f"part PUT {entry.get('path')!r} #{i + 1}", _put)
                if resp.status_code == 403:
                    # gw#570: presigned part URLs share the session's fixed
                    # expiry; on a long publish a later file's URLs are stale
                    # before its first byte moves (S3 signals expiry as 403).
                    # Re-open for fresh URLs — bounded by _REUPLOAD_ATTEMPTS,
                    # so a genuine auth failure still fails typed.
                    raise _StagingLostError(
                        f"part PUT for {entry.get('path')!r} part #{i + 1} "
                        f"rejected (403 — presigned URL likely expired)")
                if resp.status_code < 200 or resp.status_code >= 300:
                    raise HubPublishError(
                        f"part PUT failed ({resp.status_code}) for {entry.get('path')!r} "
                        f"part #{i + 1} after retries")
                etag = str(resp.headers.get("ETag") or "").strip().strip('"')
                parts.append({"part_number": i + 1, "etag": etag})
                bytes_up += len(buf)
                if part_progress is not None:
                    part_progress(len(parts), len(part_urls), bytes_up)
        resp = self._post_complete(complete_path, {"parts": parts})
        self._check_complete(resp, str(entry.get("path") or ""))

    @staticmethod
    def _check_complete(resp: requests.Response, path_label: str) -> None:
        if 200 <= resp.status_code < 300:
            return
        if resp.status_code == 409 and _error_code_of(resp) == "staging_object_missing":
            raise _StagingLostError(
                f"staged bytes for {path_label!r} are gone server-side "
                f"(409 staging_object_missing)")
        if resp.status_code == 410 and _error_code_of(resp) == "upload_session_expired":
            # gw#570: commit-create mints every file's session up-front with a
            # fixed expiry; a long publish (slow uplink, many shards) outlives
            # it for later files. The session is unusable but nothing is wrong
            # with the bytes — re-open for a fresh session and re-send just
            # this file, same as staging loss.
            raise _StagingLostError(
                f"upload session for {path_label!r} expired server-side "
                f"(410 upload_session_expired)")
        raise HubPublishError(
            f"upload complete failed ({resp.status_code}) for {path_label!r} "
            f"after retries: {resp.text[:500]}")

    def _finalize(self, repo_path: str, revision_id: str, *, poll_timeout_s: float = 1800.0) -> dict[str, Any]:
        path = f"{repo_path}/commits/{urllib.parse.quote(revision_id, safe='')}/finalize"
        deadline = time.monotonic() + poll_timeout_s
        delay = 2.0
        while True:
            resp = self._post(path)
            if resp.status_code == 202:
                if time.monotonic() > deadline:
                    raise HubPublishError("commit finalize timed out")
                time.sleep(delay)
                delay = min(delay * 1.5, 15.0)
                continue
            if 200 <= resp.status_code < 300:
                return self._json(resp)
            raise HubPublishError(
                f"commit finalize failed ({resp.status_code}): {resp.text[:800]}")

    # ---- public ----

    def commit(
        self,
        *,
        destination_repo: str,
        files: list[CommitFile],
        tags: list[str] | None = None,
        mode: str = "merge",
        flavor: str = "",
        flavors: list[str] | None = None,
        default_flavor: str = "",
        dtype: str = "",
        file_layout: str = "",
        file_type: str = "",
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        repo_spec: Mapping[str, str] | None = None,
        progress: Any = None,
        part_progress: Optional[Callable[[int, int, int], None]] = None,
    ) -> CommitResult:
        """Publish one checkpoint: one POST /commits, PUT the parts, finalize.

        ``files`` are hashed locally (blake3); server-side dedup skips PUTs
        for blobs tensorhub already has. ``mode="replace"`` publishes exactly
        this file set; ``"merge"`` (default) unions with the prior :latest.
        """
        if not files:
            raise HubPublishError("commit requires at least one file")
        repo_path = self._repo_path(destination_repo)
        resolved = [f.resolve() for f in files]

        body: dict[str, Any] = {
            "operations": [
                {"type": "add", "path": f.path, "blake3": f.blake3, "size_bytes": f.size_bytes}
                for f in resolved
            ],
            "mode": mode,
        }
        # Wire-boundary token hygiene (gw#488): tensorhub derives a flavor
        # row from the DTYPE (derivePublishFlavors) and validates every
        # token against [a-z0-9][a-z0-9._-]{0,63} — the internal dtype-axis
        # colon forms ("gguf:q4_k_m", "int8:awq") publish as "-" forms.
        # ONE implementation: gen_worker.models.refs.flavor_token (gw#492).

        if tags:
            df = _token(default_flavor)
            body["tags"] = [
                {"tag": t, **({"default_flavor": df} if df else {})}
                for t in tags
            ]
        for key, val in (
            ("message", message), ("flavor", _token(flavor)),
            ("default_flavor", _token(default_flavor)),
            ("dtype", _token(dtype)), ("file_layout", file_layout),
            ("file_type", file_type),
        ):
            if val:
                body[key] = val
        if flavors:
            body["flavors"] = [_token(f) for f in flavors]
        if metadata:
            body["metadata"] = dict(metadata)
        if provenance:
            # th#606: WORKER-ADDABLE stamp fields only (step_number,
            # epoch_number, quantization_method, quantization_library,
            # upstream_revision). Parents / derivation_op / upstream_ref are
            # orchestrator-derived (signed into the capability token) — the
            # server 400s any attempt to send them from here.
            body["provenance"] = {k: v for k, v in dict(provenance).items() if v}
        for key in ("kind", "library_name", "model_family", "class_name",
                    "adapter_for_family"):
            val = str((repo_spec or {}).get(key) or "").strip()
            if val:
                body[key] = val

        resp = self._post(f"{repo_path}/commits", body)
        if resp.status_code < 200 or resp.status_code >= 300:
            raise HubPublishError(
                f"commit create failed ({resp.status_code}): {resp.text[:800]}")
        created = self._json(resp)
        revision_id = str(created.get("revision_id") or "").strip()
        if not revision_id:
            raise HubPublishError("commit response missing revision_id")

        by_path = {f.path: f for f in resolved}
        uploaded = 0
        deduped = 0
        total = len(resolved)
        try:
            for entry in list(created.get("uploads") or []):
                if not isinstance(entry, dict):
                    continue
                if entry.get("exists"):
                    deduped += 1
                    continue
                f = by_path.get(str(entry.get("path") or ""))
                if f is None:
                    raise HubPublishError(f"server returned unknown upload path {entry.get('path')!r}")
                if f.local_path is None:
                    raise BankedBlobGoneError(
                        f"banked blob for {f.path!r} is gone from CAS "
                        f"(blake3 {f.blake3[:12]}…) — no local bytes to upload")
                self._upload_one(repo_path, revision_id, entry, Path(f.local_path),
                                 part_progress=part_progress)
                uploaded += 1
                if callable(progress):
                    progress(uploaded + deduped, total)
        except Exception:
            # Abort the revision so tensorhub can GC the staging bytes.
            try:
                _http_session().delete(
                    f"{self.base_url}{repo_path}/commits/{urllib.parse.quote(revision_id, safe='')}",
                    headers=self._headers(), timeout=30,
                )
            except Exception:
                pass
            raise

        final = self._finalize(repo_path, revision_id)
        # tensorhub nests the minted id under `checkpoint.checkpoint_id`
        # (repo_publish.go).
        ckpt = final.get("checkpoint") if isinstance(final.get("checkpoint"), dict) else {}
        return CommitResult(
            revision_id=revision_id,
            uploaded=uploaded,
            deduped=deduped,
            total_bytes=sum(f.size_bytes for f in resolved),
            checkpoint_id=str((ckpt or {}).get("checkpoint_id") or "").strip(),
            response=final,
        )

    # ---- v2: chunked sha256 CAS (th#1303 / pgw#781) ----

    @staticmethod
    def _v2_failure(resp: requests.Response) -> Optional[dict[str, Any]]:
        """The v2 routes' TYPED failure, or None.

        A v2 completion does not answer with an error envelope: it answers with
        the th#1301 PROJECTION, and a refusal is
        ``{"status": {"stage": "repudiated", "terminal": true,
                      "failure": {"code", "retryable", "message"}}}``.
        That `retryable` bit is the hub's own classification — the whole point
        of th#1301's retryable-vs-terminal split — so it is read, never guessed
        at from the body's shape.
        """
        try:
            body = resp.json() if resp.text else {}
        except Exception:  # noqa: BLE001 - an unparseable body is not a verdict
            return None
        if not isinstance(body, dict):
            return None
        status = body.get("status")
        if not isinstance(status, dict):
            return None
        failure = status.get("failure")
        if not isinstance(failure, dict) or not failure.get("code"):
            return None
        return failure

    def _post_v2_complete(self, path: str) -> requests.Response:
        """POST /complete WITHOUT the envelope-guessing retry loop.

        `_send_with_retries` decides "did we hear from the hub?" from the body's
        SHAPE, and a v2 completion's refusal is a projection rather than an
        error envelope — so it read a real, typed, `retryable: false` refusal as
        a proxy non-answer and retried it. MEASURED LIVE (twice): the retry
        found the session already terminal and returned 409
        `publish_repudiated`, so the caller was told the publish had been
        repudiated instead of `invalid_manifest_for_kind:
        missing_diffusers_single_file_safetensors` — a consequence in place of a
        cause, and the actual reason was gone.

        A publish is not idempotent to re-complete, so a blind retry here can
        only ever destroy the diagnosis. Network-level failures still retry
        (nothing was heard); an HTTP answer of any status is the hub speaking
        and is returned as-is for the typed handling below.
        """
        return _send_with_retries(
            f"POST {path}",
            lambda: _http_session().post(
                f"{self.base_url}{path}", headers=self._headers(),
                timeout=(_CONNECT_TIMEOUT_S, _COMPLETE_TIMEOUT_S),
            ),
            definite=lambda resp: True,
        )

    def publish_v2(
        self,
        *,
        destination_repo: str,
        files: list[CommitFile],
        tags: list[str] | None = None,
        mode: str = "merge",
        flavor: str = "",
        flavors: list[str] | None = None,
        default_flavor: str = "",
        dtype: str = "",
        file_layout: str = "",
        file_type: str = "",
        display_label: str = "",
        objective: str = "",
        distilled: Optional[bool] = None,
        required_paths: list[str] | None = None,
        declared_tensors: Mapping[str, Any] | None = None,
        deletions: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        repo_spec: Mapping[str, str] | None = None,
        progress: Any = None,
        part_progress: Optional[Callable[[int, int, int], None]] = None,
    ) -> CommitResult:
        """Publish one checkpoint over the CHUNKED SHA-256 CAS (th#1303).

        Declare -> `{have, need}` -> PUT the needed objects -> complete. What
        makes this different from :meth:`commit` is not the transport but WHO
        GUARANTEES INTEGRITY: each object's sha256 is signed into its presigned
        PUT, so R2 refuses bytes that do not hash to the key (400, and the
        object does not exist afterwards) and refuses a substituted claim (403).
        A claimed digest stops being assertable, which is what kills th#1305's
        inherit/overwrite class structurally rather than by guard.

        Files > 64 MiB become an ordered list of enforced chunks, so a retry
        costs one chunk instead of a 2 GB shard and progress is monotone across
        retries — the pgw#786 lemon-host pathology, fixed by the unit of work
        rather than by a new watchdog.

        THERE IS NO PROTOCOL AUTO-SELECT AND NO ENV KNOB. The caller names the
        protocol, so flipping a producer class to sha256 is a code change and a
        deploy — which is exactly the controlled, blast-radius-ordered producer
        flip th#1303 phase 3 asks for. A hub without these routes answers 404,
        and this deliberately does NOT fall back: a silent downgrade to blake3
        would make "did this publish v2?" unanswerable from the outside (and a
        404 from a PROXY is not a 404 from the hub — th#715).

        ``provenance`` carries the WORKER-ADDABLE stamp subset only, exactly as
        :meth:`commit` does — the hub resolves the authoritative stamp from the
        capability token (th#606/th#1331), so lineage cannot be forged from here
        and is identical whichever protocol published it.
        """
        from ..models import chunk_upload as _cu
        from ..models.chunk_upload import UploadGrant

        # Read the constant off the module at CALL time rather than binding a
        # default: it must equal the hub's `storage.CASChunkSizeBytes`, and a
        # call-time read keeps that single source of truth substitutable in
        # tests without exposing a tuning knob on this method.
        chunk_size = _cu.CAS_CHUNK_SIZE_BYTES

        if not files:
            raise HubPublishError("publish_v2 requires at least one file")

        repo_path = self._repo_path(destination_repo)

        # ONE streaming pass per file yields the whole-file sha256 AND every
        # per-chunk sha256. A by-reference add has no local bytes and therefore
        # cannot be declared under a protocol whose whole point is that the
        # digest is proven from the bytes.
        decls = []
        sources: dict[str, tuple[Path, int, int]] = {}
        for f in files:
            if f.local_path is None:
                raise HubPublishError(
                    f"publish_v2: {f.path!r} is a by-reference add; v2 declares "
                    "digests computed from local bytes (use commit() or supply bytes)"
                )
            local = Path(f.local_path)
            decl = _cu.hash_file_and_chunks(
                local, chunk_size=chunk_size, rel_path=f.path)
            decls.append(decl)
            if decl.chunks:
                for c in decl.chunks:
                    sources["sha256:" + c.sha256] = (local, c.offset, c.length)
            else:
                sources[decl.digest] = (local, 0, decl.size_bytes)

        body: dict[str, Any] = {
            "mode": mode,
            "files": [d.to_wire() for d in decls],
        }
        if tags:
            df = _token(default_flavor)
            body["tags"] = [
                {"tag": t, **({"default_flavor": df} if df else {})} for t in tags
            ]
        for key, val in (
            ("flavor", _token(flavor)), ("default_flavor", _token(default_flavor)),
            ("dtype", _token(dtype)), ("file_layout", file_layout),
            ("file_type", file_type), ("display_label", display_label),
            ("objective", objective),
        ):
            if val:
                body[key] = val
        if flavors:
            body["flavors"] = [_token(f) for f in flavors]
        if distilled is not None:
            body["distilled"] = bool(distilled)
        if required_paths:
            body["required_paths"] = list(required_paths)
        if declared_tensors:
            body["declared_tensors"] = dict(declared_tensors)
        if deletions:
            body["deletions"] = list(deletions)
        if metadata:
            body["metadata"] = dict(metadata)
        if provenance:
            # th#1331: the v2 declare decodes this STRICTLY into the same
            # `commitProvenanceInput` v1 uses, so the accepted subset is exactly
            # `step_number | epoch_number | quantization_method |
            # quantization_library | upstream_revision | upstream_attestation`.
            # `parents` / `derivation_op` / `upstream_ref` / `job_id` are
            # orchestrator-derived (signed into the capability token) and are a
            # 400 that NAMES the field — deliberately, because a silent drop
            # would leave the worker believing it had stamped lineage it had
            # not. The merged stamp is captured at DECLARE, where the token is
            # presented, and replayed through `resolvePublishProvenance` at
            # completion, so v1 and v2 mirrors record identical lineage.
            body["provenance"] = {k: v for k, v in dict(provenance).items() if v}
        for key in ("kind", "library_name", "model_family", "class_name",
                    "adapter_for_family"):
            val = str((repo_spec or {}).get(key) or "").strip()
            if val:
                body[key] = val

        resp = self._post(f"{repo_path}/publishes", body)
        if resp.status_code < 200 or resp.status_code >= 300:
            raise HubPublishError(
                f"publish declare failed ({resp.status_code}): {resp.text[:800]}")
        session = self._json(resp)
        publish_id = str(session.get("publish_id") or "").strip()
        if not publish_id:
            raise HubPublishError("publish response missing publish_id")

        distinct = int(session.get("distinct_objects") or 0)
        resident = int(session.get("resident_objects") or 0)
        uploaded_objects = 0
        total_bytes = sum(d.size_bytes for d in decls)

        def _grants_of(payload: Mapping[str, Any]) -> list["UploadGrant"]:
            out = []
            for g in payload.get("need") or []:
                if not isinstance(g, dict):
                    continue
                out.append(UploadGrant(
                    digest=str(g.get("digest") or "").strip().lower(),
                    size_bytes=int(g.get("size_bytes") or 0),
                    put_url=str(g.get("put_url") or ""),
                    headers={str(k): str(v) for k, v in (g.get("headers") or {}).items()},
                    staging_key=str(g.get("staging_key") or ""),
                ))
            return out

        def _source_for(digest: str) -> tuple[Path, int, int]:
            span = sources.get(digest.strip().lower())
            if span is None:
                raise ValueError(
                    f"hub granted {digest[:20]}… which this publish never declared"
                )
            return span

        try:
            grants = _grants_of(session)
            for attempt in range(_REUPLOAD_ATTEMPTS + 1):
                if not grants:
                    break
                report = _cu.upload_grants(
                    grants, _source_for,
                    on_bytes=(lambda n: part_progress(0, 0, n))
                    if callable(part_progress) else None,
                )
                uploaded_objects += report.uploaded
                if callable(progress):
                    progress(resident + uploaded_objects, distinct or len(grants))
                if report.ok:
                    break
                if attempt == _REUPLOAD_ATTEMPTS:
                    raise HubPublishError(
                        f"publish {publish_id}: {len(report.failures)} object(s) failed to "
                        f"upload after {_REUPLOAD_ATTEMPTS + 1} passes: "
                        + "; ".join(report.failures[:5])
                    )
                # RESUME NEEDS NO CLIENT STATE: re-plan and the need set comes
                # back smaller, because what landed is now resident. A kill
                # mid-upload costs the in-flight objects and nothing else.
                again = self._post(f"{repo_path}/publishes/"
                                   f"{urllib.parse.quote(publish_id, safe='')}/grants")
                if again.status_code < 200 or again.status_code >= 300:
                    raise HubPublishError(
                        f"publish re-plan failed ({again.status_code}): {again.text[:500]}")
                grants = _grants_of(self._json(again))

            done = self._post_v2_complete(
                f"{repo_path}/publishes/{urllib.parse.quote(publish_id, safe='')}/complete")
        except Exception:
            # Abort so the staging prefix is reclaimed and the session lands in
            # a TERMINAL state rather than looking forever in-flight.
            try:
                _http_session().delete(
                    f"{self.base_url}{repo_path}/publishes/"
                    f"{urllib.parse.quote(publish_id, safe='')}",
                    headers=self._headers(), timeout=30,
                )
            except Exception:
                pass
            raise

        if done.status_code < 200 or done.status_code >= 300:
            # Lead with the hub's OWN typed classification when it gave one. A
            # refusal that reports its `code`, its `retryable` bit and the stage
            # that produced it is actionable; a bare status code plus 800 bytes
            # of truncated projection is what sent this lane looking in the
            # wrong place twice.
            failure = self._v2_failure(done)
            if failure:
                stage = ""
                try:
                    for s in (done.json().get("status") or {}).get("stages") or []:
                        if s.get("status") == "failed":
                            stage = str(s.get("stage") or "")
                except Exception:  # noqa: BLE001 - the stage is a nicety
                    pass
                raise HubPublishError(
                    f"publish {publish_id} {'repudiated' if not failure.get('retryable') else 'failed'}"
                    f"{f' at {stage}' if stage else ''}: "
                    f"{failure.get('code')}: {failure.get('message')} "
                    f"(retryable={bool(failure.get('retryable'))})"
                )
            raise HubPublishError(
                f"publish complete failed ({done.status_code}): {done.text[:800]}")
        final = self._json(done)
        ckpt = final.get("checkpoint") if isinstance(final.get("checkpoint"), dict) else {}
        checkpoint_id = str((ckpt or {}).get("checkpoint_id") or "").strip()
        if not checkpoint_id:
            raise HubPublishError(
                f"publish {publish_id} completed without a checkpoint id: "
                f"{json.dumps(final)[:500]}")
        # Surfaced, not swallowed: the hub names which canonical th#1301 checks
        # it did NOT run. A list that promises 19 and silently runs 14 is worse
        # than one promising 14.
        skipped = final.get("checks_unavailable") or []
        if skipped:
            logger.info("publish %s checks_unavailable=%s", publish_id, skipped)
        logger.info(
            "publish_v2 done repo=%s publish=%s objects=%d resident=%d uploaded=%d "
            "checkpoint=%s",
            destination_repo, publish_id, distinct, resident, uploaded_objects,
            checkpoint_id[:24],
        )
        return CommitResult(
            revision_id=publish_id,
            uploaded=uploaded_objects,
            deduped=resident,
            total_bytes=total_bytes,
            checkpoint_id=checkpoint_id,
            response=final,
        )

    def lookup_clone_manifests(
        self, destination_repo: str, keys: list[str],
    ) -> dict[str, dict[str, Any]]:
        """th#592 download-skip: batch bank lookup. Returns key -> result
        ({found, ready, payload?}); raises HubPublishError on failure —
        callers treat that as a miss (fail-open)."""
        if not keys:
            return {}
        repo_path = self._repo_path(destination_repo)
        resp = self._post(f"{repo_path}/clone-manifests/lookup", {"keys": list(keys)})
        if resp.status_code < 200 or resp.status_code >= 300:
            raise HubPublishError(
                f"clone-manifest lookup failed ({resp.status_code}): {resp.text[:300]}")
        out: dict[str, dict[str, Any]] = {}
        for r in self._json(resp).get("results") or []:
            if isinstance(r, dict) and r.get("key"):
                out[str(r["key"])] = r
        return out

    def record_clone_manifests(
        self, destination_repo: str, manifests: list[dict[str, Any]],
    ) -> dict[str, str]:
        """th#592 download-skip: record published manifests under their bank
        keys ([{key, payload}]). Returns key -> status."""
        if not manifests:
            return {}
        repo_path = self._repo_path(destination_repo)
        resp = self._post(f"{repo_path}/clone-manifests", {"manifests": list(manifests)})
        if resp.status_code < 200 or resp.status_code >= 300:
            raise HubPublishError(
                f"clone-manifest record failed ({resp.status_code}): {resp.text[:300]}")
        return {
            str(r.get("key")): str(r.get("status") or "")
            for r in self._json(resp).get("results") or []
            if isinstance(r, dict)
        }


_DATASET_PAGE_LIMIT = 200  # tensorhub's cap; a larger value is clamped there.
_DATASET_PAGE_CEILING = 1000  # pages, i.e. 200k datasets — a runaway guard.


def _find_dataset_id(
    base: str, headers: dict[str, str], owner: str, name: str,
) -> str:
    """dataset_id of ``owner/name``, or "" when it genuinely does not exist.

    Raises ``AuthError``/``RuntimeError`` rather than returning "" on any
    response it cannot READ — the caller creates on "", so a swallowed error
    is a duplicate dataset (pgw#656).
    """
    wanted = name.lower()
    cursor = ""
    seen_pages = 0
    while True:
        query = {"org": owner, "limit": str(_DATASET_PAGE_LIMIT)}
        if cursor:
            query["cursor"] = cursor
        resp = requests.get(
            f"{base}/api/v1/datasets?{urllib.parse.urlencode(query)}",
            headers=headers,
            timeout=30,
        )
        if resp.status_code in (401, 403):
            raise AuthError(f"dataset list unauthorized ({resp.status_code})")
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(
                f"dataset list failed ({resp.status_code}): {resp.text[:256]}"
            )
        try:
            body = resp.json() if resp.text else {}
            items = list(body.get("items") or [])
        except (ValueError, AttributeError) as exc:
            raise RuntimeError(
                f"dataset list returned unreadable JSON: {exc}"
            ) from exc
        for it in items:
            if str(it.get("name") or "").lower() == wanted:
                return str(it.get("dataset_id") or "")
        next_cursor = str(body.get("next_cursor") or "").strip()
        seen_pages += 1
        # No cursor, no movement, or a server that ignores paging: stop. Never
        # spin — but never silently truncate the search either.
        if not next_cursor or next_cursor == cursor or not items:
            return ""
        if seen_pages >= _DATASET_PAGE_CEILING:
            raise RuntimeError(
                f"dataset list did not terminate after {seen_pages} pages "
                f"(org={owner}); refusing to create a possible duplicate"
            )
        cursor = next_cursor


def publish_dataset_revision(
    *,
    base_url: str,
    token: str,
    destination_dataset: str,
    features_json: dict[str, Any],
    row_artifacts_json: Optional[dict[str, Any]] = None,
    snapshot_manifest: Optional[list[dict[str, Any]]] = None,
    visibility: str = "private",
    kind: str = "",
    dataset_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Publish a dataset revision into ``tensorhub.datasets``.

    Hub-API plumbing for ``DatasetContext.publish_dataset_revision`` (which
    documents the tenant-facing contract). The flow:

    1. Resolve ``destination_dataset`` (owner/name) against tensorhub — the
       FULL cursor-paginated listing, and any unreadable response raises
       rather than reading as "no such dataset" (pgw#656: every silent miss
       here becomes a duplicate dataset in step 2).
    2. If the dataset row doesn't exist: ``POST /api/v1/datasets`` with
       ``{tenant, name, visibility, schema: features_json}``.
    3. Otherwise: ``PATCH /api/v1/datasets/:id`` to update the schema
       (+ row_artifacts / visibility).

    ``kind`` / ``dataset_info`` / ``snapshot_manifest`` ride inside
    ``features_json`` under reserved ``__cozy_*__`` keys until tensorhub
    grows dedicated columns (th#1162). Raises ``AuthError`` on 401/403 and
    ``RuntimeError`` on any other HTTP failure.
    """

    owner, name = _parse_owner_repo(destination_dataset)
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("publish_dataset_revision: no file_api_base_url")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Cozy-Owner": owner,
    }

    # Stash the kind (+ dataset_info if provided) inside features_json
    # using a reserved `__cozy_*__` key so it survives through the
    # server's features_json passthrough. Once tensorhub adds a
    # dedicated `kind` column, migrate these to top-level fields.
    raw_schema = dict(features_json or {})
    if isinstance(raw_schema.get("features"), dict):
        features_payload = dict(raw_schema)
    else:
        features_payload = {"features": raw_schema}
    if kind:
        features_payload["__cozy_kind__"] = kind
    if dataset_info:
        features_payload["__cozy_dataset_info__"] = dataset_info
    if snapshot_manifest:
        features_payload["__cozy_snapshot_manifest__"] = snapshot_manifest

    # Step 1: look up any existing dataset by (owner, name).
    #
    # pgw#656: every way this lookup can fail to SEE an existing row ends in
    # step 2a — CREATE — i.e. a duplicate dataset, silently. So it fails loud
    # instead of falling through, and it walks the whole listing:
    #   * a non-2xx / unparseable response now raises (it used to be
    #     indistinguishable from "no such dataset");
    #   * the listing is CURSOR-paginated (tensorhub caps `limit` at 200 and
    #     returns `next_cursor`), where it used to read one default page of 50
    #     and call that the org's whole dataset set;
    #   * the filter is `?org=`, which is the parameter the handler actually
    #     reads — `?tenant=` was silently ignored and the result only
    #     happened to be right because the token's own org was the default.
    existing_id = _find_dataset_id(base, headers, owner, name)

    if not existing_id:
        # Step 2a: create.
        create_body = {
            "tenant": owner,
            "name": name,
            "visibility": visibility,
            "schema": features_payload,
        }
        resp = requests.post(
            f"{base}/api/v1/datasets",
            headers=headers,
            data=json.dumps(create_body).encode("utf-8"),
            timeout=30,
        )
        if resp.status_code in (401, 403):
            raise AuthError(f"dataset create unauthorized ({resp.status_code})")
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(
                f"dataset create failed ({resp.status_code}): {resp.text[:256]}"
            )
        data = resp.json() if resp.text else {}
        return {
            "ok": True,
            "dataset_id": str(data.get("dataset_id") or ""),
            "owner": owner,
            "name": name,
            "existed": False,
        }

    # Step 2b: update via PATCH.
    patch_url = f"{base}/api/v1/datasets/{urllib.parse.quote(existing_id, safe='')}"
    patch_body: dict[str, Any] = {"schema": features_payload}
    if row_artifacts_json is not None:
        patch_body["row_artifacts"] = row_artifacts_json
    if visibility in ("private", "public"):
        patch_body["visibility"] = visibility
    resp = requests.patch(
        patch_url,
        headers=headers,
        data=json.dumps(patch_body).encode("utf-8"),
        timeout=30,
    )
    if resp.status_code in (401, 403):
        raise AuthError(f"dataset patch unauthorized ({resp.status_code})")
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(
            f"dataset patch failed ({resp.status_code}): {resp.text[:256]}"
        )
    return {
        "ok": True,
        "dataset_id": existing_id,
        "owner": owner,
        "name": name,
        "existed": True,
    }


def files_from_tree(tree: Path, *, prefix: str = "") -> list[CommitFile]:
    """Build CommitFile entries for every regular file under ``tree``.

    ``.cache/huggingface/**`` is skipped: huggingface_hub's local-dir download
    metadata is cache-layout junk, never repo content."""
    tree = Path(tree)
    out: list[CommitFile] = []
    for f in sorted(tree.rglob("*")):
        if not f.is_file():
            continue
        rel_parts = f.relative_to(tree).parts
        if rel_parts[:2] == (".cache", "huggingface"):
            continue
        rel = f.relative_to(tree).as_posix()
        if prefix:
            rel = f"{prefix.rstrip('/')}/{rel}"
        out.append(CommitFile(path=rel, local_path=f))
    return out


__all__ = [
    "BankedBlobGoneError",
    "HubClient",
    "HubPublishError",
    "CommitFile",
    "CommitResult",
    "blake3_file",
    "files_from_tree",
    "publish_dataset_revision",
]
