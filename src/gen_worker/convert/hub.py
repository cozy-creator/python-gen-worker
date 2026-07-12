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

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 5
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
_COMPLETE_NETWORK_MAX_WAIT_S = 1800.0

_SESSION: Optional[requests.Session] = None


def _http_session() -> requests.Session:
    """Shared session with TCP keepalives so NAT/conntrack middleboxes don't
    evict the idle flow while the server verifies (no response bytes for
    minutes)."""
    import socket

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


def _send_with_retries(what: str, send: Callable[[], requests.Response]) -> requests.Response:
    """Bounded retries on network errors, 429, and 5xx (honors Retry-After).

    Returns the last response for non-retryable statuses — callers keep their
    own status handling.
    """
    delay = _RETRY_BASE_DELAY_S
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            resp = send()
        except requests.RequestException as exc:
            if attempt == _RETRY_ATTEMPTS:
                raise HubPublishError(f"{what} failed (network): {exc}") from exc
        else:
            code = int(resp.status_code)
            if code != 429 and code < 500:
                return resp
            if attempt == _RETRY_ATTEMPTS:
                return resp
            delay = _retry_after_s(resp) or delay
        logger.warning("%s retrying (attempt %d/%d)", what, attempt, _RETRY_ATTEMPTS)
        time.sleep(delay + random.uniform(0, delay * 0.1))
        delay = min(delay * 2, _RETRY_MAX_DELAY_S)
    raise HubPublishError(f"{what} failed after {_RETRY_ATTEMPTS} attempts")


def blake3_file(path: Path, *, chunk: int = 8 * 1024 * 1024) -> str:
    from blake3 import blake3

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
        Network-severed attempts get the same treatment on a longer clock
        (_COMPLETE_NETWORK_MAX_WAIT_S): each re-POST may re-run a full
        multi-minute verify, so give it room instead of failing the commit."""
        deadline = time.monotonic() + _COMPLETE_NETWORK_MAX_WAIT_S
        while True:
            try:
                resp = self._post(complete_path, payload, timeout=_COMPLETE_TIMEOUT_S)
            except HubPublishError:
                if time.monotonic() >= deadline:
                    raise
                logger.warning("POST %s network-severed; re-POSTing (idempotent complete)", complete_path)
                time.sleep(_COMPLETE_NETWORK_RETRY_DELAY_S)
                continue
            if resp.status_code == 409 and _error_code_of(resp) == "upload_complete_in_progress":
                if time.monotonic() >= deadline:
                    return resp
                time.sleep(_COMPLETE_IN_PROGRESS_POLL_S)
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
            result = upload_file_with_grant(
                file_path=local_path,
                grant=grant,
                blake3_hex=str(entry.get("blake3") or ""),
                size_bytes=size_bytes,
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
                if resp.status_code < 200 or resp.status_code >= 300:
                    raise HubPublishError(
                        f"part PUT failed ({resp.status_code}) for {entry.get('path')!r} "
                        f"part #{i + 1} after {_RETRY_ATTEMPTS} attempts")
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
        raise HubPublishError(
            f"upload complete failed ({resp.status_code}) for {path_label!r} "
            f"after {_RETRY_ATTEMPTS} attempts: {resp.text[:500]}")

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
        from gen_worker.models.refs import flavor_token as _token

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
]
