"""Tensorhub commit client — the ONE publish path (tensorhub #515).

The write API is HF `create_commit`-shaped:

  POST /api/v1/repos/{tenant}/{name}/commits
      {operations: [{type:"add", path, blake3, size_bytes}, ...],
       tags: [{tag, default_flavor?}], mode: "merge"|"replace",
       flavor/dtype/file_layout/file_type, metadata, lineage, repo spec}
  → {revision_id, uploads: [{path, exists, upload_id, part_urls, part_size,
                             complete_url, ...}]}

Then per non-dedup'd upload: move the bytes (R2 SDK grant or presigned
multipart parts) and POST …/uploads/{id}/complete — both via the shared
per-file engine ``gen_worker.presigned_upload.upload_entry_and_complete``
(one implementation of the e2e#110 409-poll + te#44 J9 network-severed
/complete armor) — and finally POST …/commits/{revision_id}/finalize
(no body; 202 → poll). One commit == one checkpoint == one flavor.
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

from gen_worker.api.errors import ArtifactTransferError, AuthError

# ONE blake3-file implementation library-wide (multithreaded, issue #269) and
# ONE per-file upload engine (grant-or-parts + patient /complete).
from gen_worker.presigned_upload import blake3_hash_file as blake3_file
from gen_worker.presigned_upload import upload_entry_and_complete

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 5
_RETRY_BASE_DELAY_S = 1.0
_RETRY_MAX_DELAY_S = 30.0

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


def _retry_after_s(resp: requests.Response) -> Optional[float]:
    try:
        value = float(str(resp.headers.get("Retry-After") or "").strip())
    except Exception:
        return None
    return min(value, _RETRY_MAX_DELAY_S) if value > 0 else None


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


@dataclass
class CommitFile:
    """One file to add: repo path + local bytes."""

    path: str
    local_path: Path
    size_bytes: int = 0
    blake3: str = ""

    def resolve(self) -> "CommitFile":
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
            timeout=timeout or self.timeout_s,
        ))

    @staticmethod
    def _json(resp: requests.Response) -> dict[str, Any]:
        try:
            out = resp.json() if resp.text else {}
        except Exception:
            out = {}
        return out if isinstance(out, dict) else {}

    def _upload_one(self, repo_path: str, revision_id: str, entry: Mapping[str, Any],
                    local_path: Path) -> None:
        upload_id = str(entry.get("upload_id") or "").strip()
        if not upload_id:
            raise HubPublishError(f"commit upload entry missing upload_id for {entry.get('path')!r}")
        complete_path = (
            f"{repo_path}/commits/{urllib.parse.quote(revision_id, safe='')}"
            f"/uploads/{urllib.parse.quote(upload_id, safe='')}/complete"
        )
        # Byte movement + patient /complete ride the shared per-file engine
        # (grant-or-parts dispatch, e2e#110 409-poll, te#44 J9 network
        # patience, hardened PutPool part transport). The keepalive session
        # protects the long idle server-side verify (NAT/conntrack eviction).
        try:
            upload_entry_and_complete(
                file_path=local_path,
                entry=dict(entry),
                complete_url=f"{self.base_url}{complete_path}",
                headers=self._headers(),
                session=_http_session(),
            )
        except (ArtifactTransferError, AuthError) as exc:
            raise HubPublishError(
                f"upload failed for {entry.get('path')!r}: {exc}") from exc

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
        lineage: list[dict[str, Any]] | None = None,
        repo_spec: Mapping[str, str] | None = None,
        auto_create_external_parent: bool = False,
        progress: Any = None,
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
        if tags:
            body["tags"] = [
                {"tag": t, **({"default_flavor": default_flavor} if default_flavor else {})}
                for t in tags
            ]
        for key, val in (
            ("message", message), ("flavor", flavor), ("default_flavor", default_flavor),
            ("dtype", dtype), ("file_layout", file_layout), ("file_type", file_type),
        ):
            if val:
                body[key] = val
        if flavors:
            body["flavors"] = list(flavors)
        if metadata:
            body["metadata"] = dict(metadata)
        if lineage:
            body["lineage"] = list(lineage)
            body["auto_create_external_parent"] = bool(auto_create_external_parent)
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
                self._upload_one(repo_path, revision_id, entry, Path(f.local_path))
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
        return CommitResult(
            revision_id=revision_id,
            uploaded=uploaded,
            deduped=deduped,
            total_bytes=sum(f.size_bytes for f in resolved),
            checkpoint_id=str(final.get("checkpoint_id") or "").strip(),
            response=final,
        )


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
    "HubClient",
    "HubPublishError",
    "CommitFile",
    "CommitResult",
    "blake3_file",
    "files_from_tree",
]
