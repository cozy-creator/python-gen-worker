"""Upload-session manager for python-gen-worker (issue #20).

Tenant code never sees sessions directly. The manager is attached to ctx
at request dispatch; it lazily opens sessions on first use, reuses them
across subsequent save_* calls to the same destination, and auto-finalizes
(or auto-aborts on exception) at request end.

Library-internal. `_`-prefixed module name. Don't import from tenant code.

# Session lifecycle

    ctx._upload_sessions.open(kind='checkpoint', scope={'repo_id': ...})
        → opens a session on tensorhub via POST {prefix}/upload-sessions
        → caches session_id keyed by (kind, scope)
        → returns session_id

    ctx._upload_sessions.session_id_for(kind, scope)
        → returns cached session_id; opens if not cached

    ctx._upload_sessions.finalize(session_id, finalize_body)
        → POST {prefix}/upload-sessions/:session_id/finalize
        → returns response body

    ctx._upload_sessions.abort(session_id)
        → DELETE {prefix}/upload-sessions/:session_id
        → no-op if already finalized

    ctx._upload_sessions.close_all(abort_open=True)
        → called at request end; aborts any still-open sessions.

# URL routing

Each session kind maps to a URL prefix:
  - checkpoint        → /api/v1/repos/{owner}/{repo}/upload-sessions[/...]
  - dataset           → /api/v1/datasets/{dataset_id}/upload-sessions[/...]
  - endpoint_release  → /api/v1/endpoints/{owner}/{endpoint_name}/upload-sessions[/...]
  - media             → /api/v1/media/upload-sessions[/...]

Only `checkpoint` is implemented end-to-end today (issue #20 scope); the
other three are scaffolding for future work.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional, Tuple

import requests

from ..api.errors import AuthError

logger = logging.getLogger(__name__)

SessionKind = str  # 'checkpoint' | 'dataset' | 'endpoint_release' | 'media'

_SESSION_CREATE_TIMEOUT_S = 30
_SESSION_FINALIZE_TIMEOUT_S = 120
_SESSION_ABORT_TIMEOUT_S = 15


def _session_url_prefix(kind: SessionKind, scope: Dict[str, Any]) -> str:
    """Compute the resource-prefix URL segment for a session of this kind."""
    if kind == "checkpoint":
        # checkpoint scope carries {repo_owner, repo_name} — owner slug
        # and repo name, NOT the repo_id UUID (URLs use human slugs).
        owner = urllib.parse.quote(str(scope["repo_owner"]), safe="")
        repo = urllib.parse.quote(str(scope["repo_name"]), safe="")
        return f"/api/v1/repos/{owner}/{repo}"
    if kind == "dataset":
        did = urllib.parse.quote(str(scope["dataset_id"]), safe="")
        return f"/api/v1/datasets/{did}"
    if kind == "endpoint_release":
        owner = urllib.parse.quote(str(scope["endpoint_owner"]), safe="")
        name = urllib.parse.quote(str(scope["endpoint_name"]), safe="")
        return f"/api/v1/endpoints/{owner}/{name}"
    if kind == "media":
        return "/api/v1/media"
    raise ValueError(f"unsupported session kind: {kind!r}")


class _UploadSession:
    """A single open upload session. Caches session_id + URL prefix."""

    __slots__ = ("kind", "scope", "scope_key", "session_id", "url_prefix", "expires_at")

    def __init__(
        self,
        kind: SessionKind,
        scope: Dict[str, Any],
        scope_key: Tuple,
        session_id: str,
        url_prefix: str,
        expires_at: str,
    ) -> None:
        self.kind = kind
        self.scope = scope
        self.scope_key = scope_key
        self.session_id = session_id
        self.url_prefix = url_prefix
        self.expires_at = expires_at

    def upload_url(self) -> str:
        """URL for presigning a new file upload within this session."""
        return f"{self.url_prefix}/upload-sessions/{self.session_id}/uploads"

    def finalize_url(self) -> str:
        """URL for finalizing this session → creates catalog record."""
        return f"{self.url_prefix}/upload-sessions/{self.session_id}/finalize"

    def abort_url(self) -> str:
        """URL for aborting this session."""
        return f"{self.url_prefix}/upload-sessions/{self.session_id}"


class _UploadSessionManager:
    """Request-scoped session manager. Attach to ctx at dispatch.

    - Sessions are cached by `(kind, frozenset(scope.items()))`.
    - Lazy-open: first `session_id_for(...)` opens; subsequent reuse.
    - Thread-safe — sessions can be opened concurrently from multiple
      worker threads if the request fans out, but we ensure one session
      per (kind, scope) via a lock.
    """

    def __init__(
        self,
        *,
        base_url: str,
        headers_provider: Callable[[], Dict[str, str]],
        job_id: Optional[str] = None,
        repo_spec_provider: Optional[Callable[[], Dict[str, str]]] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers_provider = headers_provider
        self._job_id = (job_id or "").strip() or None
        # Pulls the current ctx's repo spec at session-open time. Threaded into
        # the request body so tensorhub can create/update the destination repo
        # without legacy library inference.
        self._repo_spec_provider = repo_spec_provider
        self._lock = threading.Lock()
        self._sessions: Dict[Tuple, _UploadSession] = {}

    def _scope_key(self, kind: SessionKind, scope: Dict[str, Any]) -> Tuple:
        # Stable key: kind + sorted scope items.
        return (kind,) + tuple(sorted((str(k), str(v)) for k, v in scope.items()))

    def session_id_for(self, kind: SessionKind, scope: Dict[str, Any]) -> str:
        """Return cached session_id for (kind, scope); open a new one if not cached."""
        key = self._scope_key(kind, scope)
        with self._lock:
            existing = self._sessions.get(key)
            if existing is not None:
                return existing.session_id
            sess = self._open_session(kind, scope, key)
            self._sessions[key] = sess
            return sess.session_id

    def get(self, kind: SessionKind, scope: Dict[str, Any]) -> Optional[_UploadSession]:
        key = self._scope_key(kind, scope)
        with self._lock:
            return self._sessions.get(key)

    def session(self, kind: SessionKind, scope: Dict[str, Any]) -> _UploadSession:
        """Return the _UploadSession handle (opening if needed). Exposed so
        callers can access the URL builders (upload_url, finalize_url)."""
        _ = self.session_id_for(kind, scope)
        key = self._scope_key(kind, scope)
        with self._lock:
            return self._sessions[key]

    def _open_session(
        self, kind: SessionKind, scope: Dict[str, Any], key: Tuple
    ) -> _UploadSession:
        prefix = _session_url_prefix(kind, scope)
        url = f"{self._base_url}{prefix}/upload-sessions"
        body: Dict[str, Any] = {}
        if self._job_id:
            body["job_id"] = self._job_id
        if self._repo_spec_provider is not None:
            try:
                repo_spec = dict(self._repo_spec_provider() or {})
            except Exception:
                repo_spec = {}
            for key in ("kind", "library_name", "model_family", "class_name", "adapter_for"):
                val = str(repo_spec.get(key) or "").strip()
                if val:
                    body[key] = val
        headers = dict(self._headers_provider())
        headers["Content-Type"] = "application/json"
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=_SESSION_CREATE_TIMEOUT_S)
        except requests.RequestException as e:
            raise RuntimeError(f"upload_session_open failed (network): {e}") from e
        if resp.status_code in (401, 403):
            raise AuthError(f"upload_session_open unauthorized ({resp.status_code})")
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(
                f"upload_session_open failed ({resp.status_code}): {resp.text[:256]}"
            )
        parsed = resp.json() if resp.text else {}
        session_id = str(parsed.get("session_id") or "").strip()
        if not session_id:
            raise RuntimeError("upload_session_open: missing session_id in response")
        expires_at = str(parsed.get("expires_at") or "")
        logger.info(
            "upload_session_opened kind=%s session_id=%s scope=%s",
            kind, session_id, scope,
        )
        return _UploadSession(
            kind=kind,
            scope=dict(scope),
            scope_key=key,
            session_id=session_id,
            url_prefix=prefix,
            expires_at=expires_at,
        )

    def finalize(self, sess: _UploadSession, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST the finalize body; returns the response JSON."""
        url = f"{self._base_url}{sess.finalize_url()}"
        headers = dict(self._headers_provider())
        headers["Content-Type"] = "application/json"
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=_SESSION_FINALIZE_TIMEOUT_S)
        except requests.RequestException as e:
            raise RuntimeError(f"upload_session_finalize failed (network): {e}") from e
        if resp.status_code in (401, 403):
            raise AuthError(f"upload_session_finalize unauthorized ({resp.status_code})")
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(
                f"upload_session_finalize failed ({resp.status_code}): {resp.text[:256]}"
            )
        parsed = resp.json() if resp.text else {}
        # Drop from cache — session is now finalized.
        with self._lock:
            self._sessions.pop(sess.scope_key, None)
        logger.info(
            "upload_session_finalized kind=%s session_id=%s",
            sess.kind, sess.session_id,
        )
        return parsed

    def abort(self, sess: _UploadSession) -> None:
        """Best-effort abort. Logs on failure; doesn't raise."""
        url = f"{self._base_url}{sess.abort_url()}"
        headers = dict(self._headers_provider())
        try:
            requests.delete(url, headers=headers, timeout=_SESSION_ABORT_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "upload_session_abort_failed kind=%s session_id=%s err=%r",
                sess.kind, sess.session_id, e,
            )
        with self._lock:
            self._sessions.pop(sess.scope_key, None)

    def close_all(self, *, abort_open: bool = True) -> None:
        """Called at request end. Aborts any still-open sessions."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        if not abort_open:
            return
        for sess in sessions:
            self.abort(sess)
