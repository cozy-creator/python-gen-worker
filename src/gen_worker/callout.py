"""Endpoint-to-endpoint call-out client — the workflow primitive."""
from __future__ import annotations

import json
import threading
import time
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Dict, List, Optional
from urllib.parse import quote

from .api.errors import (
    CanceledError,
    ChildCallError,
    ChildCallRefusedError,
    ChildCallTimeoutError,
    ChildRequestCanceledError,
    ChildRequestFailedError,
)
from .hub_error import parse_hub_error

_REFUSAL_CODES = frozenset(
    {
        "call_depth_exceeded",
        "call_cycle_detected",
        "tree_budget_exceeded",
        "tier_escalation_denied",
        "parent_not_running",
        "budget_not_root",
    }
)

_NOT_DECLARED_CODES = frozenset({"forbidden", "insufficient_scope", "unauthorized"})

DEFAULT_POLL_INTERVAL_S = 2.0
_HTTP_TIMEOUT_S = 60.0


def semver_major_segment(semver_major: int) -> str:
    if isinstance(semver_major, bool) or not isinstance(semver_major, int):
        raise TypeError(
            "semver_major must be an int naming the endpoint's semver-major "
            f"(e.g. 0 for /v0/), got {semver_major!r}; there is no default"
        )
    if semver_major < 0:
        raise ValueError(f"semver_major must be >= 0, got {semver_major}")
    return f"v{semver_major}"


def _parse_error_body(text: str) -> tuple[str, str]:
    err = parse_hub_error(text)
    return err.code, err.message


class CalloutClient:
    """HTTP client for one invocation's child calls + checkpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        parent_request_id: str,
        get_token: Callable[[], str],
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._parent_request_id = parent_request_id
        self._get_token = get_token
        self._cancel_event = cancel_event

    def _headers(self) -> Dict[str, str]:
        token = self._get_token()
        if not token:
            raise ChildCallRefusedError(
                "child_calls_not_declared",
                "no capability token in this invocation context; declare "
                "@entrypoint(child_calls=True) and run under the platform",
            )
        return {"Authorization": f"Bearer {token}"}

    def _raise_for_error(self, status_code: int, text: str) -> None:
        code, message = _parse_error_body(text)
        if code in _REFUSAL_CODES:
            raise ChildCallRefusedError(code, message)
        if status_code in (401, 403) and (code in _NOT_DECLARED_CODES or not code):
            raise ChildCallRefusedError(
                "child_calls_not_declared",
                message
                or "the platform refused this credential for child calls; "
                "declare @entrypoint(child_calls=True)",
            )
        raise ChildCallError(
            f"child call failed ({status_code}{', ' + code if code else ''}): "
            f"{message or text[:256]}"
        )

    def submit(
        self,
        endpoint: str,
        function: str,
        payload: Dict[str, Any],
        *,
        semver_major: int,
        tier: Optional[str] = None,
    ) -> str:
        """Submit one child request; returns its request id."""
        import requests

        version = semver_major_segment(semver_major)
        endpoint = (endpoint or "").strip().strip("/")
        if "/" not in endpoint:
            raise ValueError(
                f"endpoint must be 'owner/name' (e.g. 'tensorhub/music-analysis'), got {endpoint!r}"
            )
        function = (function or "").strip()
        if not function:
            raise ValueError("function is required")
        body = dict(payload or {})
        if tier:
            body["availability_tier"] = tier
        url = f"{self._base_url}/{endpoint}/{version}/{function}"
        resp = requests.post(url, headers=self._headers(), json=body, timeout=_HTTP_TIMEOUT_S)
        if resp.status_code >= 300:
            self._raise_for_error(resp.status_code, resp.text)
        doc = resp.json() if resp.text else {}
        request_id = str(doc.get("request_id") or "")
        if not request_id:
            raise ChildCallError(f"submit returned no request_id: {resp.text[:256]}")
        return request_id

    def get(self, request_id: str) -> Dict[str, Any]:
        import requests

        resp = requests.get(
            f"{self._base_url}/v1/requests/{quote(request_id, safe='')}",
            headers=self._headers(),
            timeout=_HTTP_TIMEOUT_S,
        )
        if resp.status_code >= 300:
            self._raise_for_error(resp.status_code, resp.text)
        doc = resp.json()
        if not isinstance(doc, dict):
            raise ChildCallError("malformed request read")
        return doc

    def cancel(self, request_id: str) -> None:
        """Cancel a child (idempotent: an already-terminal child is a no-op)."""
        import requests

        resp = requests.post(
            f"{self._base_url}/v1/requests/{quote(request_id, safe='')}/cancel",
            headers=self._headers(),
            timeout=_HTTP_TIMEOUT_S,
        )
        if resp.status_code in (200, 404):
            return
        self._raise_for_error(resp.status_code, resp.text)

    def wait(
        self,
        request_id: str,
        *,
        timeout_s: Optional[float],
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ) -> List[Any]:
        """Poll to a terminal state; return the child's output items."""
        deadline = time.monotonic() + timeout_s if timeout_s and timeout_s > 0 else None
        while True:
            doc = self.get(request_id)
            status = str(doc.get("status") or "")
            if status == "completed":
                out = doc.get("output")
                return out if isinstance(out, list) else []
            if status == "failed":
                err = doc.get("error") or {}
                if not isinstance(err, dict):
                    err = {"message": str(err)}
                raise ChildRequestFailedError(
                    request_id,
                    error_type=str(err.get("type") or err.get("code") or ""),
                    error_message=str(err.get("message") or ""),
                )
            if status == "canceled":
                raise ChildRequestCanceledError(request_id)
            if deadline is not None and time.monotonic() >= deadline:
                raise ChildCallTimeoutError(request_id, timeout_s or 0.0)
            if self._cancel_event is not None:
                if self._cancel_event.wait(poll_interval_s):
                    raise CanceledError("canceled")
            else:
                time.sleep(poll_interval_s)

    def checkpoint_get(self, key: str) -> tuple[Any, bool]:
        import requests

        resp = requests.get(
            self._checkpoint_url(key), headers=self._headers(), timeout=_HTTP_TIMEOUT_S
        )
        if resp.status_code == 404:
            from .http_origin import is_proxy_outage

            if is_proxy_outage(resp):
                self._raise_for_error(resp.status_code, resp.text)
            return None, False
        if resp.status_code >= 300:
            self._raise_for_error(resp.status_code, resp.text)
        return resp.json(), True

    def checkpoint_put(self, key: str, value: Any) -> None:
        import requests

        body = json.dumps(value).encode("utf-8")
        resp = requests.put(
            self._checkpoint_url(key),
            headers={**self._headers(), "Content-Type": "application/json"},
            data=body,
            timeout=_HTTP_TIMEOUT_S,
        )
        if resp.status_code >= 300:
            self._raise_for_error(resp.status_code, resp.text)

    def _checkpoint_url(self, key: str) -> str:
        key = (key or "").strip()
        if not key:
            raise ValueError("checkpoint key is required")
        return (
            f"{self._base_url}/v1/requests/"
            f"{quote(self._parent_request_id, safe='')}/checkpoints/{quote(key, safe='')}"
        )


class ChildRequest:
    """Handle for one submitted child request (wait=False variant). wait_guard: result() waits run under the executor's child-call slot guard so a parked handler yields its GPU permit exactly like ctx.call_endpoint(wait=True); single status() reads stay unguarded — one bounded HTTP GET is not a park."""

    def __init__(
        self,
        client: CalloutClient,
        request_id: str,
        *,
        wait_guard: Optional[Callable[[], ContextManager[None]]] = None,
    ) -> None:
        self._client = client
        self._request_id = request_id
        self._wait_guard = wait_guard

    @property
    def request_id(self) -> str:
        return self._request_id

    def status(self) -> str:
        """One status read: queued | in_progress | completed | failed | canceled."""
        return str(self._client.get(self._request_id).get("status") or "")

    def result(
        self,
        timeout_s: Optional[float] = None,
        *,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ) -> List[Any]:
        """Poll to terminal; return output items or raise the typed error."""
        guard = self._wait_guard() if self._wait_guard is not None else nullcontext()
        with guard:
            return self._client.wait(
                self._request_id, timeout_s=timeout_s, poll_interval_s=poll_interval_s
            )

    def cancel(self) -> None:
        """Cancel this child (and, hub-side, its own descendants)."""
        self._client.cancel(self._request_id)
