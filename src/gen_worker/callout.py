"""Endpoint-to-endpoint call-out client (th#826, the workflow primitive).

The SDK half of the platform's call-out primitive: a function declared with
``@endpoint(child_calls=True)`` receives an ``invoke_child`` grant on its
per-job capability token; this module submits/polls/cancels child requests
and reads/writes the invocation's workflow checkpoints through the ordinary
platform HTTP API using that token as the bearer.

Depth caps, cycle detection, tree budgets, tier inheritance, payer
attribution, and tree cancellation are all enforced HUB-side (the token is
proof of possession only) — this client just surfaces the typed refusals.

Wire contract: tensorhub docs/callout.md.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

from .api.errors import (
    CanceledError,
    ChildCallError,
    ChildCallRefusedError,
    ChildCallTimeoutError,
    ChildRequestCanceledError,
    ChildRequestFailedError,
)

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

# Route-scope denials for a capability token WITHOUT the invoke_child grant —
# the function didn't declare child_calls=True.
_NOT_DECLARED_CODES = frozenset({"forbidden", "insufficient_scope", "unauthorized"})

DEFAULT_POLL_INTERVAL_S = 2.0
_HTTP_TIMEOUT_S = 60.0


def _parse_error_body(text: str) -> tuple[str, str]:
    """Best-effort (code, message) from a platform error response body."""
    try:
        doc = json.loads(text or "{}")
    except ValueError:
        return "", (text or "")[:256]
    err = doc.get("error")
    if isinstance(err, dict):
        return str(err.get("code") or ""), str(err.get("message") or "")
    if isinstance(err, str):
        return "", err[:256]
    return "", (text or "")[:256]


class CalloutClient:
    """HTTP client for one invocation's child calls + checkpoints.

    Constructed by :class:`~gen_worker.request_context.RequestContext` —
    tenant code uses ``ctx.call_endpoint`` / ``ctx.workflow_checkpoint``.
    """

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

    # -- low-level ---------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        token = self._get_token()
        if not token:
            raise ChildCallRefusedError(
                "child_calls_not_declared",
                "no capability token in this invocation context; declare "
                "@endpoint(child_calls=True) and run under the platform",
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
                "declare @endpoint(child_calls=True)",
            )
        raise ChildCallError(
            f"child call failed ({status_code}{', ' + code if code else ''}): "
            f"{message or text[:256]}"
        )

    # -- submit / poll / cancel --------------------------------------------

    def submit(
        self,
        endpoint: str,
        function: str,
        payload: Dict[str, Any],
        *,
        tag: str = "prod",
        tier: Optional[str] = None,
    ) -> str:
        """Submit one child request; returns its request id."""
        import requests

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
        url = f"{self._base_url}/{endpoint}/{function}:{(tag or 'prod').strip()}"
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
        """Poll to a terminal state; return the child's output items.

        Cooperative with parent cancellation: when this invocation is
        cancelled (the hub is cascading to the children anyway), the wait
        raises :class:`CanceledError` promptly.
        """
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

    # -- checkpoints ---------------------------------------------------------

    def checkpoint_get(self, key: str) -> tuple[Any, bool]:
        import requests

        resp = requests.get(
            self._checkpoint_url(key), headers=self._headers(), timeout=_HTTP_TIMEOUT_S
        )
        if resp.status_code == 404:
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
    """Handle for one submitted child request (``wait=False`` variant)."""

    def __init__(self, client: CalloutClient, request_id: str) -> None:
        self._client = client
        self._request_id = request_id

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
        return self._client.wait(
            self._request_id, timeout_s=timeout_s, poll_interval_s=poll_interval_s
        )

    def cancel(self) -> None:
        """Cancel this child (and, hub-side, its own descendants)."""
        self._client.cancel(self._request_id)
