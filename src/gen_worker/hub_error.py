from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from .api.errors import WorkerError

__all__ = [
    "HubError",
    "HubApiError",
    "parse_hub_error",
    "hub_error_of",
    "raise_for_hub_error",
]

_MAX_MESSAGE_CHARS = 400

_CODE_TOKEN = re.compile(r"^[a-z][a-z0-9_.\-]{0,63}$")


def _one_line(text: str, limit: int = _MAX_MESSAGE_CHARS) -> str:
    return " ".join(str(text or "").split())[:limit]


@dataclass(frozen=True)
class HubError:
    """What the hub said, as far as it said anything."""

    code: str = ""
    message: str = ""
    request_id: str = ""

    def __bool__(self) -> bool:
        return bool(self.code or self.message)

    def detail(self) -> str:
        """``"code: message (hub request_id=...)"`` — code FIRST so refusals group by a stable token instead of by prose."""
        head = ": ".join(p for p in (self.code, self.message) if p)
        if self.request_id:
            head = f"{head} (hub request_id={self.request_id})" if head else (
                f"(hub request_id={self.request_id})")
        return head


def parse_hub_error(body: Any) -> HubError:
    """Best-effort ``HubError`` from a response body (``str``/``bytes``/dict)."""
    try:
        doc: Any = body
        if isinstance(doc, (bytes, bytearray)):
            doc = doc.decode("utf-8", "replace")
        if isinstance(doc, str):
            text = doc.strip()
            if not text:
                return HubError()
            try:
                doc = json.loads(text)
            except ValueError:
                return HubError(message=_one_line(text))
        if not isinstance(doc, dict):
            return HubError(message=_one_line(str(doc)))

        request_id = _one_line(doc.get("request_id") or "", 128)
        err = doc.get("error")
        if isinstance(err, dict):
            return HubError(
                code=_one_line(err.get("code") or "", 128),
                message=_one_line(err.get("message") or ""),
                request_id=_one_line(err.get("request_id") or "", 128) or request_id,
            )
        if isinstance(err, str) and err.strip():
            sibling = doc.get("message")
            if isinstance(sibling, str) and sibling.strip():
                return HubError(code=_one_line(err, 128),
                                message=_one_line(sibling),
                                request_id=request_id)
            token = err.strip()
            if _CODE_TOKEN.match(token):
                return HubError(code=token, request_id=request_id)
            return HubError(message=_one_line(token), request_id=request_id)
        msg = doc.get("message")
        if isinstance(msg, str) and msg.strip():
            return HubError(message=_one_line(msg), request_id=request_id)
        return HubError(request_id=request_id)
    except Exception:  # noqa: BLE001 - the error path never raises a second error
        return HubError()


def hub_error_of(resp: Any) -> HubError:
    """``parse_hub_error`` over a ``requests.Response``-shaped object."""
    try:
        text = resp.text
    except Exception:  # noqa: BLE001
        try:
            text = resp.content
        except Exception:  # noqa: BLE001
            return HubError()
    err = parse_hub_error(text)
    if not err.request_id:
        try:
            rid = str((resp.headers or {}).get("X-Request-Id") or "").strip()
        except Exception:  # noqa: BLE001
            rid = ""
        if rid:
            err = HubError(code=err.code, message=err.message, request_id=rid[:128])
    return err


class HubApiError(WorkerError):
    """A non-2xx from the hub's HTTP API, carrying what the hub actually said."""

    def __init__(
        self,
        status_code: int,
        *,
        code: str = "",
        message: str = "",
        request_id: str = "",
        what: str = "",
        retryable: Optional[bool] = None,
    ) -> None:
        self.status_code = int(status_code or 0)
        self.code = str(code or "")
        self.message = str(message or "")
        self.request_id = str(request_id or "")
        self.what = str(what or "")
        self._retryable = retryable
        detail = HubError(self.code, self.message, self.request_id).detail()
        head = f"{self.what} " if self.what else ""
        super().__init__(
            f"{head}refused by the hub (HTTP {self.status_code})"
            + (f": {detail}" if detail else "")
        )

    @property
    def retryable(self) -> bool:
        if self._retryable is not None:
            return bool(self._retryable)
        return self.status_code in (408, 429) or self.status_code >= 500


def raise_for_hub_error(resp: Any, *, what: str = "") -> Any:
    """``resp.raise_for_status()``'s replacement for calls to OUR hub."""
    status = int(getattr(resp, "status_code", 0) or 0)
    if 200 <= status < 300:
        return resp

    from .http_origin import REQUEST_DETERMINED_STATUSES, response_is_from_hub

    err = hub_error_of(resp)
    retryable: Optional[bool] = None
    if status not in REQUEST_DETERMINED_STATUSES and not response_is_from_hub(resp):
        retryable = True
    raise HubApiError(
        status,
        code=err.code,
        message=err.message,
        request_id=err.request_id,
        what=what or _describe(resp),
        retryable=retryable,
    )


def _describe(resp: Any) -> str:
    try:
        req = getattr(resp, "request", None)
        method = str(getattr(req, "method", "") or "").upper()
        url = str(getattr(resp, "url", "") or "")
        if method and url:
            return f"{method} {url}"
        return url
    except Exception:  # noqa: BLE001
        return ""
