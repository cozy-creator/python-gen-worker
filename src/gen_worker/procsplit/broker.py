"""Child-side client for parent-mediated hub calls (delta 1)."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional
import json as _json
import requests
from . import actions
from . import is_compute_child
from . import frames

logger = logging.getLogger(__name__)

_broker: Optional["ChildBroker"] = None
_lock = threading.Lock()


class BrokerError(RuntimeError):
    """The parent refused, or could not perform, the requested action."""


@dataclass(frozen=True)
class HubResponse:
    """The subset of a response a caller may see."""

    status_code: int
    text: str

    def json(self) -> Any:

        return _json.loads(self.text) if self.text else {}


def install(broker: Optional["ChildBroker"]) -> None:
    global _broker
    with _lock:
        _broker = broker


def active() -> bool:
    return _broker is not None


def request(
    method: str,
    path: str,
    *,
    base_url: str = "",
    bearer: str = "",
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> HubResponse:
    """One hub call, parent-mediated when the split is on."""
    broker = _broker
    if broker is not None:
        return broker.call(method, path, params=params, json=json, timeout=timeout)

    if is_compute_child():
        raise BrokerError(
            f"{method} {path}: the control seam is down and this compute child "
            "holds no credential — hub calls are parent-mediated"
        )

    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    url = base_url.rstrip("/") + path
    verb = str(method).upper()
    kwargs: Dict[str, Any] = {"headers": headers, "timeout": timeout}
    if params:
        kwargs["params"] = params
    if verb == "GET":
        resp = requests.get(url, **kwargs)
    elif verb == "POST":
        resp = requests.post(url, json=json, **kwargs)
    else:
        resp = requests.request(verb, url, json=json, **kwargs)
    return HubResponse(status_code=resp.status_code, text=resp.text)


def viewer_identity() -> Dict[str, str]:
    """Ask the parent WHO THIS POD IS."""
    b = _broker
    if b is None:
        raise BrokerError(
            "no control seam: this process holds no worker credential and has "
            "no parent to ask who it is")
    result = b.call_action(actions.ACTION_VIEWER_IDENTITY, {})
    return {
        "endpoint_id": str(result.get("endpoint_id") or ""),
        "org_id": str(result.get("org_id") or ""),
    }


def report_detail(detail: str) -> bool:
    """Ask the parent to dial the hub with a typed worker report."""
    broker = _broker
    if broker is None:
        return False
    try:
        resp = broker.call_action("report.detail", {"detail": str(detail)})
        return bool(resp.get("delivered"))
    except Exception:
        logger.warning("parent-mediated worker report failed", exc_info=True)
        return False


class ChildBroker:
    """Correlated request/response over the control seam."""

    def __init__(self, loop: asyncio.AbstractEventLoop, send: Any) -> None:
        self._loop = loop
        self._send = send
        self._next_id = 0
        self._waiters: Dict[int, asyncio.Future] = {}
        self._id_lock = threading.Lock()

    def resolve(self, meta: Dict[str, Any]) -> None:
        """Deliver a T_ACTION_RESP."""
        try:
            rid = int(meta.get("id") or 0)
        except (TypeError, ValueError):
            return
        fut = self._waiters.pop(rid, None)
        if fut is not None and not fut.done():
            fut.set_result(meta)

    def fail_all(self, reason: str) -> None:
        """The seam is gone: every in-flight ask fails now rather than hanging a handler thread until its job deadline."""
        for fut in list(self._waiters.values()):
            if not fut.done():
                fut.set_exception(BrokerError(reason))
        self._waiters.clear()

    def call(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> HubResponse:
        meta = self._roundtrip(
            {
                "method": str(method).upper(),
                "path": str(path),
                "query": {
                    str(k): (
                        [("" if i is None else str(i)) for i in v]
                        if isinstance(v, (list, tuple))
                        else ("" if v is None else str(v))
                    )
                    for k, v in (params or {}).items()
                },
                "json": json,
                "timeout": float(timeout),
            },
            timeout=timeout,
        )
        if not meta.get("ok"):
            raise BrokerError(str(meta.get("error") or "parent refused the action"))
        return HubResponse(
            status_code=int(meta.get("status") or 0),
            text=str(meta.get("body") or ""),
        )

    def call_action(self, action: str, args: Dict[str, Any], *, timeout: float = 30.0) -> Dict[str, Any]:
        """A named non-HTTP action (``report.detail``)."""
        meta = self._roundtrip({"action": str(action), **args, "timeout": float(timeout)},
                               timeout=timeout)
        if not meta.get("ok"):
            raise BrokerError(str(meta.get("error") or "parent refused the action"))
        result = meta.get("result")
        return result if isinstance(result, dict) else {}

    def _roundtrip(self, payload: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:

        with self._id_lock:
            self._next_id += 1
            rid = self._next_id
        payload = dict(payload)
        payload["id"] = rid

        wait = float(timeout) + 15.0

        async def _go() -> Dict[str, Any]:
            fut: asyncio.Future = self._loop.create_future()
            self._waiters[rid] = fut
            try:
                await self._send(frames.T_ACTION_REQ, frames.pack_meta(payload))
                return await asyncio.wait_for(fut, wait)
            finally:
                self._waiters.pop(rid, None)

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            raise BrokerError(
                "parent-mediated hub calls are blocking and must not run on the "
                "child's event loop (wrap the caller in asyncio.to_thread)"
            )
        future = asyncio.run_coroutine_threadsafe(_go(), self._loop)
        try:
            return future.result(wait + 5.0)
        except BrokerError:
            raise
        except asyncio.TimeoutError as exc:
            raise BrokerError(f"parent did not answer within {wait:.0f}s") from exc
        except Exception as exc:
            raise BrokerError(f"parent-mediated action failed: {exc}") from exc


__all__ = [
    "BrokerError",
    "ChildBroker",
    "HubResponse",
    "active",
    "install",
    "report_detail",
    "request",
]
