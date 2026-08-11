"""Child-side client for parent-mediated hub calls (pgw#763 delta 1).

In the split, the compute child holds no worker JWT — so it cannot make an
identity-bearing hub call itself, by construction. It asks the parent, which
holds the credential, chooses the host, and decides (``procsplit/actions.py``).

Call sites keep one shape for both modes: :func:`request` performs the call
directly with ``requests`` when the split is off, and over the seam when it is
on. Nothing in a call site changes behaviour based on which process it is in,
so there is no "split-mode" code path for a bug to hide in.
"""

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

# The child's live broker, installed by ChildTransport once the seam is up.
_broker: Optional["ChildBroker"] = None
_lock = threading.Lock()


class BrokerError(RuntimeError):
    """The parent refused, or could not perform, the requested action."""


@dataclass(frozen=True)
class HubResponse:
    """The subset of a response a caller may see. Deliberately not a
    ``requests.Response``: headers and cookies are the parent's business."""

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
    """One hub call, parent-mediated when the split is on.

    ``base_url``/``bearer`` are used ONLY in single-process mode. Under the
    split the parent supplies both and ignores whatever the child passed —
    a child that could choose either would still hold the authority the split
    exists to take away from it.
    """
    broker = _broker
    if broker is not None:
        return broker.call(method, path, params=params, json=json, timeout=timeout)

    if is_compute_child():
        # The seam is down (or was never up) inside a compute child. Falling
        # through to a direct call would be an unauthenticated request the hub
        # rejects — which fails closed, but by accident. Say it instead: a
        # process that holds no credential has no business dialling the hub,
        # and a silent downgrade is how "the child never calls out" quietly
        # stops being true.
        raise BrokerError(
            f"{method} {path}: the control seam is down and this compute child "
            "holds no credential — hub calls are parent-mediated"
        )

    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    url = base_url.rstrip("/") + path
    verb = str(method).upper()
    # requests.get/.post by name, not requests.request: this is the same call
    # the call sites made before the seam existed, and the suite patches those
    # names. An indirection that changes what a test can see is a worse seam.
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
    """Ask the parent WHO THIS POD IS (pgw#1122).

    Returns ``{"endpoint_id": ..., "org_id": ...}`` — the two hub-stamped
    viewer claims, decoded by the parent out of the credential it holds. The
    token itself never crosses the seam: a claim is not a credential, and the
    whole point of delta 1 is that this process never holds one.

    Raises :class:`BrokerError` when there is no parent or the parent will not
    answer. It does NOT return an empty identity in that case — "the hub
    stamped nothing" and "nobody could be asked" are different facts, and
    collapsing them is the pgw#1122 defect.
    """
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
    """Ask the parent to dial the hub with a typed worker report.

    Returns False (never raises) when there is no parent — single-process
    callers use ``worker_fatal`` directly.
    """
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
    """Correlated request/response over the control seam.

    Callers are on executor threads (uploads, receipt checks, the c2pa signing
    callback invoked from native code), so :meth:`call` is a blocking, thread-
    safe façade over the child's single event loop.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, send: Any) -> None:
        self._loop = loop
        self._send = send            # async (ftype, payload) -> None
        self._next_id = 0
        self._waiters: Dict[int, asyncio.Future] = {}
        self._id_lock = threading.Lock()

    # ---- child-side plumbing --------------------------------------------

    def resolve(self, meta: Dict[str, Any]) -> None:
        """Deliver a T_ACTION_RESP. Runs on the child's loop."""
        try:
            rid = int(meta.get("id") or 0)
        except (TypeError, ValueError):
            return
        fut = self._waiters.pop(rid, None)
        if fut is not None and not fut.done():
            fut.set_result(meta)

    def fail_all(self, reason: str) -> None:
        """The seam is gone: every in-flight ask fails now rather than hanging
        a handler thread until its job deadline."""
        for fut in list(self._waiters.values()):
            if not fut.done():
                fut.set_exception(BrokerError(reason))
        self._waiters.clear()

    # ---- callers ---------------------------------------------------------

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
                # A list value is a repeated query key (the v2 receipt fetch);
                # items are stringified, the shape survives the seam.
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

    # ---- transport -------------------------------------------------------

    def _roundtrip(self, payload: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:

        with self._id_lock:
            self._next_id += 1
            rid = self._next_id
        payload = dict(payload)
        payload["id"] = rid

        # The parent's own call is bounded by `timeout`; give the round trip a
        # margin so a caller sees the parent's typed refusal rather than a
        # local timeout that says nothing about why.
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
