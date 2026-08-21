"""WHO THIS POD IS — asked of the process that can answer, never of the one that structurally cannot."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from . import worker_credential
from .request_context._helpers import _decode_unverified_jwt_claims

logger = logging.getLogger(__name__)

CLAIM_ENDPOINT_ID = "graph_read_endpoint_id"
CLAIM_ORG_ID = "graph_read_org_id"


class IdentityUnavailable(RuntimeError):
    """Nobody in reach can name this pod."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class ViewerIdentity:
    """The endpoint this pod serves and the org that owns it."""

    endpoint_id: str = ""
    org_id: str = ""

    @property
    def named(self) -> bool:
        return bool(self.endpoint_id or self.org_id)


_LOCK = threading.Lock()
_CACHED: Optional[ViewerIdentity] = None


def from_claims(claims: Mapping[str, Any]) -> ViewerIdentity:
    """Build the identity from a decoded claim set."""
    return ViewerIdentity(
        endpoint_id=str(claims.get(CLAIM_ENDPOINT_ID) or "").strip(),
        org_id=str(claims.get(CLAIM_ORG_ID) or "").strip(),
    )


def from_token(token: str) -> ViewerIdentity:
    """Decode a worker credential's viewer claims."""
    raw = str(token or "").strip()
    if not raw:
        return ViewerIdentity()
    return from_claims(_decode_unverified_jwt_claims(raw))


def install(identity: ViewerIdentity) -> None:
    """Record a resolved identity (the parent's answer, or a test's)."""
    global _CACHED
    with _LOCK:
        _CACHED = identity


def installed() -> Optional[ViewerIdentity]:
    with _LOCK:
        return _CACHED


def reset() -> None:
    """Forget the resolved identity (test seam, and a hub re-wire)."""
    global _CACHED
    with _LOCK:
        _CACHED = None


def viewer() -> ViewerIdentity:
    """This pod's viewer identity, or a typed refusal."""
    cached = installed()
    if cached is not None:
        return cached
    identity = _resolve()
    install(identity)
    return identity


def _resolve() -> ViewerIdentity:
    token = worker_credential.current()
    if token:
        return from_token(token)

    from .procsplit import broker

    if not broker.active():
        raise IdentityUnavailable(
            "no_credential",
            "this process holds no worker credential and has no control seam "
            "to ask over, so nothing here can name the endpoint or org it "
            "serves")
    try:
        answer: Dict[str, str] = broker.viewer_identity()
    except Exception as exc:  # noqa: BLE001 — a refusal, with the cause named
        raise IdentityUnavailable(
            "relay_refused",
            f"the control parent could not name this pod: "
            f"{type(exc).__name__}: {exc}") from exc
    identity = ViewerIdentity(
        endpoint_id=str(answer.get("endpoint_id") or "").strip(),
        org_id=str(answer.get("org_id") or "").strip(),
    )
    logger.info(
        "worker identity relayed by the control parent: endpoint=%s org=%s",
        identity.endpoint_id or "<unstamped>", identity.org_id or "<unstamped>")
    return identity


__all__ = [
    "CLAIM_ENDPOINT_ID",
    "CLAIM_ORG_ID",
    "IdentityUnavailable",
    "ViewerIdentity",
    "from_claims",
    "from_token",
    "install",
    "installed",
    "reset",
    "viewer",
]
