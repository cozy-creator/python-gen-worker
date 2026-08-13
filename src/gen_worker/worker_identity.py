"""WHO THIS POD IS — asked of the process that can answer, never of the one
that structurally cannot.

THE CLASS OF DEFECT THIS MODULE DELETES
---------------------------------------
The compute child (the only execution model) holds **no worker credential by
construction**: the parent strips ``WORKER_JWT`` from its environment and no
frame carries it, so ``ChildTransport.current_worker_jwt`` returns ``""``
forever. Every gate that answers a question about *identity* by reading that
credential is therefore not merely wrong on some pods — it is wrong on
**every** real serving pod, always, and it looks like a security refusal while
it does it.

So identity is not a gate's private business. It is ONE process-wide fact with
ONE resolver, and this is it. A caller asks :func:`viewer`; it never decodes a
credential, never branches on which process it is in, and never receives ``""``
as if that were an answer.

HOW IT RESOLVES
---------------
1. **This process's own credential**, when it has one (``worker_credential``,
   the single source). Single-process/embedded workers, the mint CLI and the
   control parent all land here.
2. **The parent**, over the control seam that already mediates the resolve, the
   publish and the C2PA signature. The parent decodes ITS credential and
   returns the two claims. A claim is not a credential: the token itself never
   crosses.
3. **A typed refusal** (:class:`IdentityUnavailable`) when neither exists.
   Nothing here returns an empty identity to mean "could not ask" — that
   conflation is what makes a missing answer read as an attack.

An identity that resolves but names nothing IS an answer, and a legal one: the
hub stamps the claims only when it can (``cellgrant.Stamp``), and their absence
narrows the pod to platform-tier cells. "The hub declined to name us" and "we
could not ask anyone" must never share a value.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from . import worker_credential
from .request_context._helpers import _decode_unverified_jwt_claims

logger = logging.getLogger(__name__)

#: The hub-stamped viewer claims. ``cellgrant.Stamp`` writes
#: them from the hub's OWN record of the release — never from anything the
#: worker says — which is why they are the identity both ends of the cell
#: exchange scope by.
CLAIM_ENDPOINT_ID = "cell_read_endpoint_id"
CLAIM_ORG_ID = "cell_read_org_id"


class IdentityUnavailable(RuntimeError):
    """Nobody in reach can name this pod. A REFUSAL, never an empty identity.

    ``reason`` is the stable, greppable class:

    ``no_credential``   this process holds none and there is no seam to ask over
    ``relay_refused``   the parent was asked and could not answer
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class ViewerIdentity:
    """The endpoint this pod serves and the org that owns it.

    Both empty is legal and NARROWS the pod (platform-tier cells only); it is
    never a licence to widen. :attr:`named` distinguishes "the hub stamped
    nothing" from an identity that can match a publisher.
    """

    endpoint_id: str = ""
    org_id: str = ""

    @property
    def named(self) -> bool:
        return bool(self.endpoint_id or self.org_id)


_LOCK = threading.Lock()
_CACHED: Optional[ViewerIdentity] = None


def from_claims(claims: Mapping[str, Any]) -> ViewerIdentity:
    """Build the identity from a decoded claim set. The ONE mapping."""
    return ViewerIdentity(
        endpoint_id=str(claims.get(CLAIM_ENDPOINT_ID) or "").strip(),
        org_id=str(claims.get(CLAIM_ORG_ID) or "").strip(),
    )


def from_token(token: str) -> ViewerIdentity:
    """Decode a worker credential's viewer claims. THE only decoder.

    Unverified on purpose: this is OUR OWN credential, not an input. A worker
    that forged it would be attacking itself, and the hub verifies it on every
    call anyway. What must not be trusted is the RECEIPT, and that one is
    signature-verified before it is ever compared against this.
    """
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
    """This pod's viewer identity, or a typed refusal.

    Cached after the first successful resolution: the endpoint a pod serves and
    the org that owns it do not change for the life of the process, and a
    credential rotation carries the same two claims (``cellgrant.Stamp`` runs
    on both mint sites).

    Raises :class:`IdentityUnavailable` when nothing can answer. Callers must
    treat that as a refusal with a name — not as an unnamed pod.
    """
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

    # No local credential. Under the split that is the DESIGNED state, not a
    # fault: ask the process that holds one, over the same seam the resolve
    # itself uses.
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
