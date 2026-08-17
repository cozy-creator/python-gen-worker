"""The HUB tier of the key-set search: pull by closure digest, push after a derive.

pgw#1353 option (b) / th#2123. `GET`/`PUT /v1/worker/keysets/<closure_digest>`.

WHY A FOURTH TIER, WHEN THERE ARE ALREADY THREE
-----------------------------------------------
pgw#1327 gave a pod a document baked into its image; pgw#1353 gave it the
platform-placed durable root; §4.28 already gave it its own cache. Each fixes a
different pod and NONE of them fixes an **ephemeral private deployment**: no
network volume (so the durable root is the pod's own disk, which dies with it),
no baked document (the closure digest binds a deploy-time checkpoint ref the
image build cannot state), and a fresh container every time. That pod pays the
full derive — measured 778-833 s on four independent sdxl pods — once per pod,
forever, and it is exactly the shape a private deployment takes.

The hub is the one store every pod shape can reach. It is also the only one that
can be filled by the MINT lane at mint time, which is the point: a serving pod
should never derive, and `scripts/emit_cg_keyset.py --publish` is what makes
that true without an image rebuild.

DESIGN-RULINGS §4.29, AS AMENDED
--------------------------------
The ruling is *"the worker derives its key and asks the hub BY KEY: ONE artifact
or MISS… admission verifies the answer against the derived key"*, and Paul's
keys-shipped-as-data amendment is what puts a key SET inside it rather than only
an artifact. This module is that shape one layer up: the address is the closure
digest, the answer is one document or a miss, and admission is
`document.parse_closure(document, digest)` — the answer must NAME the address it
was asked at, or it is refused.

AN OPTIMIZATION, NEVER A GATE — AND THE ASYMMETRY THAT MATTERS
---------------------------------------------------------------
A LOCAL document that does not parse is `keyset_invalid` and PROPAGATES, because
a malformed document in the image or on the volume is a mint-lane defect that
must be visible as itself. A HUB answer that does not parse is a MISS with a
recorded reason, and the pod derives.

That asymmetry is deliberate and is not laziness. The hub is a network peer on
the boot path: it can be down, slow, rebuilding, or behind a middlebox that
rewrote the body. Every one of those must degrade to the 805 s path the pod
already had, because the alternative — a pod that refuses to boot because a
cache was unreachable — is strictly worse than the problem this fixes. The
`hub_reason` on the outcome is what keeps it from being silent.

The PUBLISH half is best-effort by the same rule and one more: it runs AFTER the
derive, so its failure cannot make this boot slower than it already was, and it
never blocks. A pod that traced for 805 s and then could not upload has still
booted; the next pod pays again and the reason is recorded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .document import (KEYSET_SCHEMA, KEYSET_VERSION, ClosureRow, KeySetDocument,
                       decode, encode, parse_closure)
from .document import ShippedClosure
from .identifiers import ClosureDigest, KeySetError

logger = logging.getLogger(__name__)

__all__ = [
    "KEYSET_PATH_PREFIX",
    "KEYSET_TIMEOUT_S",
    "HubTier",
    "fetch_closure",
    "keyset_path",
    "publish_closure",
    "single_closure_document",
]

#: The route's address space. The path segment IS the closure digest — a
#: validated :class:`ClosureDigest`, never an f-string of whatever was handy, so
#: a caller cannot put a folded key or a class hash in the one place the hub
#: reads as an address.
KEYSET_PATH_PREFIX = "/v1/worker/keysets/"

#: Mandatory (``scripts/lint_http_timeouts.py``). SHORT relative to the resolve
#: route's 30 s, and the asymmetry is the design: a resolve answer saves a full
#: cold mint and is worth waiting for, while this one saves a derive the pod can
#: simply do. A hub that has not answered in 10 s cannot beat the fallback, so
#: waiting longer only adds to the boot it is meant to shorten.
KEYSET_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class HubTier:
    """WHO to ask, as a value.

    A typed object rather than two loose strings threaded through five call
    sites, and ``None`` rather than an empty ``HubTier`` for "there is nobody to
    ask": the two states have different meanings and only one of them should be
    able to reach :func:`fetch_closure`. ``absent`` carries the CALLER's own
    sentence for why there is no hub (pgw#1127's shape), which is a detail and
    never a decision.
    """

    base_url: str = ""
    bearer: str = ""
    absent: str = ""

    @property
    def reachable(self) -> bool:
        """Whether asking is even meaningful.

        Under the process split the child holds no credential and the parent
        supplies both, so an empty ``bearer`` is NOT proof there is nobody to
        ask — ``broker.request`` decides that. What makes a tier unreachable is
        the caller SAYING so.
        """
        return not self.absent


def keyset_path(digest: ClosureDigest) -> str:
    """The route path for one address. Takes the validated type deliberately."""
    return f"{KEYSET_PATH_PREFIX}{digest}"


def single_closure_document(
    digest: ClosureDigest, row: ClosureRow,
) -> bytes:
    """The exact bytes a PUT carries: ONE closure, keyed by its own address.

    One closure and not the pod's whole document, because the address is
    per-closure and the hub refuses a body naming more than one. Sending the
    whole cache would also leak every other release this pod has ever booted
    into a row addressed at one of them.
    """
    return encode(KeySetDocument(
        schema=KEYSET_SCHEMA, version=KEYSET_VERSION,
        closures={str(digest): row}))


def fetch_closure(
    digest: ClosureDigest, tier: HubTier,
) -> Tuple[Optional[ShippedClosure], str]:
    """Ask the hub for ONE closure. Returns ``(closure, reason)``.

    ``(closure, "")`` is a hit; ``(None, reason)`` is everything else, and the
    reason is never empty on a miss — a silent miss is indistinguishable from a
    tier nobody wired, and this whole issue exists because a cost nobody could
    see went unfixed for a release cycle.

    NEVER RAISES. Every failure mode of a network peer on the boot path
    degrades to "derive", which is what the pod did before this tier existed.
    """
    if not tier.reachable:
        return None, f"no hub to ask: {tier.absent}"
    from ..procsplit import broker

    try:
        resp = broker.request(
            "GET", keyset_path(digest),
            base_url=tier.base_url, bearer=tier.bearer,
            timeout=KEYSET_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 — see the docstring
        logger.debug("keyset: hub fetch failed", exc_info=True)
        return None, f"hub unreachable ({type(exc).__name__}: {exc})"

    if resp.status_code == 404:
        # THE ORDINARY ANSWER, and a complete one. Nobody has derived this
        # closure yet — which on the first pod of a release is the truth.
        return None, "hub holds no document for this closure"
    if resp.status_code != 200:
        return None, f"hub answered {resp.status_code}: {resp.text[:200]}"

    try:
        document = decode(resp.text.encode("utf-8"))
        closure = parse_closure(document, digest)
    except KeySetError as exc:
        # A hub answer that does not parse, or that carries a DIFFERENT closure
        # than the address it was asked at, is refused and the pod derives.
        # Refused rather than trusted: the address check is the whole admission
        # (§4.29), and an answer that fails it is either a hub defect or
        # something in between that rewrote the body — neither is a key set.
        logger.warning(
            "keyset: the hub's answer for closure %s did not admit (%s: %s)",
            digest, exc.reason, exc.detail)
        return None, f"hub answer refused ({exc.reason}: {exc.detail})"
    except Exception as exc:  # noqa: BLE001 — see the docstring
        logger.debug("keyset: hub answer unreadable", exc_info=True)
        return None, f"hub answer unreadable ({type(exc).__name__}: {exc})"
    return closure, ""


def publish_closure(
    digest: ClosureDigest, row: ClosureRow, tier: HubTier,
) -> str:
    """Offer one derived closure to the hub. Returns ``""`` on success.

    BEST EFFORT AND NEVER FATAL, and the call site relies on that: it runs
    after a derive that has already produced this boot's keys, so nothing here
    can make this boot slower or less correct than it was. What it buys is the
    NEXT pod's boot.

    A 409 is not a failure worth escalating — it means another pod of this org
    already stored a different document for this closure, the store is
    write-once, and the incumbent stands. It is returned as a reason so it can
    be recorded, because two pods running the same code against the same
    subjects tracing different class hashes IS a finding; it is just not this
    boot's problem, and the mint's own honesty gate is what adjudicates it.
    """
    if not tier.reachable:
        return f"no hub to publish to: {tier.absent}"
    from ..procsplit import broker

    try:
        body = single_closure_document(digest, row)
    except KeySetError as exc:
        # This pod's OWN document did not survive re-encoding. Not a hub
        # problem, and worth a loud line: it means the deriver recorded a row
        # this worker's own writer would refuse.
        logger.warning(
            "keyset: refusing to publish closure %s — %s: %s",
            digest, exc.reason, exc.detail)
        return f"document not encodable ({exc.reason}: {exc.detail})"

    try:
        resp = broker.request(
            "PUT", keyset_path(digest),
            base_url=tier.base_url, bearer=tier.bearer,
            json=_json_of(body), timeout=KEYSET_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 — see the docstring
        logger.debug("keyset: hub publish failed", exc_info=True)
        return f"hub unreachable ({type(exc).__name__}: {exc})"

    if resp.status_code in (200, 201):
        logger.info(
            "keyset: closure %s is now on the hub (%s) — the next pod of this "
            "endpoint states its keys without tracing",
            digest, "stored" if resp.status_code == 201 else "already stored")
        return ""
    if resp.status_code == 409:
        return f"hub holds a different document for this closure: {resp.text[:200]}"
    return f"hub refused the publish with {resp.status_code}: {resp.text[:200]}"


def _json_of(body: bytes) -> Dict[str, Any]:
    """The document, as the object ``broker.request`` serializes.

    The broker's wire is ``json=``, so the bytes are re-serialized by
    ``requests`` rather than sent verbatim. That is fine and is why the hub
    computes its stored digest over WHAT IT RECEIVED rather than over anything
    this side claims: the two are different byte strings for the same document,
    and only one of them is the one anybody has to agree about.
    """
    import json as _json

    parsed = _json.loads(body.decode("utf-8"))
    if not isinstance(parsed, dict):  # pragma: no cover — encode() cannot do this
        raise KeySetError("keyset_invalid", "a document must encode to an object")
    return parsed
