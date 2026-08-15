"""Hub-signed cell receipt verification.

A ``cell_store`` row (compiled_graph_key -> artifact) is a Nix *realisation*: fetch
verifies the BYTES against the hub's recorded digest, but nothing signs the
RECORD binding the key to those bytes. Under "bucket is truth, DB is a
rebuildable index" that makes bucket write access equivalent to arbitrary cell
delivery after any index rebuild.

The hub signs a ``compiled-graph-receipt-v1`` compact JWS at publish-finalize binding:
compiled_graph_key + owning endpoint + the publisher-trust rung + the snapshot digest
(the derivation binding Nix's fingerprint omits) + the packed tarball's
ALGORITHM-TAGGED digest AND integral size (Bazel REv2: size is part of the
digest). This module is the WORKER half: before arming any hub-delivered
artifact the worker fetches the receipt, verifies the signature against
the hub's public artifact-signing JWKS, checks every binding against the
local bytes, and re-checks the operator revocation list — the targeted
recall, and the ONLY recall lever.

Refusal semantics: a failed receipt DISCARDS the delivered artifact with a
loud typed ``compiled_graph_receipt_refused`` activity event and falls through to
the ordinary miss policy (fleet workers self-mint their own replacement —
their own bytes need no receipt; the copy they publish gets one from the
publish gate). A receipt failure never kills serving.

SHA-256 ONLY. The receipt's canonical binding is
``artifact.digest``, always tagged (``sha256:<hex>``), and verification
dispatches on that tag. An untagged bare-hex digest is REFUSED rather than
read as some assumed algorithm, and a receipt with no usable digest at all is
REFUSED rather than compared against nothing. There is no bare-hex
``artifact.blake3`` arm: a worker meeting one refuses it, self-mints, and
publishes a replacement whose receipt is sha256-bound — the designed miss
policy, not a new failure.

Configuration happens at the HelloAck site in ``lifecycle`` (the same
moment ``file_base_url`` arrives). cozy-local and the CLI never configure
this module, so the gate is a no-op there — user-controlled stores keep
their local trust model.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from . import activity as activity_mod
from . import artifact_meta
from . import worker_identity
from .procsplit import broker

logger = logging.getLogger(__name__)

# `publisher_tier` and `publisher_org_id` are LOAD-BEARING at the arm gate
# below, so an older receipt must never be read as a v2 one with the trust
# fields missing: the version check refuses outright rather than defaulting.
RECEIPT_VERSION = "compiled-graph-receipt-v1"

# Publisher tiers. `platform` means the platform vouches for the
# publishing org's endpoint code, so the cell is adoptable fleet-wide within its
# family. Anything else — including an absent or unrecognised value — is `org`,
# and an org-scoped cell is adoptable only by pods of the endpoint that minted
# it. There is deliberately no third value and no "unknown" branch: every
# unparseable tier must land on the NARROWER rule.
COMPILED_GRAPH_PUBLISHER_TIER_PLATFORM = "platform"
COMPILED_GRAPH_PUBLISHER_TIER_ORG = "org"
# The algorithm this worker can actually recompute from local bytes. A receipt
# naming anything else is refused, never assumed.
# There is deliberately no multi-algorithm ceremony for a second algorithm
# that does not exist. The tag stays on the wire (a bare hex string silently
# acquires whatever algorithm the reader assumed), so adding an algorithm
# later is still a local change — it is just not pre-paid here.
ARTIFACT_DIGEST_ALGORITHM = "sha256"
JWKS_PATH = "/api/v1/artifacts/.well-known/jwks.json"
RECEIPT_PATH = "/v1/worker/compiled-graphs/receipt"
REVOCATIONS_PATH = "/v1/worker/compiled-graphs/revocations"

# Per-CALL socket budget on the three small control-plane round trips this
# module makes (JWKS, receipt, revocations). Not a kill — none of them can be
# "making progress" in any observable sense and expiry ends no work — but
# without it a hub that accepts a connection and says nothing wedges the arm
# gate every cell adoption waits behind. Shorter than `callout`'s 60 s because
# these are constant-size answers, not a child request the hub may
# legitimately take time to admit.
_HTTP_TIMEOUT_S = 30


class ReceiptError(RuntimeError):
    """Typed receipt refusal. ``reason`` is the stable, greppable class."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class Receipt:
    """Decoded, signature-verified cell receipt claims — the ones something
    CHECKS.

    Decoded: ``axes``, ``publisher``, ``artifact_path``, ``manifest_digest``,
    ``fingerprint_digest`` and ``issued_at_unix``. The signature still covers the whole payload, so a
    claim this worker does not decode is neither trusted nor forgeable — it is
    simply not a promise anyone here relies on. Two notes worth keeping:

    * ``manifest_digest``/``fingerprint_digest`` are additive ``omitempty``
      claims the hub hashes in (``api/cell_receipts.go``); the content they
      cover is already bound by ``snapshot_digest``, and removing an additive
      claim does not move ``RECEIPT_VERSION``.
    * ``issued_at_unix`` had NO freshness check anywhere, and adding one would
      be a fixed-duration condemn — the thing the no-magic-timeouts rule
      forbids. Receipt currency is the REVOCATION list (:data:`REVOCATIONS_PATH`,
      fail-closed when unreadable), which is an authority answering, not a
      clock guessing.

    A claim added back must arrive with the code that refuses on it.
    """

    version: str
    family: str
    compiled_graph_key: str
    owning_endpoint_id: str
    # The publisher-trust boundary, inside the signature.
    publisher_tier: str
    publisher_org_id: str
    snapshot_digest: str
    # Canonical, ALGORITHM-TAGGED ("<algo>:<hex>"). Never bare hex.
    artifact_digest: str
    artifact_size_bytes: int


@dataclass
class _Config:
    base_url: str
    #: A BEARER, and nothing else. Under the split the parent supplies the real
    #: credential and ignores this one; it is used only by the single-process
    #: path's direct fetches. WHO THIS POD IS comes from ``worker_identity``,
    #: never from decoding this.
    worker_jwt: Callable[[], str]
    # kid -> RSA public key, lazily fetched from the hub JWKS.
    jwks: Dict[str, rsa.RSAPublicKey] = field(default_factory=dict)


_LOCK = threading.Lock()
_CONFIG: Optional[_Config] = None


def configure(base_url: str, worker_jwt: Callable[[], str]) -> None:
    """Arm the receipt gate. Called from the HelloAck site; idempotent
    (re-configuration replaces the base URL and drops the JWKS cache so a
    hub bounce with rotated keys re-fetches)."""
    global _CONFIG
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return
    with _LOCK:
        _CONFIG = _Config(base_url=base, worker_jwt=worker_jwt)
    logger.info("receipts: gate configured against %s", base)


def configured() -> bool:
    with _LOCK:
        return _CONFIG is not None


def reset() -> None:
    """Disarm the gate (test seam)."""
    global _CONFIG
    with _LOCK:
        _CONFIG = None


# -- crypto -----------------------------------------------------------------


def _b64url_decode(segment: str) -> bytes:
    pad = "=" * (-len(segment) % 4)
    try:
        return base64.urlsafe_b64decode(segment + pad)
    except (binascii.Error, ValueError) as exc:
        raise ReceiptError("receipt_malformed", f"base64url decode failed: {exc}") from exc


# -- publisher trust --------------------------------------------------------


def _normalize_publisher_tier(raw: object) -> str:
    """Anything that is not exactly ``platform`` is ``org``.

    No error return and no third value on purpose: a caller forced to branch on
    an error eventually gets the branch wrong, and every wrong branch here ends
    in ``dlopen``.
    """
    if str(raw or "").strip() == COMPILED_GRAPH_PUBLISHER_TIER_PLATFORM:
        return COMPILED_GRAPH_PUBLISHER_TIER_PLATFORM
    return COMPILED_GRAPH_PUBLISHER_TIER_ORG


def _self_viewer() -> "worker_identity.ViewerIdentity":
    """WHO THIS POD IS, from the one resolver that can answer.

    Never decode ``cell_read_endpoint_id``/``cell_read_org_id`` out of
    ``cfg.worker_jwt()`` — *this process's* credential. The gate is armed at
    HelloAck inside the COMPUTE CHILD, which holds no credential by
    construction, so both claims are ``""`` on every real serving pod and
    every org-tier cell would be refused ``publisher_untrusted``.

    ``worker_identity.viewer`` asks this process's own credential when it has
    one and the control PARENT when it does not — the same seam the resolve
    itself uses. Its refusal is typed, so "the hub stamped no claims" (a legal
    narrowing) never wears the same face as "nobody could be asked".
    """
    return worker_identity.viewer()


def needs_viewer_identity(receipt: Receipt) -> bool:
    """Whether the trust rule will consult WHO THIS POD IS.

    One spelling, two readers: :func:`refuse_untrusted_publisher` opens with
    it, and the caller asks it BEFORE resolving an identity — a platform-tier
    cell is adoptable by any pod, so demanding an identity to arm one would
    turn a resolver outage into a refusal of the one cell class that never
    needed a resolver.
    """
    return receipt.publisher_tier != COMPILED_GRAPH_PUBLISHER_TIER_PLATFORM


def refuse_untrusted_publisher(
    receipt: Receipt, self_endpoint_id: str, self_org_id: str = "",
) -> None:
    """Raise unless this pod may adopt ``receipt``'s cell.

    THE RULE: a cell must have come from THIS endpoint, from THIS pod's OWN ORG,
    or from a publisher the platform vouches for.

    Paul's ruling is literally endpoint-scoped (*"must have come from endpoint-A
    itself, or from a publisher that is us or a trusted party"*); the ORG arm
    matches the rule the hub's listing applies (``authz.CellAdoptable``), so
    the two layers agree instead of the listing showing an org its own cell
    from a sibling endpoint and this function refusing it.

    THREAT: cross-tenant native-code execution. The artifact is a ``.so`` this
    process is about to ``dlopen``. Nothing else prevents it — the digest
    proves the bytes are the ones the hub signed, not that the hub meant them
    for US; the cell key proves nothing at all, because the hub cannot verify
    artifact-to-graph correspondence without recompiling; and the hub's own
    listing filter only covers the ONE path that goes through a listing. This
    runs on every path, and it is the last check before the load.
    """
    if not needs_viewer_identity(receipt):
        return
    owner = str(receipt.owning_endpoint_id or "").strip()
    mine = str(self_endpoint_id or "").strip()
    owner_org = str(receipt.publisher_org_id or "").strip()
    my_org = str(self_org_id or "").strip()

    # Same endpoint: the narrowest match.
    if owner and mine and owner == mine:
        return
    # Same org. BOTH sides must be non-empty: an empty-equals-empty match is a
    # vacuous guard — two cells neither of which can be attributed must not
    # match each other.
    if owner_org and my_org and owner_org == my_org:
        return

    if not owner and not owner_org:
        raise ReceiptError(
            "publisher_untrusted",
            "org-tier receipt names neither an owning endpoint nor a publisher "
            "org, so no pod may adopt it")
    if not mine and not my_org:
        raise ReceiptError(
            "publisher_untrusted",
            "this pod cannot name its own endpoint or org (no cell_read_* claim "
            "on the worker credential), so it may adopt platform-tier cells only")
    raise ReceiptError(
        "publisher_untrusted",
        f"cell was minted for endpoint {owner or '<unnamed>'} "
        f"(org {owner_org or '<unnamed>'}) and this pod serves "
        f"{mine or '<unnamed>'} (org {my_org or '<unnamed>'})")


#: THREAT: the JWKS `n` is base64url off the network with no
#: declared length, so a multi-MB modulus makes EVERY later receipt
#: verification super-linear for the life of the cached key — and nothing
#: downstream refuses it, because the signatures still check out. Real keys are
#: 2048-4096 bits. Bounded here because this is the only place the bytes become
#: a key.
MAX_RSA_MODULUS_BITS = 8192


def _rsa_key_from_jwk(jwk: Mapping[str, object]) -> Optional[rsa.RSAPublicKey]:
    if str(jwk.get("kty") or "") != "RSA":
        return None
    n_raw, e_raw = str(jwk.get("n") or ""), str(jwk.get("e") or "")
    if not n_raw or not e_raw:
        return None
    n_bytes, e_bytes = _b64url_decode(n_raw), _b64url_decode(e_raw)
    # RSA requires 1 < e < n, so ONE bound covers both operands of the modexp
    # whose cost is the threat. A key this big is refused, not skipped: the hub
    # publishing one is a fact about the hub, and quietly dropping it would
    # leave a pod verifying against whichever key happened to parse.
    oversized = max(len(n_bytes), len(e_bytes)) * 8
    if oversized > MAX_RSA_MODULUS_BITS:
        raise ReceiptError(
            "jwks_modulus_oversized",
            f"hub JWKS key {str(jwk.get('kid') or '<unnamed>')!r} carries a "
            f"{oversized}-bit modulus/exponent, over the "
            f"{MAX_RSA_MODULUS_BITS}-bit bound")
    n = int.from_bytes(n_bytes, "big")
    e = int.from_bytes(e_bytes, "big")
    return rsa.RSAPublicNumbers(e=e, n=n).public_key()


def canonical_artifact_digest(digest: str) -> str:
    """Resolve the receipt's algorithm-tagged artifact digest.

    Every route to "nothing to compare against" is a typed REFUSAL, because
    that is the shape this whole migration keeps producing: an absent field
    makes a guard vacuously true and the integrity check silently disappears.
    A bare hex string is refused for the same reason a bare CAS ref is —
    it silently acquires whatever algorithm the reader assumed.
    """
    d = str(digest or "").strip().lower()
    if not d:
        raise ReceiptError("receipt_no_artifact_digest", "receipt binds no artifact digest")
    algo, sep, hex_part = d.partition(":")
    if not sep:
        raise ReceiptError(
            "receipt_digest_untagged",
            f"artifact digest {d[:16]}… carries no algorithm tag")
    if algo != ARTIFACT_DIGEST_ALGORITHM:
        raise ReceiptError("receipt_digest_algorithm_unsupported", f"algo={algo!r}")
    if not _is_hex64(hex_part):
        raise ReceiptError(
            "receipt_digest_malformed",
            f"{algo} digest must be 64 hex characters")
    return f"{algo}:{hex_part}"


def _is_hex64(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def verify_receipt_jws(jws: str, keys: Mapping[str, rsa.RSAPublicKey]) -> Receipt:
    """Verify a compact JWS against ``keys`` (kid -> RSA public key) and
    return the decoded claims. Raises :class:`ReceiptError` with a named
    reason on ANY failure — malformed, unknown kid, alg downgrade, bad
    signature, wrong version."""
    parts = str(jws or "").strip().split(".")
    if len(parts) != 3 or not all(parts[:2]):
        raise ReceiptError("receipt_malformed", "not a compact JWS")
    try:
        header = json.loads(_b64url_decode(parts[0]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReceiptError("receipt_malformed", f"header: {exc}") from exc
    if not isinstance(header, dict):
        raise ReceiptError("receipt_malformed", "header is not an object")
    alg = str(header.get("alg") or "")
    kid = str(header.get("kid") or "").strip()
    if alg != "RS256":
        # Fail closed on every non-RS256 alg, including "none".
        raise ReceiptError("receipt_alg_unsupported", f"alg={alg!r}")
    key = keys.get(kid)
    if key is None:
        raise ReceiptError("receipt_unknown_kid", f"kid={kid!r}")
    signature = _b64url_decode(parts[2])
    signing_input = (parts[0] + "." + parts[1]).encode("ascii")
    try:
        key.verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature as exc:
        raise ReceiptError("receipt_signature_invalid", "signature check failed") from exc
    try:
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReceiptError("receipt_malformed", f"payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReceiptError("receipt_malformed", "payload is not an object")
    version = str(payload.get("crv") or "")
    if version != RECEIPT_VERSION:
        raise ReceiptError("receipt_version_unsupported", f"crv={version!r}")
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        raise ReceiptError("receipt_malformed", "no artifact binding")
    try:
        size_bytes = int(artifact.get("size_bytes") or 0)
    except (TypeError, ValueError) as exc:
        raise ReceiptError("receipt_malformed", f"numeric claim: {exc}") from exc
    return Receipt(
        version=version,
        family=str(payload.get("family") or ""),
        compiled_graph_key=str(payload.get("compiled_graph_key") or ""),
        owning_endpoint_id=str(payload.get("owning_endpoint_id") or ""),
        publisher_tier=_normalize_publisher_tier(payload.get("publisher_tier")),
        publisher_org_id=str(payload.get("publisher_org_id") or ""),
        snapshot_digest=str(payload.get("snapshot_digest") or ""),
        artifact_digest=canonical_artifact_digest(str(artifact.get("digest") or "")),
        artifact_size_bytes=size_bytes,
    )


# -- hub fetches ------------------------------------------------------------


def _fetch_jwks(cfg: _Config) -> Dict[str, rsa.RSAPublicKey]:
    resp = requests.get(cfg.base_url + JWKS_PATH, timeout=_HTTP_TIMEOUT_S)
    if resp.status_code != 200:
        raise ReceiptError("jwks_unavailable", f"{JWKS_PATH} -> {resp.status_code}")
    try:
        doc = resp.json()
    except ValueError as exc:
        raise ReceiptError("jwks_unavailable", f"jwks parse: {exc}") from exc
    keys: Dict[str, rsa.RSAPublicKey] = {}
    for jwk in doc.get("keys") or []:
        if not isinstance(jwk, dict):
            continue
        kid = str(jwk.get("kid") or "").strip()
        if not kid:
            continue
        key = _rsa_key_from_jwk(jwk)
        if key is not None:
            keys[kid] = key
    if not keys:
        raise ReceiptError("jwks_unavailable", "no usable RSA keys in the hub JWKS")
    return keys


def _jwks_for(cfg: _Config, kid_hint: str) -> Dict[str, rsa.RSAPublicKey]:
    """The cached JWKS; refetched when the hinted kid is unknown (rotation)."""
    with _LOCK:
        cached = dict(cfg.jwks)
    if cached and (not kid_hint or kid_hint in cached):
        return cached
    fresh = _fetch_jwks(cfg)
    with _LOCK:
        cfg.jwks = dict(fresh)
    return fresh


def _kid_of(jws: str) -> str:
    parts = str(jws or "").split(".")
    if len(parts) != 3:
        return ""
    try:
        header = json.loads(_b64url_decode(parts[0]).decode("utf-8"))
    except (ReceiptError, ValueError, UnicodeDecodeError):
        return ""
    return str(header.get("kid") or "").strip() if isinstance(header, dict) else ""


def _fetch_receipt_jws(cfg: _Config, digest: str, compiled_graph_key: str) -> str:
    """Fetch the signed receipt for one artifact.

    The lookup key is the ALGORITHM-TAGGED digest — never bare hex, which
    silently acquires whatever algorithm the reader assumed. There is no
    per-algorithm 404-and-retry chain: a silent downgrade would make "which
    digest armed this cell?" unanswerable, and a 404 from a proxy is not a 404
    from the hub.

    Parent-mediated when the split is on (the child holds no worker JWT); the
    identical GET otherwise.
    """
    params: Dict[str, Any] = {
        "compiled_graph_key": compiled_graph_key,
        "artifact_digest": digest,
    }
    resp = broker.request(
        "GET",
        RECEIPT_PATH,
        base_url=cfg.base_url,
        bearer=cfg.worker_jwt(),
        params=params,
        timeout=_HTTP_TIMEOUT_S,
    )
    if resp.status_code == 404:
        raise ReceiptError(
            "receipt_not_found",
            f"no hub receipt for {digest[:23]} key={compiled_graph_key}",
        )
    if resp.status_code != 200:
        raise ReceiptError("receipt_fetch_failed", f"{RECEIPT_PATH} -> {resp.status_code}")
    try:
        body = resp.json()
    except ValueError as exc:
        raise ReceiptError("receipt_fetch_failed", f"response parse: {exc}") from exc
    jws = str(body.get("receipt") or "").strip()
    if not jws:
        raise ReceiptError("receipt_fetch_failed", "empty receipt in response")
    return jws


def _fetch_revocations(cfg: _Config) -> Set[Tuple[str, str]]:
    resp = broker.request(
        "GET",
        REVOCATIONS_PATH,
        base_url=cfg.base_url,
        bearer=cfg.worker_jwt(),
        timeout=_HTTP_TIMEOUT_S,
    )
    if resp.status_code != 200:
        # Fail closed: an unreadable revocation list means the recall
        # channel is down — refusing costs one self-mint, trusting could
        # arm a recalled cell.
        raise ReceiptError("revocations_unavailable", f"{REVOCATIONS_PATH} -> {resp.status_code}")
    try:
        body = resp.json()
    except ValueError as exc:
        raise ReceiptError("revocations_unavailable", f"response parse: {exc}") from exc
    out: Set[Tuple[str, str]] = set()
    for entry in body.get("revoked") or []:
        if isinstance(entry, dict):
            key = str(entry.get("compiled_graph_key") or "").strip()
            digest = str(entry.get("snapshot_digest") or "").strip()
            if key and digest:
                out.add((key, digest))
    return out


# -- local bindings ---------------------------------------------------------


def artifact_digest(path: Path) -> str:
    """This worker's ALGORITHM-TAGGED digest of ``path`` (``sha256:<hex>``)."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return f"{ARTIFACT_DIGEST_ALGORITHM}:{hasher.hexdigest()}"


def _embedded_meta(artifact: Path) -> Dict[str, object]:
    """The artifact's packed ``metadata.json``, through the one stdlib-only
    reader — so this module still never imports the compile stack."""
    try:
        return dict(artifact_meta.read_metadata(artifact))
    except artifact_meta.ArtifactMetadataError as exc:
        raise ReceiptError("artifact_unreadable", f"{artifact.name}: {exc}") from exc


def verify_delivered_artifact(artifact: Path, family: str) -> Receipt:
    """Full verification of one hub-delivered artifact. Raises
    :class:`ReceiptError` (named reason) on any failure; returns the
    verified receipt on success.

    Chain of trust: receipt signature (hub key via JWKS) -> local bytes
    (the receipt's OWN algorithm + integral size) -> embedded metadata
    (inside the digested bytes) -> ``meta.compiled_graph_key == receipt.compiled_graph_key`` ->
    **the PUBLISHER** (platform tier, or this pod's own endpoint/org) ->
    the runtime's own computed key (enforced downstream by the selection
    brain).

    The publisher link is last because it is the only one that asks a question
    about US rather than about the bytes: every other link proves the artifact
    is the one the hub signed, none asks whether it was signed for this pod.
    """
    with _LOCK:
        cfg = _CONFIG
    if cfg is None:
        raise ReceiptError("gate_unconfigured", "receipt gate has no hub wiring")

    artifact = Path(artifact)
    meta = _embedded_meta(artifact)
    meta_key = str(meta.get("compiled_graph_key") or "").strip()
    meta_family = str(meta.get("family") or "").strip()
    if not meta_key:
        raise ReceiptError(
            "artifact_unkeyed",
            f"{artifact.name} metadata has no compiled_graph_key")

    local = artifact_digest(artifact)
    size = artifact.stat().st_size

    jws = _fetch_receipt_jws(cfg, local, meta_key)
    receipt = verify_receipt_jws(jws, _jwks_for(cfg, _kid_of(jws)))

    # Both sides are ALGORITHM-TAGGED and `canonical_artifact_digest` has
    # already refused any tag this worker cannot recompute, so this is one
    # comparison with no untagged branch to fall into and nothing compared
    # against an empty string.
    if local != receipt.artifact_digest:
        raise ReceiptError(
            "receipt_digest_mismatch",
            f"receipt={receipt.artifact_digest[:23]} local={local[:23]}")
    if receipt.artifact_size_bytes != size:
        raise ReceiptError(
            "receipt_size_mismatch",
            f"receipt={receipt.artifact_size_bytes} local={size}")
    if receipt.compiled_graph_key != meta_key:
        raise ReceiptError(
            "receipt_key_mismatch",
            f"receipt={receipt.compiled_graph_key} artifact={meta_key}")
    want_family = str(family or "").strip()
    if want_family and receipt.family != want_family:
        raise ReceiptError(
            "receipt_family_mismatch",
            f"receipt={receipt.family} arming={want_family}")
    if meta_family and receipt.family != meta_family:
        raise ReceiptError(
            "receipt_family_mismatch",
            f"receipt={receipt.family} artifact={meta_family}")
    if not receipt.snapshot_digest:
        raise ReceiptError("receipt_unbound", "no snapshot_digest (derivation binding)")

    if (receipt.compiled_graph_key, receipt.snapshot_digest) in _fetch_revocations(cfg):
        raise ReceiptError(
            "compiled_graph_revoked",
            f"key={receipt.compiled_graph_key} snapshot={receipt.snapshot_digest} is recalled")

    # LAST: who published this, and may we run it. Deliberately after the
    # signature — the tier is only meaningful once the claims are proven — and
    # deliberately before the return, because the caller's next act is to arm
    # the cell and dlopen it.
    if not needs_viewer_identity(receipt):
        # A platform-tier cell is adoptable by every pod, so nothing here has
        # to know who we are — and asking anyway would make a seam hiccup
        # refuse the one class that never needed an identity.
        #
        # pgw#1271: this branch used to call `refuse_untrusted_publisher(
        # receipt, "", "")` here — INSIDE that callee's own early-return
        # condition (`if not needs_viewer_identity(receipt): return`), so it
        # executed exactly zero checks, immediately before the caller dlopens
        # the artifact. A call that reads as the last gate before native-code
        # execution and is structurally incapable of refusing is worse than no
        # call: it answers the reader's question wrongly. What actually clears
        # this tier is the SIGNATURE, and it ran above.
        return receipt
    try:
        viewer = _self_viewer()
    except worker_identity.IdentityUnavailable as exc:
        # Fail CLOSED, and say WHICH failure this is: `identity_unavailable`
        # (nobody could be asked) is a wiring defect on our side, while
        # `publisher_untrusted` (we asked and the answer refuses) is a trust
        # decision. They must never share a reason string.
        raise ReceiptError(
            "identity_unavailable",
            f"this pod cannot be named by anything in reach ({exc}), so no "
            f"org-tier cell can be shown to be adoptable here") from exc
    refuse_untrusted_publisher(receipt, viewer.endpoint_id, viewer.org_id)

    return receipt


def gate_delivered_artifact(artifact: Path, family: str) -> bool:
    """The one arming hook (called from ``models.provision.enable_compiled``
    for every non-None delivered artifact). True = arm may proceed.

    Unconfigured (cozy-local, CLI, unit rigs): no-op True. Configured
    (fleet workers, armed at HelloAck): full verification; ANY failure
    emits the typed ``compiled_graph_receipt_refused`` wire event and returns False —
    the caller drops the delivered artifact and the ordinary miss policy
    (self-mint) takes over. Never raises; never kills serving.
    """
    if not configured():
        return True
    try:
        receipt = verify_delivered_artifact(Path(artifact), family)
    except ReceiptError as exc:
        logger.error(
            "receipts: REFUSING delivered artifact %s (%s)", Path(artifact).name, exc)
        activity_mod.emit_event(
            "compiled_graph_receipt_refused",
            f"family={family} artifact={Path(artifact).name}: {exc}",
            phase=exc.reason,
        )
        return False
    except Exception as exc:  # noqa: BLE001 — refuse, never crash the arm path
        logger.error(
            "receipts: REFUSING delivered artifact %s (unexpected %s: %s)",
            Path(artifact).name, type(exc).__name__, exc)
        activity_mod.emit_event(
            "compiled_graph_receipt_refused",
            f"family={family} artifact={Path(artifact).name}: "
            f"{type(exc).__name__}: {exc}",
            phase="internal_error",
        )
        return False
    logger.info(
        "receipts: verified %s (key=%s, kid-signed, snapshot=%s)",
        Path(artifact).name, receipt.compiled_graph_key, receipt.snapshot_digest[:16])
    return True


__all__ = [
    "ARTIFACT_DIGEST_ALGORITHM",
    "COMPILED_GRAPH_PUBLISHER_TIER_ORG",
    "COMPILED_GRAPH_PUBLISHER_TIER_PLATFORM",
    "Receipt",
    "ReceiptError",
    "artifact_digest",
    "canonical_artifact_digest",
    "configure",
    "configured",
    "gate_delivered_artifact",
    "needs_viewer_identity",
    "refuse_untrusted_publisher",
    "reset",
    "verify_delivered_artifact",
    "verify_receipt_jws",
]
