"""Hub-signed compiled graph receipt verification."""

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

RECEIPT_VERSION = "compiled-graph-receipt-v1"

COMPILED_GRAPH_PUBLISHER_TIER_PLATFORM = "platform"
COMPILED_GRAPH_PUBLISHER_TIER_ORG = "org"
ARTIFACT_DIGEST_ALGORITHM = "sha256"
JWKS_PATH = "/api/v1/artifacts/.well-known/jwks.json"
RECEIPT_PATH = "/v1/worker/compiled-graphs/receipt"
REVOCATIONS_PATH = "/v1/worker/compiled-graphs/revocations"

_HTTP_TIMEOUT_S = 30


class ReceiptError(RuntimeError):
    """Typed receipt refusal."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class Receipt:
    """Decoded, signature-verified compiled graph receipt claims — the ones something CHECKS."""

    version: str
    family: str
    compiled_graph_key: str
    owning_endpoint_id: str
    publisher_tier: str
    publisher_org_id: str
    snapshot_digest: str
    artifact_digest: str
    artifact_size_bytes: int


@dataclass
class _Config:
    base_url: str
    worker_jwt: Callable[[], str]
    jwks: Dict[str, rsa.RSAPublicKey] = field(default_factory=dict)


_LOCK = threading.Lock()
_CONFIG: Optional[_Config] = None
_LOCAL_TRUST: str = ""

POSTURE_ARMED = "armed"
POSTURE_LOCAL = "local"
POSTURE_UNSET = "unset"


def configure(base_url: str, worker_jwt: Callable[[], str]) -> None:
    """Arm the receipt gate against the hub."""
    global _CONFIG, _LOCAL_TRUST
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise ReceiptError(
            "gate_unconfigurable",
            "receipts.configure() was given no base URL; a hub session that "
            "names no file API cannot arm the receipt gate, and pretending it "
            "did is how the gate came to be silently open")
    with _LOCK:
        _CONFIG = _Config(base_url=base, worker_jwt=worker_jwt)
        _LOCAL_TRUST = ""
    logger.info("receipts: gate configured against %s", base)


def trust_local_store(reason: str) -> None:
    """Declare that this process's compiled-graph store is USER-CONTROLLED, so hub-signed receipts do not apply to it."""
    global _LOCAL_TRUST, _CONFIG
    why = str(reason or "").strip()
    if not why:
        raise ReceiptError(
            "local_trust_unattributed",
            "trust_local_store() needs a reason — an anonymous decision to arm "
            "unverified compiled graphs is the fail-open this parameter exists "
            "to abolish")
    with _LOCK:
        _LOCAL_TRUST = why
        _CONFIG = None
    logger.warning(
        "receipts: gate DISARMED for a user-controlled store (%s) — delivered "
        "artifacts are armed without a hub-signed receipt here", why)


def posture() -> str:
    """:data:`POSTURE_ARMED` / :data:`POSTURE_LOCAL` / :data:`POSTURE_UNSET`."""
    with _LOCK:
        if _CONFIG is not None:
            return POSTURE_ARMED
        return POSTURE_LOCAL if _LOCAL_TRUST else POSTURE_UNSET


def reset() -> None:
    """Back to the DEFAULT posture — unset, i.e."""
    global _CONFIG, _LOCAL_TRUST
    with _LOCK:
        _CONFIG = None
        _LOCAL_TRUST = ""


def _b64url_decode(segment: str) -> bytes:
    pad = "=" * (-len(segment) % 4)
    try:
        return base64.urlsafe_b64decode(segment + pad)
    except (binascii.Error, ValueError) as exc:
        raise ReceiptError("receipt_malformed", f"base64url decode failed: {exc}") from exc


def _normalize_publisher_tier(raw: object) -> str:
    if str(raw or "").strip() == COMPILED_GRAPH_PUBLISHER_TIER_PLATFORM:
        return COMPILED_GRAPH_PUBLISHER_TIER_PLATFORM
    return COMPILED_GRAPH_PUBLISHER_TIER_ORG


def _self_viewer() -> "worker_identity.ViewerIdentity":
    return worker_identity.viewer()


def needs_viewer_identity(receipt: Receipt) -> bool:
    """Whether the trust rule will consult WHO THIS POD IS."""
    return receipt.publisher_tier != COMPILED_GRAPH_PUBLISHER_TIER_PLATFORM


def refuse_untrusted_publisher(
    receipt: Receipt, self_endpoint_id: str, self_org_id: str = "",
) -> None:
    """Raise unless this pod may adopt ``receipt``'s compiled graph."""
    if not needs_viewer_identity(receipt):
        return
    owner = str(receipt.owning_endpoint_id or "").strip()
    mine = str(self_endpoint_id or "").strip()
    owner_org = str(receipt.publisher_org_id or "").strip()
    my_org = str(self_org_id or "").strip()

    if owner and mine and owner == mine:
        return
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
            "this pod cannot name its own endpoint or org (no graph_read_* claim "
            "on the worker credential), so it may adopt platform-tier compiled graphs only")
    raise ReceiptError(
        "publisher_untrusted",
        f"compiled graph was minted for endpoint {owner or '<unnamed>'} "
        f"(org {owner_org or '<unnamed>'}) and this pod serves "
        f"{mine or '<unnamed>'} (org {my_org or '<unnamed>'})")


MAX_RSA_MODULUS_BITS = 8192


def _rsa_key_from_jwk(jwk: Mapping[str, object]) -> Optional[rsa.RSAPublicKey]:
    if str(jwk.get("kty") or "") != "RSA":
        return None
    n_raw, e_raw = str(jwk.get("n") or ""), str(jwk.get("e") or "")
    if not n_raw or not e_raw:
        return None
    n_bytes, e_bytes = _b64url_decode(n_raw), _b64url_decode(e_raw)
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
    """Resolve the receipt's algorithm-tagged artifact digest."""
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
    """Verify a compact JWS against ``keys`` (kid -> RSA public key) and return the decoded claims."""
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


def artifact_digest(path: Path) -> str:
    """This worker's ALGORITHM-TAGGED digest of ``path`` (``sha256:<hex>``)."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return f"{ARTIFACT_DIGEST_ALGORITHM}:{hasher.hexdigest()}"


def _embedded_meta(artifact: Path) -> Dict[str, object]:
    try:
        return dict(artifact_meta.read_metadata(artifact))
    except artifact_meta.ArtifactMetadataError as exc:
        raise ReceiptError("artifact_unreadable", f"{artifact.name}: {exc}") from exc


def verify_delivered_artifact(artifact: Path, family: str) -> Receipt:
    """Full verification of one hub-delivered artifact."""
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

    if not needs_viewer_identity(receipt):
        # A platform-tier compiled graph is adoptable by every pod; what clears this tier is the SIGNATURE, which already ran above. Do NOT add a refuse_untrusted_publisher(receipt, "", "") call here: it lands inside that callee's own early return and executes ZERO checks immediately before the caller dlopens the artifact — a call that reads as the last gate before native-code execution and cannot refuse is worse than no call.
        return receipt
    try:
        viewer = _self_viewer()
    except worker_identity.IdentityUnavailable as exc:
        raise ReceiptError(
            "identity_unavailable",
            f"this pod cannot be named by anything in reach ({exc}), so no "
            f"org-tier compiled graph can be shown to be adoptable here") from exc
    refuse_untrusted_publisher(receipt, viewer.endpoint_id, viewer.org_id)

    return receipt


def gate_delivered_artifact(artifact: Path, family: str) -> bool:
    """The one arming hook (called from ``models.provision.enable_compiled`` for every non-None delivered artifact)."""
    if posture() == POSTURE_LOCAL:
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
    "POSTURE_ARMED",
    "POSTURE_LOCAL",
    "POSTURE_UNSET",
    "Receipt",
    "ReceiptError",
    "artifact_digest",
    "canonical_artifact_digest",
    "configure",
    "gate_delivered_artifact",
    "needs_viewer_identity",
    "posture",
    "refuse_untrusted_publisher",
    "reset",
    "trust_local_store",
    "verify_delivered_artifact",
    "verify_receipt_jws",
]
