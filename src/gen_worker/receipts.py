"""Verify hub-signed compiled-graph receipts before bytes can execute.

Resolve embeds one compact JWS per compiled graph.  The worker verifies that
receipt before transport, then verifies the materialized bytes against the
same typed receipt before TCG imports them.  There is no receipt lookup,
revocation lookup, legacy claim spelling, or unsigned projection of claims.

The signed object is deliberately the compiled graph, never a resolve batch or
manifest.  Its identity axes derive its key through TCG's authority, while the
artifact digest and integral size bind the exact transported bytes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from torch_compiled_graphs import IdentityError
from torch_compiled_graphs.identity import from_artifact_metadata, from_axes

from . import artifact_meta, worker_identity

logger = logging.getLogger(__name__)

RECEIPT_VERSION = "compiled-graph-receipt-v1"
RECEIPT_TYPE = "compiled-graph-receipt-v1+jws"
PUBLISHER_TIER_PLATFORM = "platform"
PUBLISHER_TIER_ORG = "org"
ARTIFACT_DIGEST_ALGORITHM = "sha256"
JWKS_PATH = "/api/v1/artifacts/.well-known/jwks.json"
# A JWKS response is constant-size control data: 30 seconds bounds a silent
# socket without guessing how long compilation, transfer, or import may take.
_HTTP_TIMEOUT_S = 30
# Production RSA signing keys are 2048-4096 bits. 8192 accepts a full size
# class of rotation headroom while refusing modulus/exponent inputs large
# enough to make every cached-key verification super-linear.
MAX_RSA_MODULUS_BITS = 8192

_CLAIM_FIELDS = frozenset({
    "crv",
    "family",
    "compiled_graph_key",
    "identity_axes",
    "owning_endpoint_id",
    "publisher",
    "publisher_tier",
    "publisher_org_id",
    "snapshot_digest",
    "artifact",
    "iat",
})
_ARTIFACT_FIELDS = frozenset({"path", "digest", "size_bytes"})


class ReceiptError(RuntimeError):
    """Typed refusal whose ``reason`` is stable wire/debug vocabulary."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True, slots=True)
class Receipt:
    """Every decoded claim is verified and consumed by an admission gate."""

    version: str
    family: str
    compiled_graph_key: str
    identity_axes: tuple[tuple[str, str], ...]
    owning_endpoint_id: str
    publisher: str
    publisher_tier: str
    publisher_org_id: str
    snapshot_digest: str
    artifact_path: str
    artifact_digest: str
    artifact_size_bytes: int
    issued_at_unix: int


@dataclass
class _Config:
    base_url: str
    jwks: dict[str, rsa.RSAPublicKey] = field(default_factory=dict)


_LOCK = threading.Lock()
_CONFIG: Optional[_Config] = None


def configure(base_url: str) -> None:
    """Arm the gate and discard cached keys on hub reconfiguration."""
    global _CONFIG
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return
    with _LOCK:
        _CONFIG = _Config(base)
    logger.info("receipts: gate configured against %s", base)


def _reset_for_tests() -> None:
    """Disarm the gate and its JWKS cache (test/process-reset seam)."""
    global _CONFIG
    with _LOCK:
        _CONFIG = None


def _b64url_decode(segment: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
    except (binascii.Error, ValueError) as exc:
        raise ReceiptError(
            "receipt_malformed", f"base64url decode failed: {exc}"
        ) from exc


def _required_string(block: Mapping[str, Any], name: str) -> str:
    value = block.get(name)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ReceiptError("receipt_claim_missing", f"{name} must be a canonical string")
    return value


def _required_positive_int(
    block: Mapping[str, Any], name: str, reason: str,
) -> int:
    value = block.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReceiptError(reason, f"{name} must be a positive integer")
    return value


def _canonical_artifact_path(raw: object) -> str:
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise ReceiptError(
            "receipt_artifact_path_invalid", "artifact path must be a canonical string"
        )
    value = raw
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ReceiptError(
            "receipt_artifact_path_invalid",
            f"artifact path {value!r} must be canonical and relative",
        )
    return value


def _canonical_artifact_digest(raw: object) -> str:
    if not isinstance(raw, str) or not raw or raw != raw.strip() or raw != raw.lower():
        raise ReceiptError(
            "receipt_digest_malformed", "artifact digest must be a canonical string"
        )
    value = raw
    algorithm, separator, hexadecimal = value.partition(":")
    if not separator:
        raise ReceiptError(
            "receipt_digest_untagged", "artifact digest has no algorithm tag"
        )
    if algorithm != ARTIFACT_DIGEST_ALGORITHM:
        raise ReceiptError(
            "receipt_digest_algorithm_unsupported", f"algorithm={algorithm!r}"
        )
    if len(hexadecimal) != 64 or any(c not in "0123456789abcdef" for c in hexadecimal):
        raise ReceiptError(
            "receipt_digest_malformed", "sha256 digest must be 64 lowercase hex characters"
        )
    return value


def _rsa_key_from_jwk(jwk: Mapping[str, object]) -> Optional[rsa.RSAPublicKey]:
    if jwk.get("kty") != "RSA":
        return None
    modulus = str(jwk.get("n") or "")
    exponent = str(jwk.get("e") or "")
    if not modulus or not exponent:
        return None
    n_bytes = _b64url_decode(modulus)
    e_bytes = _b64url_decode(exponent)
    bits = max(len(n_bytes), len(e_bytes)) * 8
    if bits > MAX_RSA_MODULUS_BITS:
        raise ReceiptError(
            "jwks_modulus_oversized",
            f"key {str(jwk.get('kid') or '<unnamed>')!r} is {bits} bits",
        )
    try:
        return rsa.RSAPublicNumbers(
            e=int.from_bytes(e_bytes, "big"), n=int.from_bytes(n_bytes, "big")
        ).public_key()
    except ValueError as exc:
        raise ReceiptError("jwks_unavailable", f"invalid RSA key: {exc}") from exc


def _fetch_jwks(cfg: _Config) -> dict[str, rsa.RSAPublicKey]:
    try:
        response = requests.get(cfg.base_url + JWKS_PATH, timeout=_HTTP_TIMEOUT_S)
    except requests.RequestException as exc:
        raise ReceiptError("jwks_unavailable", str(exc)) from exc
    if response.status_code != 200:
        raise ReceiptError("jwks_unavailable", f"{JWKS_PATH} -> {response.status_code}")
    try:
        document = response.json()
    except ValueError as exc:
        raise ReceiptError("jwks_unavailable", f"JWKS parse failed: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ReceiptError("jwks_unavailable", "JWKS is not an object")
    keys: dict[str, rsa.RSAPublicKey] = {}
    rows = document.get("keys")
    if not isinstance(rows, list):
        raise ReceiptError("jwks_unavailable", "JWKS carries no key list")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        kid = str(row.get("kid") or "").strip()
        if not kid:
            continue
        key = _rsa_key_from_jwk(row)
        if key is not None:
            keys[kid] = key
    if not keys:
        raise ReceiptError("jwks_unavailable", "JWKS carries no usable RSA key")
    return keys


def _jwks_for(cfg: _Config, kid: str) -> dict[str, rsa.RSAPublicKey]:
    with _LOCK:
        cached = dict(cfg.jwks)
    if cached and kid in cached:
        return cached
    fresh = _fetch_jwks(cfg)
    with _LOCK:
        cfg.jwks = dict(fresh)
    return fresh


def _header(jws: str) -> tuple[list[str], dict[str, Any]]:
    if not isinstance(jws, str) or not jws or jws != jws.strip():
        raise ReceiptError("receipt_malformed", "compact JWS must be a canonical string")
    parts = jws.split(".")
    if len(parts) != 3 or not all(parts):
        raise ReceiptError("receipt_malformed", "not a compact JWS")
    try:
        raw = json.loads(_b64url_decode(parts[0]).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ReceiptError("receipt_malformed", f"header: {exc}") from exc
    if not isinstance(raw, dict):
        raise ReceiptError("receipt_malformed", "header is not an object")
    if set(raw) != {"alg", "kid", "typ"}:
        raise ReceiptError("receipt_header_shape", "header must contain only alg, kid, typ")
    if raw.get("alg") != "RS256":
        raise ReceiptError("receipt_alg_unsupported", f"alg={raw.get('alg')!r}")
    if raw.get("typ") != RECEIPT_TYPE:
        raise ReceiptError("receipt_type_unsupported", f"typ={raw.get('typ')!r}")
    _required_string(raw, "kid")
    return parts, raw


def _verify_jws(
    parts: list[str],
    header: Mapping[str, Any],
    keys: Mapping[str, rsa.RSAPublicKey],
) -> Receipt:
    kid = str(header["kid"])
    key = keys.get(kid)
    if key is None:
        raise ReceiptError("receipt_unknown_kid", f"kid={kid!r}")
    try:
        signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    except UnicodeEncodeError as exc:
        raise ReceiptError("receipt_malformed", "JWS segments must be ASCII") from exc
    try:
        key.verify(
            _b64url_decode(parts[2]),
            signing_input,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
    except InvalidSignature as exc:
        raise ReceiptError("receipt_signature_invalid", "signature check failed") from exc
    try:
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ReceiptError("receipt_malformed", f"payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReceiptError("receipt_malformed", "payload is not an object")
    if set(payload) != _CLAIM_FIELDS:
        missing = sorted(_CLAIM_FIELDS - set(payload))
        unknown = sorted(set(payload) - _CLAIM_FIELDS)
        raise ReceiptError(
            "receipt_claim_shape", f"missing={missing!r} unknown={unknown!r}"
        )
    version = _required_string(payload, "crv")
    if version != RECEIPT_VERSION:
        raise ReceiptError("receipt_version_unsupported", f"crv={version!r}")
    artifact = payload.get("artifact")
    if not isinstance(artifact, Mapping) or set(artifact) != _ARTIFACT_FIELDS:
        raise ReceiptError(
            "receipt_artifact_shape", "artifact must contain exactly path, digest, size_bytes"
        )
    axes = payload.get("identity_axes")
    if not isinstance(axes, Mapping):
        raise ReceiptError("receipt_identity_invalid", "identity_axes is not an object")
    try:
        identity = from_axes(axes)
    except IdentityError as exc:
        raise ReceiptError("receipt_identity_invalid", str(exc)) from exc
    compiled_graph_key = _required_string(payload, "compiled_graph_key")
    if identity.value != compiled_graph_key:
        raise ReceiptError(
            "receipt_identity_mismatch",
            f"identity axes derive {identity.value}, receipt names {compiled_graph_key}",
        )
    publisher_tier = _required_string(payload, "publisher_tier")
    if publisher_tier not in {PUBLISHER_TIER_PLATFORM, PUBLISHER_TIER_ORG}:
        raise ReceiptError("receipt_publisher_tier_invalid", publisher_tier)
    artifact_size = _required_positive_int(
        artifact, "size_bytes", "receipt_artifact_size_invalid"
    )
    issued_at = _required_positive_int(payload, "iat", "receipt_issuance_invalid")
    return Receipt(
        version=version,
        family=_required_string(payload, "family"),
        compiled_graph_key=compiled_graph_key,
        identity_axes=identity.axes,
        owning_endpoint_id=_required_string(payload, "owning_endpoint_id"),
        publisher=_required_string(payload, "publisher"),
        publisher_tier=publisher_tier,
        publisher_org_id=_required_string(payload, "publisher_org_id"),
        snapshot_digest=_required_string(payload, "snapshot_digest"),
        artifact_path=_canonical_artifact_path(artifact.get("path")),
        artifact_digest=_canonical_artifact_digest(artifact.get("digest")),
        artifact_size_bytes=artifact_size,
        issued_at_unix=issued_at,
    )


def _refuse_untrusted_publisher(receipt: Receipt) -> None:
    if receipt.publisher_tier == PUBLISHER_TIER_PLATFORM:
        return
    try:
        viewer = worker_identity.viewer()
    except worker_identity.IdentityUnavailable as exc:
        raise ReceiptError("identity_unavailable", str(exc)) from exc
    if receipt.owning_endpoint_id == viewer.endpoint_id:
        return
    if receipt.publisher_org_id == viewer.org_id and viewer.org_id:
        return
    raise ReceiptError(
        "publisher_untrusted",
        f"publisher endpoint/org {receipt.owning_endpoint_id}/{receipt.publisher_org_id} "
        f"does not match viewer {viewer.endpoint_id}/{viewer.org_id}",
    )


def verify_receipt(
    receipt_jws: str,
    *,
    family: str,
    compiled_graph_key: str,
    snapshot_digest: str,
    artifact_path: str,
    artifact_digest: str,
    artifact_size_bytes: int,
) -> Receipt:
    """Verify every signed resolve/transport binding before download."""
    with _LOCK:
        cfg = _CONFIG
    if cfg is None:
        raise ReceiptError("gate_unconfigured", "receipt gate has no hub wiring")
    parts, header = _header(receipt_jws)
    kid = str(header["kid"])
    receipt = _verify_jws(parts, header, _jwks_for(cfg, kid))
    expected = {
        "family": _required_string({"family": family}, "family"),
        "compiled_graph_key": _required_string(
            {"compiled_graph_key": compiled_graph_key}, "compiled_graph_key"
        ),
        "snapshot_digest": _required_string(
            {"snapshot_digest": snapshot_digest}, "snapshot_digest"
        ),
        "artifact_path": _canonical_artifact_path(artifact_path),
        "artifact_digest": _canonical_artifact_digest(artifact_digest),
    }
    for name, wanted in expected.items():
        if getattr(receipt, name) != wanted:
            raise ReceiptError(
                f"receipt_{name}_mismatch",
                f"receipt={getattr(receipt, name)!r} expected={wanted!r}",
            )
    expected_size = _required_positive_int(
        {"artifact_size_bytes": artifact_size_bytes},
        "artifact_size_bytes",
        "receipt_expected_artifact_size_invalid",
    )
    if receipt.artifact_size_bytes != expected_size:
        raise ReceiptError(
            "receipt_artifact_size_bytes_mismatch",
            f"receipt={receipt.artifact_size_bytes} expected={artifact_size_bytes}",
        )
    _refuse_untrusted_publisher(receipt)
    return receipt


def _artifact_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 << 20):
            digest.update(block)
    return f"{ARTIFACT_DIGEST_ALGORITHM}:{digest.hexdigest()}"


def _embedded_meta(artifact: Path) -> dict[str, object]:
    try:
        return dict(artifact_meta.read_metadata(artifact))
    except artifact_meta.ArtifactMetadataError as exc:
        raise ReceiptError("artifact_unreadable", f"{artifact.name}: {exc}") from exc


def verify_delivered_artifact(
    artifact: Path, family: str, receipt: Receipt,
) -> Receipt:
    """Bind downloaded bytes to a pre-verified receipt before TCG import."""
    if not isinstance(receipt, Receipt):
        raise ReceiptError("receipt_unverified", "expected a verified Receipt object")
    path = Path(artifact)
    expected_family = _required_string({"family": family}, "family")
    if receipt.family != expected_family:
        raise ReceiptError("receipt_family_mismatch", receipt.family)
    size = path.stat().st_size
    if size != receipt.artifact_size_bytes:
        raise ReceiptError(
            "receipt_size_mismatch", f"receipt={receipt.artifact_size_bytes} local={size}"
        )
    local_digest = _artifact_digest(path)
    if local_digest != receipt.artifact_digest:
        raise ReceiptError(
            "receipt_digest_mismatch",
            f"receipt={receipt.artifact_digest[:24]} local={local_digest[:24]}",
        )
    metadata = _embedded_meta(path)
    try:
        identity = from_artifact_metadata(metadata)
    except IdentityError as exc:
        raise ReceiptError("artifact_identity_invalid", str(exc)) from exc
    stamped = str(metadata.get("compiled_graph_key") or "").strip()
    if stamped != identity.value:
        raise ReceiptError(
            "artifact_key_mismatch", f"metadata={stamped!r} derived={identity.value!r}"
        )
    if identity.axes != receipt.identity_axes or identity.value != receipt.compiled_graph_key:
        raise ReceiptError(
            "receipt_identity_mismatch",
            "artifact identity axes do not match the signed receipt",
        )
    return receipt


__all__ = [
    "ARTIFACT_DIGEST_ALGORITHM",
    "JWKS_PATH",
    "PUBLISHER_TIER_ORG",
    "PUBLISHER_TIER_PLATFORM",
    "RECEIPT_TYPE",
    "RECEIPT_VERSION",
    "Receipt",
    "ReceiptError",
    "configure",
    "verify_delivered_artifact",
    "verify_receipt",
]
