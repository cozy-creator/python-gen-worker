"""Hub-signed cell receipt verification (pgw#709, build-systems review R1/R2).

A ``cell_store`` row (cell_key -> artifact) is a Nix *realisation*: fetch
verifies the BYTES against the hub's recorded digest, but nothing ever
signed the RECORD binding the key to those bytes. Under "bucket is truth,
DB is a rebuildable index" (th#659) that made bucket write access
equivalent to arbitrary cell delivery after any index rebuild.

The hub now signs a ``cell-receipt-v1`` compact JWS at publish-finalize
binding: cell_key + the hub-ATTESTED axes + owning endpoint + the snapshot
digest (the derivation binding Nix's fingerprint famously omits) + the
packed tarball's ALGORITHM-TAGGED digest AND integral size (Bazel REv2:
size is part of the digest). This module is the WORKER half: before arming any hub-delivered
artifact the worker fetches the receipt, verifies the signature against
the hub's public artifact-signing JWKS, checks every binding against the
local bytes, and re-checks the operator revocation list (R2's targeted
recall — the env_seal ``epoch`` salt is too broad for a single bad cell).

Refusal semantics: a failed receipt DISCARDS the delivered artifact with a
loud typed ``cell_receipt_refused`` activity event and falls through to
the ordinary miss policy (fleet workers self-mint their own replacement —
their own bytes need no receipt; the copy they publish gets one from the
th#910 gate). A receipt failure never kills serving.

th#1303/pgw#807 — SHA-256 ONLY. The receipt's canonical binding is
``artifact.digest``, always tagged (``sha256:<hex>``), and verification
dispatches on that tag. An untagged bare-hex digest is REFUSED rather than
read as some assumed algorithm, and a receipt with no usable digest at all is
REFUSED rather than compared against nothing — the empty compare is this
migration's signature defect. The legacy bare-hex ``artifact.blake3`` arm is
GONE with the v1 publish protocol that minted it: a cell it could describe
can no longer be republished under v1 anyway, so a worker meeting one refuses
it, self-mints, and publishes a replacement whose receipt is sha256-bound —
the designed miss policy, not a new failure.

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
import tarfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Set, Tuple

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from . import activity as activity_mod
from .procsplit import broker

logger = logging.getLogger(__name__)

RECEIPT_VERSION = "cell-receipt-v1"
# The algorithms this worker can actually recompute from local bytes. A
# receipt naming anything else is refused, never assumed.
ARTIFACT_DIGEST_ALGORITHMS = ("sha256",)
JWKS_PATH = "/api/v1/artifacts/.well-known/jwks.json"
RECEIPT_PATH = "/v1/worker/cells/receipt"
REVOCATIONS_PATH = "/v1/worker/cells/revocations"

_HTTP_TIMEOUT_S = 30
_METADATA_NAME = "metadata.json"


class ReceiptError(RuntimeError):
    """Typed receipt refusal. ``reason`` is the stable, greppable class."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class Receipt:
    """Decoded, signature-verified cell receipt claims."""

    version: str
    family: str
    cell_key: str
    axes: Dict[str, str]
    owning_endpoint_id: str
    publisher: str
    snapshot_digest: str
    artifact_path: str
    # Canonical, ALGORITHM-TAGGED ("<algo>:<hex>"). Never bare hex.
    artifact_digest: str
    artifact_size_bytes: int
    manifest_digest: str
    fingerprint_digest: str
    issued_at_unix: int


@dataclass
class _Config:
    base_url: str
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


def _rsa_key_from_jwk(jwk: Mapping[str, object]) -> Optional[rsa.RSAPublicKey]:
    if str(jwk.get("kty") or "") != "RSA":
        return None
    n_raw, e_raw = str(jwk.get("n") or ""), str(jwk.get("e") or "")
    if not n_raw or not e_raw:
        return None
    n = int.from_bytes(_b64url_decode(n_raw), "big")
    e = int.from_bytes(_b64url_decode(e_raw), "big")
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
    if algo not in ARTIFACT_DIGEST_ALGORITHMS:
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
    axes_raw = payload.get("axes")
    axes: Dict[str, str] = {}
    if isinstance(axes_raw, dict):
        axes = {str(k): str(v) for k, v in axes_raw.items()}
    try:
        size_bytes = int(artifact.get("size_bytes") or 0)
        issued_at = int(payload.get("iat") or 0)
    except (TypeError, ValueError) as exc:
        raise ReceiptError("receipt_malformed", f"numeric claim: {exc}") from exc
    return Receipt(
        version=version,
        family=str(payload.get("family") or ""),
        cell_key=str(payload.get("cell_key") or ""),
        axes=axes,
        owning_endpoint_id=str(payload.get("owning_endpoint_id") or ""),
        publisher=str(payload.get("publisher") or ""),
        snapshot_digest=str(payload.get("snapshot_digest") or ""),
        artifact_path=str(artifact.get("path") or ""),
        artifact_digest=canonical_artifact_digest(str(artifact.get("digest") or "")),
        artifact_size_bytes=size_bytes,
        manifest_digest=str(payload.get("manifest_digest") or ""),
        fingerprint_digest=str(payload.get("fingerprint_digest") or ""),
        issued_at_unix=issued_at,
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


def _fetch_receipt_jws(cfg: _Config, digests: Mapping[str, str], cell_key: str) -> str:
    """Fetch the signed receipt for one artifact.

    The lookup key is the ALGORITHM-TAGGED digest. It stays a list-valued
    param — one request carrying every digest this worker computed, matched
    hub-side — rather than a 404-and-retry chain per algorithm: a silent
    per-algorithm downgrade would make "which digest armed this cell?"
    unanswerable, and th#715 says a 404 from a proxy is not a 404 from the
    hub. Today the list has one member; the SHAPE is what keeps a future
    algorithm from needing a new round-trip protocol.

    pgw#763 delta 1: parent-mediated when the split is on (the child holds no
    worker JWT); the identical GET otherwise. The repeated ``artifact_digest``
    key rides as a list value — requests encodes that as repeated params, and
    the seam's action table allowlists it the same way.
    """
    params: Dict[str, Any] = {
        "cell_key": cell_key,
        "artifact_digest": [f"{algo}:{hex_digest}" for algo, hex_digest in digests.items()],
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
        offered = ",".join(f"{a}:{h[:16]}" for a, h in sorted(digests.items()))
        raise ReceiptError(
            "receipt_not_found",
            f"no hub receipt for {offered} key={cell_key}",
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
            key = str(entry.get("cell_key") or "").strip()
            digest = str(entry.get("snapshot_digest") or "").strip()
            if key and digest:
                out.add((key, digest))
    return out


# -- local bindings ---------------------------------------------------------


def artifact_digests(path: Path, algorithms: Iterable[str] = ARTIFACT_DIGEST_ALGORITHMS) -> Dict[str, str]:
    """Every digest this worker can offer for ``path``, in ONE read pass.

    One algorithm today (sha256, pgw#807): the map SHAPE survives so a second
    one is a table entry rather than a rewrite of the fetch and the compare.
    """
    hashers: Dict[str, object] = {}
    for algo in algorithms:
        if algo == "sha256":
            hashers[algo] = hashlib.sha256()
        else:
            raise ReceiptError("receipt_digest_algorithm_unsupported", f"algo={algo!r}")
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            for hasher in hashers.values():
                hasher.update(chunk)  # type: ignore[attr-defined]
    return {algo: str(h.hexdigest()) for algo, h in hashers.items()}  # type: ignore[attr-defined]


def _embedded_meta(artifact: Path) -> Dict[str, object]:
    """The artifact's packed ``metadata.json`` — read directly (stdlib
    tarfile) so this module never imports the compile stack."""
    try:
        with tarfile.open(artifact, mode="r:*") as tar:
            for member in tar:
                if member.name == _METADATA_NAME and member.isfile():
                    f = tar.extractfile(member)
                    if f is None:
                        break
                    loaded = json.loads(f.read().decode("utf-8"))
                    return dict(loaded) if isinstance(loaded, dict) else {}
    except (OSError, tarfile.TarError, ValueError, UnicodeDecodeError) as exc:
        raise ReceiptError("artifact_unreadable", f"{artifact.name}: {exc}") from exc
    raise ReceiptError("artifact_unreadable", f"{artifact.name}: no {_METADATA_NAME}")


def verify_delivered_artifact(artifact: Path, family: str) -> Receipt:
    """Full verification of one hub-delivered artifact. Raises
    :class:`ReceiptError` (named reason) on any failure; returns the
    verified receipt on success.

    Chain of trust: receipt signature (hub key via JWKS) -> local bytes
    (the receipt's OWN algorithm + integral size) -> embedded metadata
    (inside the digested bytes) -> ``meta.cell_key == receipt.cell_key`` ->
    the runtime's own computed key (enforced downstream by the th#883
    selection brain).
    """
    with _LOCK:
        cfg = _CONFIG
    if cfg is None:
        raise ReceiptError("gate_unconfigured", "receipt gate has no hub wiring")

    artifact = Path(artifact)
    meta = _embedded_meta(artifact)
    meta_key = str(meta.get("cell_key") or "").strip()
    meta_family = str(meta.get("family") or "").strip()
    if not meta_key:
        raise ReceiptError("artifact_unkeyed", f"{artifact.name} metadata has no cell_key")

    digests = artifact_digests(artifact)
    size = artifact.stat().st_size

    jws = _fetch_receipt_jws(cfg, digests, meta_key)
    receipt = verify_receipt_jws(jws, _jwks_for(cfg, _kid_of(jws)))

    # Dispatch on the receipt's OWN algorithm tag. Hardcoding one here is how
    # a sha256-bound cell gets checked with blake3 and every honest artifact
    # looks corrupt; `canonical_artifact_digest` has already guaranteed the
    # tag is present and supported, so there is no untagged branch to fall
    # into and nothing compares against an empty string.
    algo, _, want = receipt.artifact_digest.partition(":")
    local = digests.get(algo, "")
    if not local:
        raise ReceiptError(
            "receipt_digest_algorithm_unsupported",
            f"receipt binds {algo} which this worker did not compute")
    if local != want:
        raise ReceiptError(
            "receipt_digest_mismatch",
            f"{algo} receipt={want[:16]} local={local[:16]}")
    if receipt.artifact_size_bytes != size:
        raise ReceiptError(
            "receipt_size_mismatch",
            f"receipt={receipt.artifact_size_bytes} local={size}")
    if receipt.cell_key != meta_key:
        raise ReceiptError(
            "receipt_key_mismatch",
            f"receipt={receipt.cell_key} artifact={meta_key}")
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

    if (receipt.cell_key, receipt.snapshot_digest) in _fetch_revocations(cfg):
        raise ReceiptError(
            "cell_revoked",
            f"key={receipt.cell_key} snapshot={receipt.snapshot_digest} is recalled")

    return receipt


def gate_delivered_artifact(artifact: Path, family: str) -> bool:
    """The one arming hook (called from ``models.provision.enable_compiled``
    for every non-None delivered artifact). True = arm may proceed.

    Unconfigured (cozy-local, CLI, unit rigs): no-op True. Configured
    (fleet workers, armed at HelloAck): full verification; ANY failure
    emits the typed ``cell_receipt_refused`` wire event and returns False —
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
            "cell_receipt_refused",
            f"family={family} artifact={Path(artifact).name}: {exc}",
            phase=exc.reason,
        )
        return False
    except Exception as exc:  # noqa: BLE001 — refuse, never crash the arm path
        logger.error(
            "receipts: REFUSING delivered artifact %s (unexpected %s: %s)",
            Path(artifact).name, type(exc).__name__, exc)
        activity_mod.emit_event(
            "cell_receipt_refused",
            f"family={family} artifact={Path(artifact).name}: "
            f"{type(exc).__name__}: {exc}",
            phase="internal_error",
        )
        return False
    logger.info(
        "receipts: verified %s (key=%s, kid-signed, snapshot=%s)",
        Path(artifact).name, receipt.cell_key, receipt.snapshot_digest[:16])
    return True


__all__ = [
    "ARTIFACT_DIGEST_ALGORITHMS",
    "Receipt",
    "ReceiptError",
    "artifact_digests",
    "canonical_artifact_digest",
    "configure",
    "configured",
    "gate_delivered_artifact",
    "reset",
    "verify_delivered_artifact",
    "verify_receipt_jws",
]
