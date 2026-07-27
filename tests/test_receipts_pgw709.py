"""pgw#709: cell-receipt verification — REAL signatures, REAL HTTP, no mocks.

The signer here mirrors the hub's production format byte-for-byte (RS256
PKCS1v15/SHA256 compact JWS, kid header, cell-receipt-v1 claims); the hub
half's Go tests pin the same format from the signing side. The tamper cases
are the red verification: each one asserts the gate actually REFUSES.
"""

from __future__ import annotations

import base64
import io
import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from gen_worker import receipts

KID = "test-kid-1"
FAMILY = "sdxl"
CELL_KEY = "ck5-0123456789abcdef0123456789abcdef0123456789abcdef01234567"
SNAPSHOT = "snapdigest-abc123"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def sign_receipt(
    key: rsa.RSAPrivateKey,
    claims: Dict[str, Any],
    *,
    kid: str = KID,
    alg: str = "RS256",
) -> str:
    header = {"alg": alg, "kid": kid, "typ": "cell-receipt-v1+jws"}
    signing_input = (
        _b64url(json.dumps(header).encode()) + "." + _b64url(json.dumps(claims).encode())
    )
    sig = key.sign(signing_input.encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    return signing_input + "." + _b64url(sig)


def make_claims(artifact_blake3: str, size_bytes: int, **overrides: Any) -> Dict[str, Any]:
    claims: Dict[str, Any] = {
        "crv": "cell-receipt-v1",
        "family": FAMILY,
        "cell_key": CELL_KEY,
        "axes": {"sku": "rtx-4090", "image_digest": "sha256:feed", "gen_worker": "0.75.1"},
        "owning_endpoint_id": "3e0f8f7a-1111-2222-3333-444455556666",
        "publisher": "selfmint:worker=w1:pod=p1:release=r1",
        "snapshot_digest": SNAPSHOT,
        "artifact": {"path": "cell.tar.gz", "blake3": artifact_blake3, "size_bytes": size_bytes},
        "manifest_digest": "sha256:aa",
        "fingerprint_digest": "sha256:bb",
        "iat": 1_700_000_000,
    }
    claims.update(overrides)
    return claims


@pytest.fixture()
def rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture()
def pub_map(rsa_key: rsa.RSAPrivateKey) -> Dict[str, rsa.RSAPublicKey]:
    return {KID: rsa_key.public_key()}


def make_artifact(tmp_path: Path, *, cell_key: str = CELL_KEY, family: str = FAMILY) -> Path:
    meta = {"cell_key": cell_key, "family": family, "format": "cozy-compile-cache/v1"}
    target = tmp_path / "cell.tar.gz"
    with tarfile.open(target, "w:gz") as tar:
        raw = json.dumps(meta).encode()
        ti = tarfile.TarInfo("metadata.json")
        ti.size = len(raw)
        tar.addfile(ti, io.BytesIO(raw))
        payload = b"fake-inductor-entry" * 64
        ti = tarfile.TarInfo("inductor/aa/entry.py")
        ti.size = len(payload)
        tar.addfile(ti, io.BytesIO(payload))
    return target


# ---------------------------------------------------------------------------
# Pure JWS verification
# ---------------------------------------------------------------------------


class TestVerifyReceiptJWS:
    def test_round_trip(self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]) -> None:
        jws = sign_receipt(rsa_key, make_claims("ab12", 4096))
        receipt = receipts.verify_receipt_jws(jws, pub_map)
        assert receipt.cell_key == CELL_KEY
        assert receipt.family == FAMILY
        assert receipt.snapshot_digest == SNAPSHOT
        assert receipt.artifact_blake3 == "ab12"
        assert receipt.artifact_size_bytes == 4096
        assert receipt.axes["sku"] == "rtx-4090"

    def test_tampered_payload_refused(self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]) -> None:
        # RED: re-point the signed payload at a different cell key — the
        # poisoning move receipts exist to prevent.
        jws = sign_receipt(rsa_key, make_claims("ab12", 4096))
        head, _, sig = jws.split(".")
        forged_claims = make_claims("ab12", 4096, cell_key="ck5-" + "f" * 56)
        forged = head + "." + _b64url(json.dumps(forged_claims).encode()) + "." + sig
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_receipt_jws(forged, pub_map)
        assert exc.value.reason == "receipt_signature_invalid"

    def test_unknown_kid_refused(self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]) -> None:
        jws = sign_receipt(rsa_key, make_claims("ab12", 4096), kid="rogue-kid")
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_receipt_jws(jws, pub_map)
        assert exc.value.reason == "receipt_unknown_kid"

    def test_alg_downgrade_refused(self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]) -> None:
        jws = sign_receipt(rsa_key, make_claims("ab12", 4096), alg="none")
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_receipt_jws(jws, pub_map)
        assert exc.value.reason == "receipt_alg_unsupported"

    def test_wrong_key_refused(self, rsa_key: rsa.RSAPrivateKey) -> None:
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jws = sign_receipt(rsa_key, make_claims("ab12", 4096))
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_receipt_jws(jws, {KID: other.public_key()})
        assert exc.value.reason == "receipt_signature_invalid"

    def test_wrong_version_refused(self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]) -> None:
        jws = sign_receipt(rsa_key, make_claims("ab12", 4096, crv="cell-receipt-v0"))
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_receipt_jws(jws, pub_map)
        assert exc.value.reason == "receipt_version_unsupported"

    def test_garbage_refused(self, pub_map: Dict[str, rsa.RSAPublicKey]) -> None:
        for junk in ("", "a.b", "not-a-jws", "a.b.c.d"):
            with pytest.raises(receipts.ReceiptError):
                receipts.verify_receipt_jws(junk, pub_map)


# ---------------------------------------------------------------------------
# End-to-end gate over real HTTP (a live hub stub on localhost)
# ---------------------------------------------------------------------------


class HubStub:
    """A real HTTP server speaking the hub's three receipt surfaces."""

    def __init__(self, key: rsa.RSAPrivateKey) -> None:
        self.key = key
        self.receipts: Dict[Tuple[str, str], str] = {}
        self.revoked: List[Dict[str, str]] = []
        self.receipt_status: Optional[int] = None  # force an error status
        pub = key.public_key().public_numbers()

        def b64_int(v: int) -> str:
            raw = v.to_bytes((v.bit_length() + 7) // 8, "big")
            return _b64url(raw)

        self.jwks = {"keys": [{"kty": "RSA", "kid": KID, "use": "sig", "alg": "RS256",
                               "n": b64_int(pub.n), "e": b64_int(pub.e)}]}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # noqa: N802
                pass

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == receipts.JWKS_PATH:
                    self._json(200, stub.jwks)
                elif parsed.path == receipts.RECEIPT_PATH:
                    if stub.receipt_status is not None:
                        self._json(stub.receipt_status, {"error": "forced"})
                        return
                    q = parse_qs(parsed.query)
                    lookup = (q.get("blake3", [""])[0], q.get("cell_key", [""])[0])
                    jws = stub.receipts.get(lookup)
                    if jws is None:
                        self._json(404, {"error": "cell_receipt_not_found"})
                    else:
                        self._json(200, {"receipt": jws, "snapshot_digest": SNAPSHOT})
                elif parsed.path == receipts.REVOCATIONS_PATH:
                    self._json(200, {"revoked": stub.revoked})
                else:
                    self._json(404, {"error": "unknown route"})

            def _json(self, status: int, body: Dict[str, Any]) -> None:
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def serve_receipt_for(self, artifact: Path, **claim_overrides: Any) -> None:
        digest = receipts._blake3_file(artifact)
        size = artifact.stat().st_size
        claims = make_claims(digest, size, **claim_overrides)
        self.receipts[(digest, str(claims["cell_key"]))] = sign_receipt(self.key, claims)


@pytest.fixture()
def hub(rsa_key: rsa.RSAPrivateKey) -> Iterator[HubStub]:
    stub = HubStub(rsa_key)
    yield stub
    stub.close()
    receipts.reset()


def _configure(stub: HubStub) -> None:
    receipts.configure(base_url=stub.base_url, worker_jwt=lambda: "test-worker-jwt")


class TestGateDeliveredArtifact:
    def test_unconfigured_gate_is_noop(self, tmp_path: Path) -> None:
        receipts.reset()
        artifact = make_artifact(tmp_path)
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is True

    def test_verified_artifact_arms(self, tmp_path: Path, hub: HubStub) -> None:
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(artifact)
        _configure(hub)
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is True

    def test_missing_receipt_refused(self, tmp_path: Path, hub: HubStub) -> None:
        # The pre-receipt-cell rollout path: hub has no receipt -> refuse,
        # the miss policy self-mints.
        artifact = make_artifact(tmp_path)
        _configure(hub)
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False

    def test_tampered_bytes_refused(self, tmp_path: Path, hub: HubStub) -> None:
        # RED: receipt minted for the original bytes; artifact then altered
        # (the delivery-substitution attack).
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(artifact)
        original_digest = receipts._blake3_file(artifact)
        with artifact.open("ab") as f:
            f.write(b"\x00poison")
        # Serve the original receipt under the NEW digest too, so the fetch
        # succeeds and the refusal is the digest binding, not a 404.
        jws = hub.receipts[(original_digest, CELL_KEY)]
        hub.receipts[(receipts._blake3_file(artifact), CELL_KEY)] = jws
        _configure(hub)
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False

    def test_key_mismatch_refused(self, tmp_path: Path, hub: HubStub) -> None:
        # Receipt signed for a DIFFERENT key than the artifact claims: the
        # Nix Deriver lesson — key binding must be inside the signature.
        artifact = make_artifact(tmp_path)
        digest = receipts._blake3_file(artifact)
        claims = make_claims(digest, artifact.stat().st_size, cell_key="ck5-" + "e" * 56)
        hub.receipts[(digest, CELL_KEY)] = sign_receipt(hub.key, claims)
        _configure(hub)
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False

    def test_family_mismatch_refused(self, tmp_path: Path, hub: HubStub) -> None:
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(artifact)
        _configure(hub)
        assert receipts.gate_delivered_artifact(artifact, "qwen-image") is False

    def test_revoked_pair_refused(self, tmp_path: Path, hub: HubStub) -> None:
        # R2: an operator recall beats a perfectly valid signature.
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(artifact)
        hub.revoked.append({"cell_key": CELL_KEY, "snapshot_digest": SNAPSHOT, "reason": "bad image"})
        _configure(hub)
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False

    def test_other_revocation_does_not_block(self, tmp_path: Path, hub: HubStub) -> None:
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(artifact)
        hub.revoked.append({"cell_key": "ck5-" + "d" * 56, "snapshot_digest": "other", "reason": "x"})
        _configure(hub)
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is True

    def test_hub_error_fails_closed(self, tmp_path: Path, hub: HubStub) -> None:
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(artifact)
        hub.receipt_status = 503
        _configure(hub)
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False

    def test_verify_names_the_reason(self, tmp_path: Path, hub: HubStub) -> None:
        artifact = make_artifact(tmp_path)
        _configure(hub)
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_delivered_artifact(artifact, FAMILY)
        assert exc.value.reason == "receipt_not_found"


class TestProvisionHook:
    def test_enable_compiled_drops_unreceipted_artifact(
        self, tmp_path: Path, hub: HubStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The provision.enable_compiled hook: a refused delivered artifact
        must be dropped BEFORE compile_cache.enable sees it."""
        from gen_worker.models import provision

        artifact = make_artifact(tmp_path)
        _configure(hub)  # no receipt served -> refusal

        seen: Dict[str, Any] = {}

        def fake_enable(pipe: Any, cfg: Any, cache_dir: Any, art: Any) -> bool:
            seen["artifact"] = art
            return False

        from gen_worker import compile_cache

        monkeypatch.setattr(compile_cache, "enable", fake_enable)

        class Cfg:
            family = FAMILY
            lora_bucket = 0

        armed = provision.enable_compiled(object(), Cfg(), tmp_path, artifact)
        assert armed is False
        assert seen["artifact"] is None, (
            "refused delivered artifact leaked through to compile_cache.enable"
        )

    def test_enable_compiled_passes_verified_artifact(
        self, tmp_path: Path, hub: HubStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gen_worker.models import provision

        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(artifact)
        _configure(hub)

        seen: Dict[str, Any] = {}

        def fake_enable(pipe: Any, cfg: Any, cache_dir: Any, art: Any) -> bool:
            seen["artifact"] = art
            return True

        from gen_worker import compile_cache

        monkeypatch.setattr(compile_cache, "enable", fake_enable)

        class Cfg:
            family = FAMILY
            lora_bucket = 0

        armed = provision.enable_compiled(object(), Cfg(), tmp_path, artifact)
        assert armed is True
        assert seen["artifact"] == artifact
