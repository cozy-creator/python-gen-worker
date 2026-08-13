"""A REAL compiled graph-receipt hub: real RSA keys, real JWS, real HTTP on localhost.

Promoted out of ``test_receipts_pgw709.py`` by pgw#1152, unchanged. The signer
mirrors the hub's production format byte-for-byte (RS256 PKCS1v15/SHA256
compact JWS, kid header, ``compiled graph-receipt-v1+jws`` typ, ``compiled graph-receipt-v2``
claims); the hub half's Go tests pin the same format from the signing side.
:class:`HubStub` serves the three real routes the worker calls — JWKS, receipt
lookup, revocations — on a real socket.

It is here rather than in a test module because the receipt gate is on the ARM
path: :mod:`harness.adopt_rig` drives a boot-adopt through it, and pgw#1122's
identity seam drives it from the other side. A shared vehicle belongs where
shared vehicles live.
"""

from __future__ import annotations

import base64
import hashlib
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

from gen_worker import receipts, worker_credential, worker_identity

KID = "test-kid-1"
FAMILY = "sdxl"
COMPILED_GRAPH_KEY = "ck1-0123456789abcdef0123456789abcdef0123456789abcdef01234567"
SNAPSHOT = "snapdigest-abc123"
# th#1657: the endpoint the test pod serves, and the one every fixture receipt
# is minted FOR unless a test deliberately mints it for someone else. Keeping
# the default a matching ORG-tier pair means the publisher gate is live in every
# case below, not just the ones that name it.
SELF_ENDPOINT = "3e0f8f7a-1111-2222-3333-444455556666"
OTHER_ENDPOINT = "9c1d2e3f-9999-8888-7777-666655554444"
SELF_ORG = "11111111-2222-3333-4444-555555555555"
B3_HEX = "ab12cd34" * 8
SHA_HEX = "12ab34cd" * 8


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def sign_receipt(
    key: rsa.RSAPrivateKey,
    claims: Dict[str, Any],
    *,
    kid: str = KID,
    alg: str = "RS256",
) -> str:
    header = {"alg": alg, "kid": kid, "typ": "compiled_graph-receipt-v1+jws"}
    signing_input = (
        _b64url(json.dumps(header).encode()) + "." + _b64url(json.dumps(claims).encode())
    )
    sig = key.sign(signing_input.encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    return signing_input + "." + _b64url(sig)


def make_claims(
    artifact_digest: str,
    size_bytes: int,
    *,
    legacy_blake3_only: bool = False,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build receipt claims binding an ALGORITHM-TAGGED artifact digest.

    ``legacy_blake3_only`` reproduces a receipt minted before pgw#807: the
    bare-hex ``blake3`` claim and no ``digest`` at all. It exists so the tests
    can prove that shape is now REFUSED — the v1 protocol that minted it is
    gone from this SDK, so a compiled graph it names is re-minted, never armed.
    """
    algo, _, hex_part = artifact_digest.partition(":")
    artifact: Dict[str, Any] = {"path": "compiled_graph.tar.gz", "size_bytes": size_bytes}
    if legacy_blake3_only:
        assert algo == "blake3", "legacy receipts only ever carried blake3"
        artifact["blake3"] = hex_part
    else:
        artifact["digest"] = artifact_digest
    claims: Dict[str, Any] = {
        "crv": "compiled_graph-receipt-v2",
        "family": FAMILY,
        "compiled_graph_key": COMPILED_GRAPH_KEY,
        "axes": {"sku": "rtx-4090", "image_digest": "sha256:feed", "gen_worker": "0.75.1"},
        "owning_endpoint_id": SELF_ENDPOINT,
        "publisher": "selfmint:worker=w1:pod=p1:release=r1",
        # th#1657 publisher trust, inside the signature.
        "publisher_tier": "org",
        "publisher_org_id": SELF_ORG,
        "snapshot_digest": SNAPSHOT,
        "artifact": artifact,
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


def make_artifact(tmp_path: Path, *, compiled_graph_key: str = COMPILED_GRAPH_KEY, family: str = FAMILY) -> Path:
    meta = {"compiled_graph_key": compiled_graph_key, "family": family, "format": "cozy-compile-cache/v1"}
    target = tmp_path / "compiled_graph.tar.gz"
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


class HubStub:
    """A real HTTP server speaking the hub's three receipt surfaces."""

    def __init__(self, key: rsa.RSAPrivateKey) -> None:
        self.key = key
        self.receipts: Dict[Tuple[str, str], str] = {}
        self.last_query: Tuple[List[str], str] = ([], "")
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
                    compiled_graph_key = q.get("compiled_graph_key", [""])[0]
                    # The real route matches on the SET of tagged digests the
                    # worker offers. pgw#807 deleted the bare-hex `blake3`
                    # param with the protocol that needed it.
                    offered = list(q.get("artifact_digest", []))
                    stub.last_query = (offered, compiled_graph_key)
                    jws = None
                    for ref in offered:
                        jws = stub.receipts.get((ref, compiled_graph_key))
                        if jws is not None:
                            break
                    if jws is None:
                        self._json(404, {"error": "compiled_graph_receipt_not_found"})
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

    def serve_receipt_for(
        self, artifact: Path, *, algo: str = "sha256", **claim_overrides: Any
    ) -> str:
        """Publish a receipt for ``artifact`` bound with ``algo``; returns the ref."""
        assert algo == "sha256"
        ref = receipts.artifact_digest(artifact)
        size = artifact.stat().st_size
        claims = make_claims(ref, size, **claim_overrides)
        self.receipts[(ref, str(claims["compiled_graph_key"]))] = sign_receipt(self.key, claims)
        return ref


@pytest.fixture()
def hub(rsa_key: rsa.RSAPrivateKey) -> Iterator[HubStub]:
    stub = HubStub(rsa_key)
    yield stub
    stub.close()
    receipts.reset()
    # pgw#1122: identity is a PROCESS fact now, so the fixture must unwind it
    # too or the next test inherits this one's pod.
    worker_identity.reset()
    worker_credential.reset()


def worker_jwt_for(endpoint_id: str, org_id: str = "") -> str:
    """A hub-shaped worker credential naming the endpoint this pod serves.

    th#1657: the pod's own identity comes from the `compiled_graph_read_endpoint_id` the
    hub stamps on the compiled graph-read grant (th#1335), so the test builds the same
    thing. The signature is never checked — this is our OWN bearer token, not an
    input — so an unsigned third segment is faithful to what the gate reads.
    """
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    claims = {"sub": "worker-1", "compiled_graph_read_endpoint_id": endpoint_id}
    # th#1680: the org rides the same grant. Omitted when empty, exactly as a
    # pre-th#1680 hub (or one that could not resolve the org) leaves it out.
    if org_id:
        claims["compiled_graph_read_org_id"] = org_id
    payload = _b64url(json.dumps(claims).encode())
    return header + "." + payload + ".not-checked-here"


def _identify(token: str) -> None:
    """Give this process the credential a single-process worker holds.

    pgw#1122: the gate no longer decodes its OWN bearer for identity — the
    compute child holds none by construction, so the process-wide credential
    (``worker_credential``, pgw#848's single source) is what
    ``worker_identity.viewer`` reads, exactly as production does after the
    transport installs a rotation. Installing it here is what production
    writes; passing a token to ``receipts.configure`` was only ever a bearer.
    """
    worker_identity.reset()
    worker_credential.reset()
    if token:
        worker_credential.install(token)


def _configure(stub: HubStub, *, endpoint_id: str = SELF_ENDPOINT) -> None:
    token = worker_jwt_for(endpoint_id)
    _identify(token)
    receipts.configure(base_url=stub.base_url, worker_jwt=lambda: token)

