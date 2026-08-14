"""Real compiled-graph receipt signer and JWKS-only HTTP test server.

Resolve embeds the signed receipt, so the hub double serves only the public
JWKS route. Tests receive the JWS directly from :meth:`HubStub.serve_receipt_for`
and verify it before the artifact helper opens local bytes.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from torch_compiled_graphs.identity import (
    from_artifact_metadata,
    toolchain_axis_digest,
)

from gen_worker import artifact_meta, receipts, worker_credential, worker_identity

KID = "test-compiled-graph-receipt-key"
FAMILY = "sdxl"
SNAPSHOT = "sha256:" + "9" * 64
SELF_ENDPOINT = "3e0f8f7a-1111-2222-3333-444455556666"
OTHER_ENDPOINT = "9c1d2e3f-9999-8888-7777-666655554444"
SELF_ORG = "11111111-2222-3333-4444-555555555555"
PUBLISHER = "selfmint:worker=w1:pod=p1:release=r1"
ISSUED_AT = 1_700_000_000


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def sign_receipt(
    key: rsa.RSAPrivateKey,
    claims: dict[str, Any],
    *,
    kid: str = KID,
    alg: str = "RS256",
    typ: str = receipts.RECEIPT_TYPE,
) -> str:
    header = {"alg": alg, "kid": kid, "typ": typ}
    signing_input = ".".join((
        _b64url(json.dumps(header, sort_keys=True).encode()),
        _b64url(json.dumps(claims, sort_keys=True).encode()),
    ))
    signature = key.sign(
        signing_input.encode("ascii"), padding.PKCS1v15(), hashes.SHA256()
    )
    return f"{signing_input}.{_b64url(signature)}"


def _metadata(
    *, graph_hash: str = "receipt-harness-graph", compiled_graph_key: str = "",
) -> dict[str, Any]:
    toolchain = {"torch": "torch-content", "triton": "triton-content"}
    axes = {
        "graph": graph_hash,
        "sm": "sm_86",
        "toolchain": toolchain_axis_digest(toolchain),
    }
    derived = from_artifact_metadata({
        "kind": "aot-inductor",
        "graph_class": {"class_hash": graph_hash},
        "sm": "sm_86",
        "toolchain": toolchain,
    }).value
    return {
        "compiled_graph_format": 1,
        "kind": "aot-inductor",
        "compiled_graph_key": compiled_graph_key or derived,
        "graph_class": {
            "name": "denoiser/b=1",
            "class_hash": axes["graph"],
        },
        "sm": axes["sm"],
        "toolchain": toolchain,
        "host_isa": {"avx2": "true"},
        "package_constants_in_so": False,
        "constant_folding_fenced": True,
    }


def make_artifact(
    tmp_path: Path,
    *,
    compiled_graph_key: str = "",
    graph_hash: str = "receipt-harness-graph",
) -> Path:
    """Pack one TCG-shaped artifact for receipt verification tests."""
    target = tmp_path / "compiled_graph.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        raw = json.dumps(
            _metadata(graph_hash=graph_hash, compiled_graph_key=compiled_graph_key),
            sort_keys=True,
        ).encode()
        member = tarfile.TarInfo(artifact_meta.METADATA_NAME)
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
        payload = b"fake-compiled-graph-shared-object"
        member = tarfile.TarInfo("data/aoti_eager/denoiser.so")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    return target


def artifact_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_identity(path: Path) -> tuple[str, dict[str, str]]:
    metadata = artifact_meta.read_metadata(path)
    identity = from_artifact_metadata(metadata)
    return identity.value, identity.as_dict()


def make_claims(
    artifact: Path,
    *,
    family: str = FAMILY,
    snapshot_digest: str = SNAPSHOT,
    owning_endpoint_id: str = SELF_ENDPOINT,
    publisher: str = PUBLISHER,
    publisher_tier: str = receipts.PUBLISHER_TIER_ORG,
    publisher_org_id: str = SELF_ORG,
    **overrides: Any,
) -> dict[str, Any]:
    key, axes = _artifact_identity(artifact)
    claims: dict[str, Any] = {
        "crv": receipts.RECEIPT_VERSION,
        "family": family,
        "compiled_graph_key": key,
        "identity_axes": axes,
        "owning_endpoint_id": owning_endpoint_id,
        "publisher": publisher,
        "publisher_tier": publisher_tier,
        "publisher_org_id": publisher_org_id,
        "snapshot_digest": snapshot_digest,
        "artifact": {
            "path": artifact.name,
            "digest": artifact_digest(artifact),
            "size_bytes": artifact.stat().st_size,
        },
        "iat": ISSUED_AT,
    }
    claims.update(overrides)
    return claims


@pytest.fixture()
def rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture()
def pub_map(rsa_key: rsa.RSAPrivateKey) -> dict[str, rsa.RSAPublicKey]:
    return {KID: rsa_key.public_key()}


class HubStub:
    """A real HTTP server exposing only the public artifact-signing JWKS."""

    def __init__(self, key: rsa.RSAPrivateKey) -> None:
        self.key = key
        self.requests: list[str] = []
        public = key.public_key().public_numbers()

        def integer(value: int) -> str:
            return _b64url(value.to_bytes((value.bit_length() + 7) // 8, "big"))

        jwks = {"keys": [{
            "kty": "RSA",
            "kid": KID,
            "use": "sig",
            "alg": "RS256",
            "n": integer(public.n),
            "e": integer(public.e),
        }]}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                stub.requests.append(self.path)
                if self.path != receipts.JWKS_PATH:
                    self.send_error(404)
                    return
                body = json.dumps(jwks).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def serve_receipt_for(self, artifact: Path, **claim_overrides: Any) -> str:
        """Return the embedded JWS; no receipt lookup route exists."""
        return sign_receipt(self.key, make_claims(artifact, **claim_overrides))


def verify_artifact(
    artifact: Path, family: str, receipt_jws: str,
) -> receipts.Receipt:
    """Drive the production pre-transport then pre-import receipt gates."""
    key, _axes = _artifact_identity(artifact)
    verified = receipts.verify_receipt(
        receipt_jws,
        family=family,
        compiled_graph_key=key,
        snapshot_digest=SNAPSHOT,
        artifact_path=artifact.name,
        artifact_digest=artifact_digest(artifact),
        artifact_size_bytes=artifact.stat().st_size,
    )
    return receipts.verify_delivered_artifact(artifact, family, verified)


@pytest.fixture()
def hub(rsa_key: rsa.RSAPrivateKey) -> Iterator[HubStub]:
    stub = HubStub(rsa_key)
    receipts.configure(stub.base_url)
    yield stub
    stub.close()
    receipts._reset_for_tests()
    worker_identity.reset()
    worker_credential.reset()


def worker_jwt_for(endpoint_id: str, org_id: str = "") -> str:
    """Build the viewer claims a real worker credential carries."""
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    claims = {"sub": "worker-1", "cell_read_endpoint_id": endpoint_id}
    if org_id:
        claims["cell_read_org_id"] = org_id
    return f"{header}.{_b64url(json.dumps(claims).encode())}.not-checked-here"


def identify(token: str) -> None:
    """Install this process's own worker identity exactly once."""
    worker_identity.reset()
    worker_credential.reset()
    if token:
        worker_credential.install(token)
