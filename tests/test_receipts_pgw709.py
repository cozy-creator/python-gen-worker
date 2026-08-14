"""Compiled-graph receipt v1: real RS256, real JWKS HTTP, strict bindings."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import tarfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from torch_compiled_graphs.identity import from_axes, toolchain_axis_digest

from gen_worker import artifact_meta, receipts, worker_identity

KID = "compiled-graph-receipts-test"
FAMILY = "sdxl"
SNAPSHOT = "sha256:" + "9" * 64
ENDPOINT = "endpoint-a"
ORG = "org-a"
PUBLISHER = "selfmint:worker=w1:pod=p1:release=r1"
ISSUED_AT = 1_700_000_000


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _sign(
    key: rsa.RSAPrivateKey,
    claims: dict[str, Any],
    *,
    kid: str = KID,
    alg: str = "RS256",
    typ: str = receipts.RECEIPT_TYPE,
    extra_header: bool = False,
) -> str:
    header: dict[str, Any] = {"alg": alg, "kid": kid, "typ": typ}
    if extra_header:
        header["legacy"] = True
    signing_input = ".".join((
        _b64url(json.dumps(header, sort_keys=True).encode()),
        _b64url(json.dumps(claims, sort_keys=True).encode()),
    ))
    signature = key.sign(
        signing_input.encode("ascii"), padding.PKCS1v15(), hashes.SHA256()
    )
    return f"{signing_input}.{_b64url(signature)}"


@dataclass(frozen=True)
class ArtifactCase:
    path: Path
    key: str
    axes: dict[str, str]
    digest: str
    size: int


def _artifact(
    directory: Path, *, graph_hash: str = "graph-class-a", name: str = "compiled_graph.tar.gz"
) -> ArtifactCase:
    toolchain = {"torch": "torch-content", "triton": "triton-content"}
    axes = {
        "graph": graph_hash,
        "sm": "sm_86",
        "toolchain": toolchain_axis_digest(toolchain),
    }
    key = from_axes(axes).value
    metadata = {
        "compiled_graph_format": 1,
        "kind": "aot-inductor",
        "compiled_graph_key": key,
        "graph_class": {"name": "denoiser/b=1", "class_hash": graph_hash},
        "sm": "sm_86",
        "toolchain": toolchain,
        "host_isa": {"avx2": "true"},
        "package_constants_in_so": False,
        "constant_folding_fenced": True,
    }
    path = directory / name
    with tarfile.open(path, "w:gz") as archive:
        raw = json.dumps(metadata, sort_keys=True).encode()
        member = tarfile.TarInfo(artifact_meta.METADATA_NAME)
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
        payload = b"compiled-shared-object"
        member = tarfile.TarInfo("data/aoti_eager/denoiser.so")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    body = path.read_bytes()
    return ArtifactCase(
        path=path,
        key=key,
        axes=axes,
        digest="sha256:" + hashlib.sha256(body).hexdigest(),
        size=len(body),
    )


def _claims(case: ArtifactCase) -> dict[str, Any]:
    return {
        "crv": receipts.RECEIPT_VERSION,
        "family": FAMILY,
        "compiled_graph_key": case.key,
        "identity_axes": dict(case.axes),
        "owning_endpoint_id": ENDPOINT,
        "publisher": PUBLISHER,
        "publisher_tier": receipts.PUBLISHER_TIER_PLATFORM,
        "publisher_org_id": ORG,
        "snapshot_digest": SNAPSHOT,
        "artifact": {
            "path": "compiled_graph.tar.gz",
            "digest": case.digest,
            "size_bytes": case.size,
        },
        "iat": ISSUED_AT,
    }


class JWKSServer:
    def __init__(self, key: rsa.RSAPrivateKey) -> None:
        public = key.public_key().public_numbers()

        def integer(value: int) -> str:
            return _b64url(value.to_bytes((value.bit_length() + 7) // 8, "big"))

        document = {
            "keys": [{
                "kty": "RSA",
                "kid": KID,
                "alg": "RS256",
                "use": "sig",
                "n": integer(public.n),
                "e": integer(public.e),
            }]
        }

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                if self.path != receipts.JWKS_PATH:
                    self.send_error(404)
                    return
                body = json.dumps(document).encode()
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


@pytest.fixture()
def signing_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture()
def jwks(signing_key: rsa.RSAPrivateKey) -> Iterator[JWKSServer]:
    server = JWKSServer(signing_key)
    receipts.configure(server.base_url)
    yield server
    server.close()


@pytest.fixture(autouse=True)
def reset_receipts() -> Iterator[None]:
    receipts.reset()
    worker_identity.reset()
    yield
    receipts.reset()
    worker_identity.reset()


def _verify(jws: str, case: ArtifactCase) -> receipts.Receipt:
    return receipts.verify_receipt(
        jws,
        family=FAMILY,
        compiled_graph_key=case.key,
        snapshot_digest=SNAPSHOT,
        artifact_path="compiled_graph.tar.gz",
        artifact_digest=case.digest,
        artifact_size_bytes=case.size,
    )


def test_exact_v1_round_trip_exposes_every_verified_claim(
    tmp_path: Path, signing_key: rsa.RSAPrivateKey, jwks: JWKSServer,
) -> None:
    del jwks
    case = _artifact(tmp_path)
    receipt = _verify(_sign(signing_key, _claims(case)), case)

    assert receipt == receipts.Receipt(
        version="compiled-graph-receipt-v1",
        family=FAMILY,
        compiled_graph_key=case.key,
        identity_axes=tuple(sorted(case.axes.items())),
        owning_endpoint_id=ENDPOINT,
        publisher=PUBLISHER,
        publisher_tier="platform",
        publisher_org_id=ORG,
        snapshot_digest=SNAPSHOT,
        artifact_path="compiled_graph.tar.gz",
        artifact_digest=case.digest,
        artifact_size_bytes=case.size,
        issued_at_unix=ISSUED_AT,
    )


def test_materialized_bytes_and_tcg_identity_match_the_preverified_receipt(
    tmp_path: Path, signing_key: rsa.RSAPrivateKey, jwks: JWKSServer,
) -> None:
    del jwks
    case = _artifact(tmp_path)
    receipt = _verify(_sign(signing_key, _claims(case)), case)

    assert receipts.verify_delivered_artifact(case.path, FAMILY, receipt) is receipt

    body = bytearray(case.path.read_bytes())
    body[-1] ^= 1
    case.path.write_bytes(body)
    with pytest.raises(receipts.ReceiptError) as exc:
        receipts.verify_delivered_artifact(case.path, FAMILY, receipt)
    assert exc.value.reason == "receipt_digest_mismatch"


def test_artifact_identity_axes_must_match_the_signed_axes(
    tmp_path: Path, signing_key: rsa.RSAPrivateKey, jwks: JWKSServer,
) -> None:
    del jwks
    signed_identity = _artifact(tmp_path, name="signed.tar.gz")
    delivered = _artifact(tmp_path, graph_hash="graph-class-b")
    claims = _claims(signed_identity)
    artifact_claim = claims["artifact"]
    assert isinstance(artifact_claim, dict)
    artifact_claim.update({
        "digest": delivered.digest,
        "size_bytes": delivered.size,
    })
    receipt = receipts.verify_receipt(
        _sign(signing_key, claims),
        family=FAMILY,
        compiled_graph_key=signed_identity.key,
        snapshot_digest=SNAPSHOT,
        artifact_path="compiled_graph.tar.gz",
        artifact_digest=delivered.digest,
        artifact_size_bytes=delivered.size,
    )

    with pytest.raises(receipts.ReceiptError) as exc:
        receipts.verify_delivered_artifact(delivered.path, FAMILY, receipt)
    assert exc.value.reason == "receipt_identity_mismatch"


def test_only_a_typed_preverified_receipt_reaches_local_bytes(tmp_path: Path) -> None:
    case = _artifact(tmp_path)
    with pytest.raises(receipts.ReceiptError) as exc:
        receipts.verify_delivered_artifact(
            case.path, FAMILY, "header.payload.signature"  # type: ignore[arg-type]
        )
    assert exc.value.reason == "receipt_unverified"


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("family", "receipt_family_mismatch"),
        ("compiled_graph_key", "receipt_identity_mismatch"),
        ("identity_axes", "receipt_identity_mismatch"),
        ("owning_endpoint_id", "receipt_claim_missing"),
        ("publisher", "receipt_claim_missing"),
        ("publisher_tier", "receipt_publisher_tier_invalid"),
        ("publisher_org_id", "receipt_claim_missing"),
        ("snapshot_digest", "receipt_snapshot_digest_mismatch"),
        ("artifact_path", "receipt_artifact_path_mismatch"),
        ("artifact_digest", "receipt_artifact_digest_mismatch"),
        ("artifact_size_bytes", "receipt_artifact_size_bytes_mismatch"),
        ("iat", "receipt_issuance_invalid"),
    ],
)
def test_every_signed_binding_tamper_refuses_before_download(
    field: str,
    reason: str,
    tmp_path: Path,
    signing_key: rsa.RSAPrivateKey,
    jwks: JWKSServer,
) -> None:
    del jwks
    case = _artifact(tmp_path)
    claims = _claims(case)
    if field == "family":
        claims["family"] = "another-family"
    elif field == "compiled_graph_key":
        claims[field] = "cg-key-v1-" + "f" * 56
    elif field == "identity_axes":
        axes = dict(case.axes)
        axes["graph"] = "another-graph"
        claims[field] = axes
    elif field in {"owning_endpoint_id", "publisher", "publisher_org_id"}:
        claims[field] = ""
    elif field == "publisher_tier":
        claims[field] = "platform-ish"
    elif field == "snapshot_digest":
        claims[field] = "sha256:" + "8" * 64
    elif field.startswith("artifact_"):
        artifact = claims["artifact"]
        assert isinstance(artifact, dict)
        if field == "artifact_path":
            artifact["path"] = "other.tar.gz"
        elif field == "artifact_digest":
            artifact["digest"] = "sha256:" + "7" * 64
        else:
            artifact["size_bytes"] = case.size + 1
    elif field == "iat":
        claims[field] = 0
    downloads: list[str] = []

    def verify_then_download() -> None:
        _verify(_sign(signing_key, claims), case)
        downloads.append("download")

    with pytest.raises(receipts.ReceiptError) as exc:
        verify_then_download()
    assert exc.value.reason == reason
    assert downloads == [], f"{field} reached transport"


def test_signature_tamper_and_non_rs256_headers_refuse(
    tmp_path: Path, signing_key: rsa.RSAPrivateKey, jwks: JWKSServer,
) -> None:
    del jwks
    case = _artifact(tmp_path)
    original = _sign(signing_key, _claims(case))
    header, _payload, signature = original.split(".")
    forged_claims = _claims(case)
    forged_claims["family"] = "forged-family"
    forged = f"{header}.{_b64url(json.dumps(forged_claims).encode())}.{signature}"
    with pytest.raises(receipts.ReceiptError) as exc:
        _verify(forged, case)
    assert exc.value.reason == "receipt_signature_invalid"

    for jws, reason in (
        (_sign(signing_key, _claims(case), alg="none"), "receipt_alg_unsupported"),
        (_sign(signing_key, _claims(case), kid="unknown"), "receipt_unknown_kid"),
        (
            _sign(signing_key, _claims(case), typ="cell-receipt-v1+jws"),
            "receipt_type_unsupported",
        ),
        (_sign(signing_key, _claims(case), extra_header=True), "receipt_header_shape"),
    ):
        with pytest.raises(receipts.ReceiptError) as exc:
            _verify(jws, case)
        assert exc.value.reason == reason


def test_old_versions_fields_routes_and_redundant_claims_are_absent(
    tmp_path: Path, signing_key: rsa.RSAPrivateKey, jwks: JWKSServer,
) -> None:
    del jwks
    case = _artifact(tmp_path)
    assert receipts.RECEIPT_VERSION == "compiled-graph-receipt-v1"
    assert receipts.RECEIPT_TYPE == "compiled-graph-receipt-v1+jws"
    for name in (
        "RECEIPT_PATH",
        "REVOCATIONS_PATH",
        "gate_delivered_artifact",
        "_fetch_receipt_jws",
        "_fetch_revocations",
    ):
        assert not hasattr(receipts, name)
    assert not hasattr(receipts.Receipt, "cell_key")

    old_version = _claims(case)
    old_version["crv"] = "cell-receipt-v2"
    with pytest.raises(receipts.ReceiptError) as exc:
        _verify(_sign(signing_key, old_version), case)
    assert exc.value.reason == "receipt_version_unsupported"

    for legacy in ("cell_key", "axes", "manifest_digest", "fingerprint_digest"):
        claims = copy.deepcopy(_claims(case))
        claims[legacy] = claims["compiled_graph_key"]
        with pytest.raises(receipts.ReceiptError) as exc:
            _verify(_sign(signing_key, claims), case)
        assert exc.value.reason == "receipt_claim_shape", legacy


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("digest", "SHA256:" + "A" * 64, "receipt_digest_malformed"),
        ("path", "./compiled_graph.tar.gz", "receipt_artifact_path_invalid"),
        ("size_bytes", "123", "receipt_artifact_size_invalid"),
    ],
)
def test_artifact_claims_have_no_coercion_or_canonicalization_alias(
    field: str,
    value: object,
    reason: str,
    tmp_path: Path,
    signing_key: rsa.RSAPrivateKey,
    jwks: JWKSServer,
) -> None:
    del jwks
    case = _artifact(tmp_path)
    claims = _claims(case)
    artifact = claims["artifact"]
    assert isinstance(artifact, dict)
    artifact[field] = value
    with pytest.raises(receipts.ReceiptError) as exc:
        _verify(_sign(signing_key, claims), case)
    assert exc.value.reason == reason


def test_org_receipts_require_this_endpoint_or_org_before_download(
    tmp_path: Path,
    signing_key: rsa.RSAPrivateKey,
    jwks: JWKSServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del jwks
    case = _artifact(tmp_path)
    claims = _claims(case)
    claims["publisher_tier"] = receipts.PUBLISHER_TIER_ORG
    monkeypatch.setattr(
        receipts.worker_identity,
        "viewer",
        lambda: SimpleNamespace(endpoint_id="endpoint-b", org_id=ORG),
    )
    assert _verify(_sign(signing_key, claims), case).publisher_org_id == ORG

    monkeypatch.setattr(
        receipts.worker_identity,
        "viewer",
        lambda: SimpleNamespace(endpoint_id="endpoint-b", org_id="org-b"),
    )
    with pytest.raises(receipts.ReceiptError) as exc:
        _verify(_sign(signing_key, claims), case)
    assert exc.value.reason == "publisher_untrusted"
