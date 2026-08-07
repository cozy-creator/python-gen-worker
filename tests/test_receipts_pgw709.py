"""pgw#709: cell-receipt verification — REAL signatures, REAL HTTP, no mocks.

The signer here mirrors the hub's production format byte-for-byte (RS256
PKCS1v15/SHA256 compact JWS, kid header, cell-receipt-v1 claims); the hub
half's Go tests pin the same format from the signing side. The tamper cases
are the red verification: each one asserts the gate actually REFUSES.
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

from gen_worker import receipts

KID = "test-kid-1"
FAMILY = "sdxl"
CELL_KEY = "ck1-0123456789abcdef0123456789abcdef0123456789abcdef01234567"
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
    header = {"alg": alg, "kid": kid, "typ": "cell-receipt-v1+jws"}
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
    gone from this SDK, so a cell it names is re-minted, never armed.
    """
    algo, _, hex_part = artifact_digest.partition(":")
    artifact: Dict[str, Any] = {"path": "cell.tar.gz", "size_bytes": size_bytes}
    if legacy_blake3_only:
        assert algo == "blake3", "legacy receipts only ever carried blake3"
        artifact["blake3"] = hex_part
    else:
        artifact["digest"] = artifact_digest
    claims: Dict[str, Any] = {
        "crv": "cell-receipt-v2",
        "family": FAMILY,
        "cell_key": CELL_KEY,
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
        jws = sign_receipt(rsa_key, make_claims("sha256:" + SHA_HEX, 4096))
        receipt = receipts.verify_receipt_jws(jws, pub_map)
        assert receipt.cell_key == CELL_KEY
        assert receipt.family == FAMILY
        assert receipt.snapshot_digest == SNAPSHOT
        assert receipt.artifact_digest == "sha256:" + SHA_HEX
        assert receipt.artifact_size_bytes == 4096
        assert receipt.axes["sku"] == "rtx-4090"

    def test_tampered_payload_refused(self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]) -> None:
        # RED: re-point the signed payload at a different cell key — the
        # poisoning move receipts exist to prevent.
        jws = sign_receipt(rsa_key, make_claims("sha256:" + SHA_HEX, 4096))
        head, _, sig = jws.split(".")
        forged_claims = make_claims("sha256:" + SHA_HEX, 4096, cell_key="ck1-" + "f" * 56)
        forged = head + "." + _b64url(json.dumps(forged_claims).encode()) + "." + sig
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_receipt_jws(forged, pub_map)
        assert exc.value.reason == "receipt_signature_invalid"

    def test_unknown_kid_refused(self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]) -> None:
        jws = sign_receipt(rsa_key, make_claims("sha256:" + SHA_HEX, 4096), kid="rogue-kid")
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_receipt_jws(jws, pub_map)
        assert exc.value.reason == "receipt_unknown_kid"

    def test_alg_downgrade_refused(self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]) -> None:
        jws = sign_receipt(rsa_key, make_claims("sha256:" + SHA_HEX, 4096), alg="none")
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_receipt_jws(jws, pub_map)
        assert exc.value.reason == "receipt_alg_unsupported"

    def test_wrong_key_refused(self, rsa_key: rsa.RSAPrivateKey) -> None:
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jws = sign_receipt(rsa_key, make_claims("sha256:" + SHA_HEX, 4096))
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_receipt_jws(jws, {KID: other.public_key()})
        assert exc.value.reason == "receipt_signature_invalid"

    def test_wrong_version_refused(self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]) -> None:
        jws = sign_receipt(rsa_key, make_claims("sha256:" + SHA_HEX, 4096, crv="cell-receipt-v1"))
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
                    cell_key = q.get("cell_key", [""])[0]
                    # The real route matches on the SET of tagged digests the
                    # worker offers. pgw#807 deleted the bare-hex `blake3`
                    # param with the protocol that needed it.
                    offered = list(q.get("artifact_digest", []))
                    stub.last_query = (offered, cell_key)
                    jws = None
                    for ref in offered:
                        jws = stub.receipts.get((ref, cell_key))
                        if jws is not None:
                            break
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

    def serve_receipt_for(
        self, artifact: Path, *, algo: str = "sha256", **claim_overrides: Any
    ) -> str:
        """Publish a receipt for ``artifact`` bound with ``algo``; returns the ref."""
        ref = algo + ":" + receipts.artifact_digests(artifact)[algo]
        size = artifact.stat().st_size
        claims = make_claims(ref, size, **claim_overrides)
        self.receipts[(ref, str(claims["cell_key"]))] = sign_receipt(self.key, claims)
        return ref


@pytest.fixture()
def hub(rsa_key: rsa.RSAPrivateKey) -> Iterator[HubStub]:
    stub = HubStub(rsa_key)
    yield stub
    stub.close()
    receipts.reset()


def worker_jwt_for(endpoint_id: str) -> str:
    """A hub-shaped worker credential naming the endpoint this pod serves.

    th#1657: the pod's own identity comes from the `cell_read_endpoint_id` the
    hub stamps on the cell-read grant (th#1335), so the test builds the same
    thing. The signature is never checked — this is our OWN bearer token, not an
    input — so an unsigned third segment is faithful to what the gate reads.
    """
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({
        "sub": "worker-1", "cell_read_endpoint_id": endpoint_id,
    }).encode())
    return header + "." + payload + ".not-checked-here"


def _configure(stub: HubStub, *, endpoint_id: str = SELF_ENDPOINT) -> None:
    receipts.configure(
        base_url=stub.base_url, worker_jwt=lambda: worker_jwt_for(endpoint_id))


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
        original_ref = "sha256:" + receipts.artifact_digests(artifact)["sha256"]
        with artifact.open("ab") as f:
            f.write(b"\x00poison")
        # Serve the original receipt under the NEW digest too, so the fetch
        # succeeds and the refusal is the digest binding, not a 404.
        jws = hub.receipts[(original_ref, CELL_KEY)]
        new_ref = "sha256:" + receipts.artifact_digests(artifact)["sha256"]
        hub.receipts[(new_ref, CELL_KEY)] = jws
        _configure(hub)
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False

    def test_key_mismatch_refused(self, tmp_path: Path, hub: HubStub) -> None:
        # Receipt signed for a DIFFERENT key than the artifact claims: the
        # Nix Deriver lesson — key binding must be inside the signature.
        artifact = make_artifact(tmp_path)
        ref = "sha256:" + receipts.artifact_digests(artifact)["sha256"]
        claims = make_claims(ref, artifact.stat().st_size, cell_key="ck1-" + "e" * 56)
        hub.receipts[(ref, CELL_KEY)] = sign_receipt(hub.key, claims)
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
        hub.revoked.append({"cell_key": "ck1-" + "d" * 56, "snapshot_digest": "other", "reason": "x"})
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

        armed = provision.enable_compiled(object(), Cfg(), tmp_path, artifact).armed
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

        armed = provision.enable_compiled(object(), Cfg(), tmp_path, artifact).armed
        assert armed is True
        assert seen["artifact"] == artifact


# ---------------------------------------------------------------------------
# th#1303 / pgw#807: algorithm-agnostic receipts
# ---------------------------------------------------------------------------


class TestAlgorithmAgnosticReceipts:
    """The guards that let the cell self-mint producer publish over v2.

    A v2 (chunked sha256 CAS) publish has no blake3 anywhere. If arming still
    reads a blake3-named field, every newly minted cell fails to arm and the
    fleet silently re-mints — a fleet-wide re-mint through the receipt door.
    pgw#807 finished the job: blake3 is not a second supported arm, it is a
    refusal, because the protocol that could mint one is gone from this SDK.
    """

    def test_v2_receipt_arms_a_v2_cell(self, tmp_path: Path, hub: HubStub) -> None:
        artifact = make_artifact(tmp_path)
        ref = hub.serve_receipt_for(artifact, algo="sha256")
        assert ref.startswith("sha256:")
        _configure(hub)
        receipt = receipts.verify_delivered_artifact(artifact, FAMILY)
        assert receipt.artifact_digest == ref
        # The worker cannot know the algorithm in advance, so it offers every
        # digest it computed — one request, no per-algorithm 404 retry chain.
        offered, asked_key = hub.last_query
        local = receipts.artifact_digests(artifact)
        assert asked_key == CELL_KEY
        for algo, hex_digest in local.items():
            assert f"{algo}:{hex_digest}" in offered
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is True

    def test_legacy_blake3_receipt_is_refused_not_dual_read(
        self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]
    ) -> None:
        """pgw#807: the pre-v2 receipt shape (bare-hex `blake3`, no `digest`)
        no longer verifies. Its cell cannot be republished over v1 either, so
        the worker refuses it and self-mints a sha256-bound replacement — the
        designed miss policy, not a silent arm on a retired algorithm."""
        claims = make_claims("blake3:" + B3_HEX, 4096, legacy_blake3_only=True)
        jws = sign_receipt(rsa_key, claims)
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_receipt_jws(jws, pub_map)
        assert exc.value.reason == "receipt_no_artifact_digest"

        # And a TAGGED blake3 digest is refused by algorithm, not read.
        tagged = make_claims("blake3:" + B3_HEX, 4096)
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_receipt_jws(sign_receipt(rsa_key, tagged), pub_map)
        assert exc.value.reason == "receipt_digest_algorithm_unsupported"

    def test_wrong_bytes_under_a_correct_tag_refused(
        self, tmp_path: Path, hub: HubStub
    ) -> None:
        # A receipt whose sha256 claim names OTHER bytes: the index row is
        # keyed so the fetch succeeds, and only the digest COMPARE catches it.
        artifact = make_artifact(tmp_path)
        ref = "sha256:" + receipts.artifact_digests(artifact)["sha256"]
        claims = make_claims("sha256:" + SHA_HEX, artifact.stat().st_size)
        hub.receipts[(ref, CELL_KEY)] = sign_receipt(hub.key, claims)
        _configure(hub)
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_delivered_artifact(artifact, FAMILY)
        assert exc.value.reason == "receipt_digest_mismatch"
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False

    def test_digestless_receipt_refused(self, tmp_path: Path, hub: HubStub) -> None:
        # THE trap this migration keeps setting: a receipt binding no digest
        # must REFUSE, not compare an empty string to an empty string.
        artifact = make_artifact(tmp_path)
        ref = "sha256:" + receipts.artifact_digests(artifact)["sha256"]
        claims = make_claims(ref, artifact.stat().st_size)
        claims["artifact"] = {"path": "cell.tar.gz", "size_bytes": artifact.stat().st_size}
        hub.receipts[(ref, CELL_KEY)] = sign_receipt(hub.key, claims)
        _configure(hub)
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_delivered_artifact(artifact, FAMILY)
        assert exc.value.reason == "receipt_no_artifact_digest"
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False

    def test_untagged_and_unsupported_digests_refused(
        self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]
    ) -> None:
        cases = {
            SHA_HEX: "receipt_digest_untagged",           # bare hex assumes nothing
            "md5:" + SHA_HEX: "receipt_digest_algorithm_unsupported",
            "sha256:" + SHA_HEX[:40]: "receipt_digest_malformed",
            "sha256:": "receipt_digest_malformed",        # tag with no value
        }
        for raw, reason in cases.items():
            claims = make_claims("sha256:" + SHA_HEX, 4096)
            claims["artifact"] = {"path": "cell.tar.gz", "digest": raw, "size_bytes": 4096}
            jws = sign_receipt(rsa_key, claims)
            with pytest.raises(receipts.ReceiptError) as exc:
                receipts.verify_receipt_jws(jws, pub_map)
            assert exc.value.reason == reason, f"{raw!r} -> {exc.value.reason}"

    def test_digests_are_computed_in_one_pass(self, tmp_path: Path) -> None:
        artifact = make_artifact(tmp_path)
        got = receipts.artifact_digests(artifact)
        assert set(got) == set(receipts.ARTIFACT_DIGEST_ALGORITHMS) == {"sha256"}
        raw = artifact.read_bytes()
        assert got["sha256"] == hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# th#1657 — the publisher trust boundary at the arm gate (pgw#1008)
# ---------------------------------------------------------------------------


class TestPublisherTrustTh1657:
    """A cell must have come from THIS endpoint, or from a publisher the
    platform vouches for.

    THREAT: cross-tenant native-code execution. The artifact is a `.so` this
    process is about to dlopen. Every other link in the chain proves the bytes
    are the ones the hub signed; none of them asks whether the hub signed them
    FOR THIS POD. `owning_endpoint_id` has ridden the signed receipt since
    pgw#709 and was decoded into `Receipt` and compared against nothing.

    The first test is the RED one: before pgw#1008 it ARMS.
    """

    def test_another_endpoints_org_cell_is_refused(
        self, tmp_path: Path, hub: HubStub
    ) -> None:
        """A genuine, correctly-signed, un-revoked receipt — for someone else.

        Nothing about this artifact is malformed. The signature verifies, the
        digest matches, the size matches, the packed key matches, the family
        matches, the pair is not revoked. It is simply not ours, and before
        pgw#1008 that made no difference at all.
        """
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id=OTHER_ENDPOINT, publisher_tier="org")
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        with pytest.raises(receipts.ReceiptError) as excinfo:
            receipts.verify_delivered_artifact(artifact, FAMILY)
        assert excinfo.value.reason == "publisher_untrusted"
        assert OTHER_ENDPOINT in str(excinfo.value)
        # And the arm hook refuses rather than raising into the boot.
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False

    def test_our_own_org_cell_arms(self, tmp_path: Path, hub: HubStub) -> None:
        """THE CONTROL. Without it, a gate that refused everything would pass
        every other assertion in this class."""
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id=SELF_ENDPOINT, publisher_tier="org")
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        receipt = receipts.verify_delivered_artifact(artifact, FAMILY)
        assert receipt.owning_endpoint_id == SELF_ENDPOINT
        assert receipt.publisher_tier == "org"
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is True

    def test_platform_tier_arms_anywhere(self, tmp_path: Path, hub: HubStub) -> None:
        """Platform-tier is the escape hatch the fleet actually runs on: the
        platform authored that code and already runs it everywhere."""
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id=OTHER_ENDPOINT, publisher_tier="platform")
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        receipt = receipts.verify_delivered_artifact(artifact, FAMILY)
        assert receipt.publisher_tier == "platform"
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is True

    @pytest.mark.parametrize("tier", [None, "", "platform-ish", "PLATFORM", "org"])
    def test_only_exactly_platform_widens(
        self, tmp_path: Path, hub: HubStub, tier: object
    ) -> None:
        """§4.24 point 4: absence must be explicit. An unset, mis-cased or
        invented tier must land on the NARROWER rule, never the wider one — a
        receipt is a permanent statement, so there is no reader-side leniency
        to lean on."""
        artifact = make_artifact(tmp_path)
        overrides: Dict[str, Any] = {"owning_endpoint_id": OTHER_ENDPOINT}
        if tier is not None:
            overrides["publisher_tier"] = tier
        else:
            overrides["publisher_tier"] = None
        hub.serve_receipt_for(artifact, **overrides)
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        with pytest.raises(receipts.ReceiptError) as excinfo:
            receipts.verify_delivered_artifact(artifact, FAMILY)
        assert excinfo.value.reason == "publisher_untrusted"

    def test_pod_that_cannot_name_itself_gets_platform_only(
        self, tmp_path: Path, hub: HubStub
    ) -> None:
        """A worker credential with no `cell_read_endpoint_id` (a hub too old
        for th#1657, or a grant that could not resolve one) narrows this pod to
        platform-tier cells. It does NOT widen it — an identity we cannot
        establish is not an identity that matches everyone."""
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id=SELF_ENDPOINT, publisher_tier="org")
        receipts.configure(base_url=hub.base_url, worker_jwt=lambda: "not-a-jwt")

        with pytest.raises(receipts.ReceiptError) as excinfo:
            receipts.verify_delivered_artifact(artifact, FAMILY)
        assert excinfo.value.reason == "publisher_untrusted"
        assert "cannot name its own endpoint" in str(excinfo.value)

    def test_org_receipt_naming_no_endpoint_is_adoptable_by_nobody(
        self, tmp_path: Path, hub: HubStub
    ) -> None:
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id="", publisher_tier="org")
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        with pytest.raises(receipts.ReceiptError) as excinfo:
            receipts.verify_delivered_artifact(artifact, FAMILY)
        assert excinfo.value.reason == "publisher_untrusted"

    def test_v1_receipt_is_refused_not_defaulted(
        self, tmp_path: Path, hub: HubStub
    ) -> None:
        """The trust fields are load-bearing, so a receipt minted before they
        existed must not be read as a v2 one with them missing. That is the
        `omitempty` collapse §4.24 point 4 names, and here it would silently
        delete the boundary."""
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(artifact, crv="cell-receipt-v1")
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        with pytest.raises(receipts.ReceiptError) as excinfo:
            receipts.verify_delivered_artifact(artifact, FAMILY)
        assert excinfo.value.reason == "receipt_version_unsupported"

    def test_the_refusal_reaches_the_wire(
        self, tmp_path: Path, hub: HubStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pgw#824/pgw#999: a refusal nobody can count is a refusal nobody can
        act on. The class must reach the activity event, not just the log."""
        events: List[Tuple[str, str, str]] = []
        monkeypatch.setattr(
            receipts.activity_mod, "emit_event",
            lambda kind, detail, phase="", **_: events.append((kind, detail, phase)))

        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id=OTHER_ENDPOINT, publisher_tier="org")
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False
        assert events, "the refusal never reached the wire"
        kind, detail, phase = events[-1]
        assert kind == "cell_receipt_refused"
        assert phase == "publisher_untrusted"
        assert FAMILY in detail
