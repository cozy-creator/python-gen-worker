"""pgw#709: cell-receipt verification — REAL signatures, REAL HTTP, no mocks.

The signer here mirrors the hub's production format byte-for-byte (RS256
PKCS1v15/SHA256 compact JWS, kid header, cell-receipt-v1 claims); the hub
half's Go tests pin the same format from the signing side. The tamper cases
are the red verification: each one asserts the gate actually REFUSES.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from gen_worker import receipts, worker_credential, worker_identity

# pgw#1152: the signer + the live hub moved to `tests/harness/receipt_hub.py`.
# pgw#1122's identity seam already imported them from here, and the adopt-path
# rig needs the same real receipt gate. Same objects, one home.
from harness.receipt_hub import (  # noqa: F401 — fixtures come with it
    B3_HEX, CELL_KEY, FAMILY, KID, OTHER_ENDPOINT, SELF_ENDPOINT, SELF_ORG,
    SHA_HEX, SNAPSHOT,
    HubStub, _b64url, _configure, _identify, hub, make_artifact, make_claims,
    pub_map, rsa_key, sign_receipt, worker_jwt_for,
)


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
        # pgw#1034: claims nothing checks are not decoded. `make_claims` still
        # signs `axes`/`publisher`/`manifest_digest`/`fingerprint_digest`/`iat`
        # — the hub emits them — and an undecoded claim must be inert, never a
        # parse failure.
        for dropped in ("axes", "publisher", "artifact_path",
                        "manifest_digest", "fingerprint_digest",
                        "issued_at_unix"):
            assert not hasattr(receipt, dropped), dropped

    def test_tampered_payload_refused(self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]) -> None:
        # RED: re-point the signed payload at a different cell key — the
        # poisoning move receipts exist to prevent.
        jws = sign_receipt(rsa_key, make_claims("sha256:" + SHA_HEX, 4096))
        head, _, sig = jws.split(".")
        forged_claims = make_claims("sha256:" + SHA_HEX, 4096, cell_key="ek1-" + "f" * 56)
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
        original_ref = receipts.artifact_digest(artifact)
        with artifact.open("ab") as f:
            f.write(b"\x00poison")
        # Serve the original receipt under the NEW digest too, so the fetch
        # succeeds and the refusal is the digest binding, not a 404.
        jws = hub.receipts[(original_ref, CELL_KEY)]
        new_ref = receipts.artifact_digest(artifact)
        hub.receipts[(new_ref, CELL_KEY)] = jws
        _configure(hub)
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False

    def test_key_mismatch_refused(self, tmp_path: Path, hub: HubStub) -> None:
        # Receipt signed for a DIFFERENT key than the artifact claims: the
        # Nix Deriver lesson — key binding must be inside the signature.
        artifact = make_artifact(tmp_path)
        ref = receipts.artifact_digest(artifact)
        claims = make_claims(ref, artifact.stat().st_size, cell_key="ek1-" + "e" * 56)
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
        hub.revoked.append({"cell_key": "ek1-" + "d" * 56, "snapshot_digest": "other", "reason": "x"})
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
        # One request carrying the ALGORITHM-TAGGED digest — no per-algorithm
        # 404 retry chain, and never bare hex (pgw#1034).
        offered, asked_key = hub.last_query
        assert asked_key == CELL_KEY
        assert receipts.artifact_digest(artifact) in offered
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
        ref = receipts.artifact_digest(artifact)
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
        ref = receipts.artifact_digest(artifact)
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

    def test_the_local_digest_is_tagged_and_single_algorithm(
        self, tmp_path: Path
    ) -> None:
        artifact = make_artifact(tmp_path)
        got = receipts.artifact_digest(artifact)
        raw = artifact.read_bytes()
        assert got == "sha256:" + hashlib.sha256(raw).hexdigest()
        assert receipts.ARTIFACT_DIGEST_ALGORITHM == "sha256"


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
        _identify("not-a-jwt")
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


# ---------------------------------------------------------------------------
# th#1680 / pgw#1021 — the two layers apply ONE rule
# ---------------------------------------------------------------------------

# THE SHARED ADOPTION TABLE. Twin of tensorhub's
# `internal/authz/cell_adoption_table_th1680_test.go` (`TH1680AdoptionTable`).
# Same rows, same order, same verdicts — a change made to one layer and not the
# other fails a test here instead of producing a quiet disagreement that costs a
# cold mint per boot. Tuple order matches the Go struct's field order so the two
# can be diffed by eye.
#
#   (name, tier, publisher_org, viewer_org, want_adoptable, why)
TH1680_ADOPTION_TABLE = [
    ("platform tier, foreign org", "platform", "org-a", "org-b", True,
     "the platform authored that code and already runs it everywhere"),
    ("platform tier, no publisher org", "platform", "", "org-b", True,
     "platform tier needs no publisher identity — the tier IS the identity"),
    ("platform tier, no viewer", "platform", "org-a", "", True,
     "a caller we cannot name still may adopt platform cells"),
    ("org tier, same org", "org", "org-a", "org-a", True,
     "th#1680: same org adopts across its OWN endpoints — the case th#1657 refused"),
    ("org tier, foreign org", "org", "org-a", "org-b", False,
     "THE THREAT: cross-tenant native-code execution"),
    ("org tier, no publisher org", "org", "", "org-a", False,
     "a cell whose publisher is unresolvable is adoptable by nobody"),
    ("org tier, no viewer org", "org", "org-a", "", False,
     "an identity we cannot establish is not an identity that matches everyone"),
    ("org tier, both unresolvable", "org", "", "", False,
     "empty-equals-empty must NOT match — the vacuous-guard shape"),
    ("empty tier is org, foreign", "", "org-a", "org-b", False,
     "§4.24 point 4: an unset tier lands on the NARROWER rule"),
    ("empty tier is org, same org", "", "org-a", "org-a", True,
     "…and still adopts within its own org"),
    ("invented tier is org", "platform-ish", "org-a", "org-b", False,
     "only exactly `platform` widens"),
    ("mis-cased tier is org", "PLATFORM", "org-a", "org-b", False,
     "no case folding — the receipt is a permanent statement, normalized before signing"),
]


def _receipt_for(tier: str, publisher_org: str, owning_endpoint: str = "") -> receipts.Receipt:
    """A Receipt carrying only the fields the publisher gate reads."""
    return receipts.Receipt(
        version=receipts.RECEIPT_VERSION, family=FAMILY, cell_key=CELL_KEY,
        owning_endpoint_id=owning_endpoint,
        publisher_tier=receipts._normalize_publisher_tier(tier),
        publisher_org_id=publisher_org,
        snapshot_digest=SNAPSHOT,
        artifact_digest="sha256:" + SHA_HEX, artifact_size_bytes=1)


class TestSharedAdoptionTableTh1680:
    @pytest.mark.parametrize(
        "name,tier,publisher_org,viewer_org,want,why", TH1680_ADOPTION_TABLE)
    def test_row(self, name: str, tier: str, publisher_org: str,
                 viewer_org: str, want: bool, why: str) -> None:
        # Endpoints are deliberately DIFFERENT on every row, so each row tests
        # the ORG axis alone — the endpoint rule must not mask an org verdict.
        receipt = _receipt_for(tier, publisher_org, owning_endpoint="ep-publisher")
        try:
            receipts.refuse_untrusted_publisher(receipt, "ep-viewer", viewer_org)
            got = True
        except receipts.ReceiptError as exc:
            assert exc.reason == "publisher_untrusted"
            got = False
        assert got is want, (
            f"{name}: adoptable={got}, want {want} — {why}. "
            "If you changed this rule, change tensorhub's "
            "internal/authz/cell_adoption_table_th1680_test.go too.")

    def test_table_covers_both_verdicts(self) -> None:
        """A guard on the TABLE, not the code: an all-true or all-false table
        would pass every row above while proving nothing."""
        adoptable = sum(1 for row in TH1680_ADOPTION_TABLE if row[4])
        refused = len(TH1680_ADOPTION_TABLE) - adoptable
        assert adoptable and refused, (
            f"table must exercise both verdicts; got {adoptable}/{refused}")
        assert len(TH1680_ADOPTION_TABLE) == 12, (
            "tensorhub's TH1680AdoptionTable has 12 rows — add the row THERE "
            "too, then update both counts")


class TestOrgWideningTh1680:
    """RED both directions, through the real receipt path."""

    def test_same_org_different_endpoint_adopts(
        self, tmp_path: Path, hub: HubStub
    ) -> None:
        """THE RED CASE. The hub's listing shows this cell (it is the same org);
        before pgw#1021 the arm gate refused it, costing a wasted download and a
        full cold mint."""
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id=OTHER_ENDPOINT,
            publisher_tier="org", publisher_org_id=SELF_ORG)
        _identify(worker_jwt_for(SELF_ENDPOINT, org_id=SELF_ORG))
        receipts.configure(
            base_url=hub.base_url,
            worker_jwt=lambda: worker_jwt_for(SELF_ENDPOINT, org_id=SELF_ORG))

        receipt = receipts.verify_delivered_artifact(artifact, FAMILY)
        assert receipt.publisher_org_id == SELF_ORG
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is True

    def test_different_org_still_refused(self, tmp_path: Path, hub: HubStub) -> None:
        """The threat is unchanged: another ORG's native code never arms."""
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id=OTHER_ENDPOINT,
            publisher_tier="org", publisher_org_id="99999999-0000-0000-0000-000000000000")
        _identify(worker_jwt_for(SELF_ENDPOINT, org_id=SELF_ORG))
        receipts.configure(
            base_url=hub.base_url,
            worker_jwt=lambda: worker_jwt_for(SELF_ENDPOINT, org_id=SELF_ORG))

        with pytest.raises(receipts.ReceiptError) as excinfo:
            receipts.verify_delivered_artifact(artifact, FAMILY)
        assert excinfo.value.reason == "publisher_untrusted"

    def test_hub_without_th1680_degrades_to_endpoint_only(
        self, tmp_path: Path, hub: HubStub
    ) -> None:
        """THE SAFE DEGRADATION, and the reason this needs no coupled deploy:
        a grant with no `cell_read_org_id` (a hub older than th#1680) leaves
        this pod on pgw#1008's endpoint-only rule. Narrower, never wider."""
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id=OTHER_ENDPOINT,
            publisher_tier="org", publisher_org_id=SELF_ORG)
        _configure(hub, endpoint_id=SELF_ENDPOINT)  # no org stamped

        with pytest.raises(receipts.ReceiptError) as excinfo:
            receipts.verify_delivered_artifact(artifact, FAMILY)
        assert excinfo.value.reason == "publisher_untrusted"

    def test_same_endpoint_still_adopts_without_org(
        self, tmp_path: Path, hub: HubStub
    ) -> None:
        """The endpoint rule survives untouched — an old hub's pods keep
        adopting their own cells."""
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id=SELF_ENDPOINT,
            publisher_tier="org", publisher_org_id="")
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        assert receipts.gate_delivered_artifact(artifact, FAMILY) is True

    def test_receipt_version_did_not_move(self) -> None:
        """th#1678's lesson: `publisher_org_id` already shipped in v2, so this
        change is additive on the GRANT and touches no wire constant. If this
        assertion ever needs updating, the change is a COUPLED hub+fleet cut and
        must say so in its own body."""
        assert receipts.RECEIPT_VERSION == "cell-receipt-v2"
