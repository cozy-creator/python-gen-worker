"""Cell receipts and the trust gate: who signed it, who may arm it.

Sections keep their incident id; the full narratives live in the tracker.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import msgspec
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from harness.hub_double import is_ready, is_result_for
from harness.receipt_hub import (  # noqa: F401 — fixtures ride along
    B3_HEX,
    COMPILED_GRAPH_KEY,
    FAMILY,
    KID,
    OTHER_ENDPOINT,
    SELF_ENDPOINT,
    SELF_ORG,
    SHA_HEX,
    SNAPSHOT,
    HubStub,
    _b64url,
    _configure,
    _identify,
    hub,
    make_artifact,
    make_claims,
    rsa_key,
    sign_receipt,
    worker_jwt_for,
)
from test_procsplit_pgw763 import (  # noqa: F401 — fixtures come with it
    CHILD_MAIN,
    SplitHarness,
    _payload,
    captured_dials,
    isolated_postmortem,
)

from gen_worker import (
    activity as activity_mod,
)
from gen_worker import (
    aot_serve,
    artifact_meta,
    author_ci,
    boot_adopt,
    cell_adopt,
    compile_cache,
    fleet_cells,
    keyset,
    lifecycle,
    mint_supervisor,
    numerics_ladder,
    numerics_probe,
    presigned_upload,
    receipts,
    rigcheck,
    worker_credential,
    worker_identity,
)
from gen_worker import (
    executor as executor_mod,
)
from gen_worker._vendor.torchcg import GRAPH_CLASS_BLOCK
from gen_worker.api.errors import ArtifactTransferError
from gen_worker.cell_adopt import AdoptOutcome
from gen_worker.hubio.transport import TransportError
from gen_worker.keyset import document as keyset_doc
from gen_worker.keyset import store as keyset_store
from gen_worker.parallel.cp import (
    ContextParallelUnavailable,
    CpComms,
    install_context_parallel,
)
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.procsplit import actions, broker
from gen_worker.topology import TopologyError

# ============================================================================
# pgw#709 — cell-receipt verification — REAL signatures, REAL HTTP,
#   no mocks.
# ============================================================================

class TestVerifyReceiptJWS:
    def test_round_trip(self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]) -> None:
        jws = sign_receipt(rsa_key, make_claims("sha256:" + SHA_HEX, 4096))
        receipt = receipts.verify_receipt_jws(jws, pub_map)
        assert receipt.compiled_graph_key == COMPILED_GRAPH_KEY
        assert receipt.family == FAMILY
        assert receipt.snapshot_digest == SNAPSHOT
        assert receipt.artifact_digest == "sha256:" + SHA_HEX
        assert receipt.artifact_size_bytes == 4096
        # claims nothing checks are not decoded. `make_claims` still
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
        forged_claims = make_claims("sha256:" + SHA_HEX, 4096, compiled_graph_key="cg-key-v1-" + "f" * 56)
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
        # pgw#1278: the version the hub signed BEFORE the compiled-graph
        # wire cut. It must refuse, not be read as a v1 one.
        jws = sign_receipt(rsa_key, make_claims("sha256:" + SHA_HEX, 4096,
                                                crv="cell-receipt-v2"))
        with pytest.raises(receipts.ReceiptError) as exc:
            receipts.verify_receipt_jws(jws, pub_map)
        assert exc.value.reason == "receipt_version_unsupported"

    def test_garbage_refused(self, pub_map: Dict[str, rsa.RSAPublicKey]) -> None:
        for junk in ("", "a.b", "not-a-jws", "a.b.c.d"):
            with pytest.raises(receipts.ReceiptError):
                receipts.verify_receipt_jws(junk, pub_map)


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
        jws = hub.receipts[(original_ref, COMPILED_GRAPH_KEY)]
        new_ref = receipts.artifact_digest(artifact)
        hub.receipts[(new_ref, COMPILED_GRAPH_KEY)] = jws
        _configure(hub)
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False

    def test_key_mismatch_refused(self, tmp_path: Path, hub: HubStub) -> None:
        # Receipt signed for a DIFFERENT key than the artifact claims: the
        # Nix Deriver lesson — key binding must be inside the signature.
        artifact = make_artifact(tmp_path)
        ref = receipts.artifact_digest(artifact)
        claims = make_claims(ref, artifact.stat().st_size, compiled_graph_key="cg-key-v1-" + "e" * 56)
        hub.receipts[(ref, COMPILED_GRAPH_KEY)] = sign_receipt(hub.key, claims)
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
        hub.revoked.append({"compiled_graph_key": COMPILED_GRAPH_KEY, "snapshot_digest": SNAPSHOT, "reason": "bad image"})
        _configure(hub)
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False

    def test_other_revocation_does_not_block(self, tmp_path: Path, hub: HubStub) -> None:
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(artifact)
        hub.revoked.append({"compiled_graph_key": "cg-key-v1-" + "d" * 56, "snapshot_digest": "other", "reason": "x"})
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
        """pgw#709: The provision.enable_compiled hook: a refused delivered artifact must be dropped BEFORE anyt..."""
        from gen_worker.models import provision

        artifact = make_artifact(tmp_path)
        _configure(hub)  # no receipt served -> refusal

        seen: Dict[str, Any] = {}

        def _dispatched(path: Any) -> Any:
            seen["dispatched"] = path
            return {}

        monkeypatch.setattr(provision, "_compiled_graph_metadata", _dispatched)

        from gen_worker import compile_cache

        monkeypatch.setattr(compile_cache, "enable", lambda pipe, cfg: False)

        class Cfg:
            family = FAMILY
            lora_bucket = 0

        armed = provision.enable_compiled(object(), Cfg(), tmp_path, artifact).armed
        assert armed is False
        assert "dispatched" not in seen, (
            "refused delivered artifact leaked through to TCG admission"
        )

    def test_enable_compiled_passes_verified_artifact(
        self, tmp_path: Path, hub: HubStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gen_worker.models import provision

        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(artifact)
        _configure(hub)

        seen: Dict[str, Any] = {}

        def _dispatched(path: Any) -> Any:
            seen["dispatched"] = path
            return {}

        monkeypatch.setattr(provision, "_compiled_graph_metadata", _dispatched)

        from gen_worker import compile_cache

        monkeypatch.setattr(compile_cache, "enable", lambda pipe, cfg: True)

        class Cfg:
            family = FAMILY
            lora_bucket = 0

        armed = provision.enable_compiled(object(), Cfg(), tmp_path, artifact).armed
        assert armed is True
        assert seen["dispatched"] == artifact


class TestAlgorithmAgnosticReceipts:
    """pgw#709: The guards that let the cell self-mint producer publish over v2."""

    def test_v2_receipt_arms_a_v2_cell(self, tmp_path: Path, hub: HubStub) -> None:
        artifact = make_artifact(tmp_path)
        ref = hub.serve_receipt_for(artifact, algo="sha256")
        assert ref.startswith("sha256:")
        _configure(hub)
        receipt = receipts.verify_delivered_artifact(artifact, FAMILY)
        assert receipt.artifact_digest == ref
        # One request carrying the ALGORITHM-TAGGED digest — no per-algorithm
        # 404 retry chain, and never bare hex.
        offered, asked_key = hub.last_query
        assert asked_key == COMPILED_GRAPH_KEY
        assert receipts.artifact_digest(artifact) in offered
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is True

    def test_legacy_blake3_receipt_is_refused_not_dual_read(
        self, rsa_key: rsa.RSAPrivateKey, pub_map: Dict[str, rsa.RSAPublicKey]
    ) -> None:
        """pgw#807: the pre-v2 receipt shape (bare-hex `blake3`, no `digest`) no longer verifies."""
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
        hub.receipts[(ref, COMPILED_GRAPH_KEY)] = sign_receipt(hub.key, claims)
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
        hub.receipts[(ref, COMPILED_GRAPH_KEY)] = sign_receipt(hub.key, claims)
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


class TestPublisherTrustTh1657:
    """pgw#709: A cell must have come from THIS endpoint, or from a publisher the platform vouches for."""

    def test_another_endpoints_org_cell_is_refused(
        self, tmp_path: Path, hub: HubStub
    ) -> None:
        """pgw#709: A genuine, correctly-signed, un-revoked receipt — for someone else."""
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
        """pgw#709: THE CONTROL."""
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id=SELF_ENDPOINT, publisher_tier="org")
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        receipt = receipts.verify_delivered_artifact(artifact, FAMILY)
        assert receipt.owning_endpoint_id == SELF_ENDPOINT
        assert receipt.publisher_tier == "org"
        assert receipts.gate_delivered_artifact(artifact, FAMILY) is True

    def test_platform_tier_arms_anywhere(self, tmp_path: Path, hub: HubStub) -> None:
        """pgw#709: Platform-tier is the escape hatch the fleet actually runs on: the platform authored that cod..."""
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
        """pgw#709: §4.24 point 4: absence must be explicit."""
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
        """A worker credential with no `cell_read_endpoint_id` (a hub too old for th#1657, or a grant that could..."""
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

    def test_a_pre_cut_receipt_is_refused_not_defaulted(
        self, tmp_path: Path, hub: HubStub
    ) -> None:
        """pgw#709: The trust fields are load-bearing, so a receipt minted under an older version must not be re..."""
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(artifact, crv="cell-receipt-v2")
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        with pytest.raises(receipts.ReceiptError) as excinfo:
            receipts.verify_delivered_artifact(artifact, FAMILY)
        assert excinfo.value.reason == "receipt_version_unsupported"

    def test_the_refusal_reaches_the_wire(
        self, tmp_path: Path, hub: HubStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pgw#824/pgw#999: a refusal nobody can count is a refusal nobody can act on."""
        events: List[Tuple[str, str, str]] = []
        monkeypatch.setattr(
            receipts.activity_mod,  # type: ignore[attr-defined]
        "emit_event",
            lambda kind, detail, phase="", **_: events.append((kind, detail, phase)))

        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id=OTHER_ENDPOINT, publisher_tier="org")
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        assert receipts.gate_delivered_artifact(artifact, FAMILY) is False
        assert events, "the refusal never reached the wire"
        kind, detail, phase = events[-1]
        assert kind == "compiled_graph_receipt_refused"
        assert phase == "publisher_untrusted"
        assert FAMILY in detail


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
        version=receipts.RECEIPT_VERSION, family=FAMILY, compiled_graph_key=COMPILED_GRAPH_KEY,
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
        """pgw#709: A guard on the TABLE, not the code: an all-true or all-false table would pass every row abov..."""
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
        """pgw#709: THE RED CASE."""
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
        """THE SAFE DEGRADATION, and the reason this needs no coupled deploy: a grant with no `cell_read_org_id`..."""
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
        """pgw#709: The endpoint rule survives untouched — an old hub's pods keep adopting their own cells."""
        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id=SELF_ENDPOINT,
            publisher_tier="org", publisher_org_id="")
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        assert receipts.gate_delivered_artifact(artifact, FAMILY) is True

    def test_receipt_version_did_not_move(self) -> None:
        """th#1678's lesson: `publisher_org_id` already shipped in v2, so this change is additive on the GRANT a..."""
        assert receipts.RECEIPT_VERSION == "compiled-graph-receipt-v1"


# ============================================================================
# pgw#1122 — the cell RECEIPT TRUST GATE runs in the process that
#   holds no credential — so it must not answer "who am I?" by decoding one.
# ============================================================================

POD_ENDPOINT = SELF_ENDPOINT


POD_ORG = SELF_ORG


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def stamped_worker_jwt(
    *, endpoint_id: str = POD_ENDPOINT, org_id: str = POD_ORG,
) -> str:
    claims: Dict[str, Any] = {
        "sub": "w-parent",
        "release_id": "rel-1122",
        "cap_kind": "worker_capability",
        "cell_read_endpoint_id": endpoint_id,
    }
    if org_id:
        claims["cell_read_org_id"] = org_id
    return (
        _b64(json.dumps({"alg": "HS256"}).encode())
        + "." + _b64(json.dumps(claims).encode())
        + ".sig"
    )


@pytest.fixture(autouse=True)
def _clean_identity() -> Iterator[None]:
    """Identity is a PROCESS fact; unwind it around every row."""
    worker_identity.reset()
    worker_credential.reset()
    yield
    worker_identity.reset()
    worker_credential.reset()
    receipts.reset()
    broker.install(None)


@pytest.fixture()
def stamped_split(tmp_path, captured_dials, monkeypatch):  # noqa: F811
    """pgw#1122: A split whose PARENT holds a credential carrying the viewer claims."""
    token = stamped_worker_jwt()
    monkeypatch.setenv("WORKER_JWT", token)
    h = SplitHarness(
        tmp_path,
        extra_child_env={"PGW763_CHILD_MODULES": "harness.procsplit_endpoints"},
    )
    h.pc._settings = msgspec.structs.replace(
        h.pc._settings, bootstrap_worker_jwt=token)
    h.pc.transport._settings = h.pc._settings
    try:
        yield h
    finally:
        h.close()


def test_the_compute_child_names_itself_without_holding_a_credential(
    stamped_split: Any,
) -> None:
    """pgw#1122: THE regression, on the production path: tenant-adjacent code inside the real compute child asks..."""
    conn = stamped_split.scheduler.wait_connection(0)
    conn.wait_for(is_ready)

    conn.send(run_job=pb.RunJob(
        request_id="r-whoami", attempt=1, function_name="who-am-i",
        input_payload=_payload()))
    got = conn.wait_for(is_result_for("r-whoami"), timeout=60.0)
    assert got.job_result.status == pb.JOB_STATUS_OK
    answer = msgspec.msgpack.decode(got.job_result.inline)["response"]

    assert answer == f"endpoint={POD_ENDPOINT} org={POD_ORG}", (
        "a compute child could not name the endpoint/org it serves "
        f"({answer!r}) — the pgw#1122 gate reads exactly this and refuses "
        "every org-tier cell when it comes back empty")


def test_the_relay_carries_the_claims_and_never_the_credential(
    stamped_split: Any,
) -> None:
    """pgw#1122: A claim is not a credential."""
    pc = stamped_split.pc
    answer = pc._viewer_identity()

    assert set(answer) == {"endpoint_id", "org_id"}
    assert answer["endpoint_id"] == POD_ENDPOINT
    assert answer["org_id"] == POD_ORG
    token = pc.transport.current_worker_jwt
    assert token, "the fixture's premise is that the PARENT holds a credential"
    assert token not in json.dumps(answer)


def test_the_parent_refuses_to_invent_an_identity_it_does_not_have(
    tmp_path: Path, captured_dials: Any,  # noqa: F811
) -> None:
    """pgw#1122: A parent with no credential REFUSES rather than answering ``("", "")``."""
    h = SplitHarness(tmp_path)
    try:
        h.pc._settings = msgspec.structs.replace(
            h.pc._settings, bootstrap_worker_jwt="")
        h.pc.transport._settings = h.pc._settings
        # pgw#893 §2 deleted the transport's stream-local credential cache;
        # `worker_credential` is now the ONE home, so emptying it is what
        # "this parent holds no credential" means.
        worker_credential.reset()
        with pytest.raises(actions.ActionRefused) as exc:
            h.pc._viewer_identity()
        assert "holds no worker credential" in str(exc.value)
    finally:
        h.close()


class _FakeParent:
    """pgw#1122: A control seam that answers exactly what the parent answers."""

    def __init__(self, endpoint_id: str, org_id: str, base_url: str = "") -> None:
        self.endpoint_id = endpoint_id
        self.org_id = org_id
        self.base_url = base_url.rstrip("/")
        self.asks: List[str] = []

    def call_action(
        self, action: str, args: Dict[str, Any], *, timeout: float = 30.0,
    ) -> Dict[str, Any]:
        self.asks.append(action)
        assert action == actions.ACTION_VIEWER_IDENTITY
        assert args == {}, "the child names no field in an identity ask"
        return {"endpoint_id": self.endpoint_id, "org_id": self.org_id}

    def call(
        self, method: str, path: str, *, params: Any = None, json: Any = None,
        timeout: float = 30.0,
    ) -> broker.HubResponse:
        """pgw#1122: The parent's half of a mediated HTTP call: it names the host, it attaches the credential, t..."""
        import requests

        self.asks.append(f"{method} {path}")
        resp = requests.request(
            method, self.base_url + path, params=params, json=json,
            headers={"Authorization": "Bearer parent-holds-this"},
            timeout=timeout)
        return broker.HubResponse(status_code=resp.status_code, text=resp.text)


def _child_gate(stub: HubStub, parent: Optional[_FakeParent]) -> None:
    """pgw#1122: Arm the receipt gate exactly as ``lifecycle.on_hello_ack`` does IN THE COMPUTE CHILD: the provi..."""
    worker_credential.reset()
    worker_identity.reset()
    receipts.configure(base_url=stub.base_url, worker_jwt=lambda: "")
    if parent is not None:
        parent.base_url = stub.base_url
    broker.install(parent)  # type: ignore[arg-type]


def test_the_child_arms_an_org_tier_cell_it_is_entitled_to(
    tmp_path: Path, hub: HubStub,  # noqa: F811
) -> None:
    """pgw#1122: THE POD FAILURE, reproduced: a resolved, materialized, correctly-owned org-tier cell reaching t..."""
    artifact = make_artifact(tmp_path)
    hub.serve_receipt_for(
        artifact, owning_endpoint_id=POD_ENDPOINT,
        publisher_tier="org", publisher_org_id=POD_ORG)
    parent = _FakeParent(POD_ENDPOINT, POD_ORG)
    _child_gate(hub, parent)

    receipt = receipts.verify_delivered_artifact(artifact, FAMILY)

    assert receipt.publisher_org_id == POD_ORG
    assert receipts.gate_delivered_artifact(artifact, FAMILY) is True
    assert parent.asks.count(actions.ACTION_VIEWER_IDENTITY) == 1, (
        "identity does not change for the life of a pod; asking per arm puts a "
        "seam round trip on every cell")


def test_a_sibling_endpoint_in_the_same_org_arms_from_the_child(
    tmp_path: Path, hub: HubStub,  # noqa: F811
) -> None:
    """th#1680's rule, now actually reachable under the split: the org matches even when the endpoint does not."""
    artifact = make_artifact(tmp_path)
    hub.serve_receipt_for(
        artifact, owning_endpoint_id=OTHER_ENDPOINT,
        publisher_tier="org", publisher_org_id=POD_ORG)
    _child_gate(hub, _FakeParent(POD_ENDPOINT, POD_ORG))

    assert receipts.verify_delivered_artifact(
        artifact, FAMILY).publisher_org_id == POD_ORG


def test_another_orgs_cell_is_still_refused_from_the_child(
    tmp_path: Path, hub: HubStub,  # noqa: F811
) -> None:
    """pgw#1122: The threat is unchanged and this fix must not widen it: the artifact is a ``.so`` this process ..."""
    artifact = make_artifact(tmp_path)
    hub.serve_receipt_for(
        artifact, owning_endpoint_id=OTHER_ENDPOINT, publisher_tier="org",
        publisher_org_id="99999999-0000-0000-0000-000000000000")
    _child_gate(hub, _FakeParent(POD_ENDPOINT, POD_ORG))

    with pytest.raises(receipts.ReceiptError) as exc:
        receipts.verify_delivered_artifact(artifact, FAMILY)
    assert exc.value.reason == "publisher_untrusted"


def test_no_identity_at_all_refuses_LOUDLY_and_by_its_own_name(
    tmp_path: Path, hub: HubStub,  # noqa: F811
) -> None:
    """pgw#1122: The structurally-impossible case: no credential here, no seam to ask over."""
    artifact = make_artifact(tmp_path)
    hub.serve_receipt_for(
        artifact, owning_endpoint_id=POD_ENDPOINT,
        publisher_tier="org", publisher_org_id=POD_ORG)
    _child_gate(hub, None)  # no parent, no credential

    with pytest.raises(receipts.ReceiptError) as exc:
        receipts.verify_delivered_artifact(artifact, FAMILY)
    assert exc.value.reason == "identity_unavailable", (
        "a pod that could not be ASKED about its identity reported the same "
        "reason as a pod whose identity does not match the publisher")

    with pytest.raises(worker_identity.IdentityUnavailable) as ident:
        worker_identity.viewer()
    assert ident.value.reason == "no_credential"


def test_a_platform_tier_cell_still_arms_with_no_identity(
    tmp_path: Path, hub: HubStub,  # noqa: F811
) -> None:
    """pgw#1122: The refusal must stay scoped to the org-tier decision it is about: a platform-tier cell needs n..."""
    artifact = make_artifact(tmp_path)
    hub.serve_receipt_for(
        artifact, owning_endpoint_id="", publisher_tier="platform",
        publisher_org_id="")
    _child_gate(hub, None)

    assert receipts.verify_delivered_artifact(artifact, FAMILY).publisher_tier \
        == "platform"


def test_a_hub_that_stamped_no_claims_is_an_ANSWER_not_a_failure(
    tmp_path: Path, hub: HubStub,  # noqa: F811
) -> None:
    """pgw#1122: ``cellgrant.Stamp`` omits both claims when the hub cannot resolve them, which legally narrows t..."""
    _child_gate(hub, _FakeParent("", ""))
    me = worker_identity.viewer()
    assert not me.named

    artifact = make_artifact(tmp_path)
    hub.serve_receipt_for(
        artifact, owning_endpoint_id=POD_ENDPOINT,
        publisher_tier="org", publisher_org_id=POD_ORG)
    with pytest.raises(receipts.ReceiptError) as exc:
        receipts.verify_delivered_artifact(artifact, FAMILY)
    assert exc.value.reason == "publisher_untrusted"


class _Events:
    """Collect the typed activity events this boot emitted."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.rows: List[Tuple[str, str, str]] = []
        monkeypatch.setattr(
            activity_mod, "emit_event",
            lambda kind, detail, phase="", duration_ms=0, **_kw: self.rows.append(
                (kind, phase, detail)))

    def phases(self, kind: str) -> List[str]:
        return [p for k, p, _ in self.rows if k == kind]

    def detail(self, kind: str, phase: str) -> str:
        return next(d for k, p, d in self.rows if k == kind and p == phase)


def _hit(family: str = FAMILY, function: str = "generate") -> Any:
    """pgw#1122: The ``BootAdoptOutcome`` §4.27 produces on a HIT — the exact object the executor now carries on..."""
    return boot_adopt.BootAdoptOutcome(
        adoption=None, reason=boot_adopt.HIT,
        derived_key="cg-key-v1-" + "f0" * 28, derive_ms=10_895,
        family=family, function=function)


def _executor(tmp_path: Path) -> Any:
    from gen_worker.executor import Executor
    from gen_worker.models.store import ModelStore

    async def _send(msg: Any) -> None:
        pass

    return Executor([], _send, store=ModelStore(_send, cache_dir=tmp_path / "cas"))


class _Cfg:
    family = FAMILY
    lora_bucket = 0


def _refusing_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*a: Any, **k: Any) -> Any:
        raise fleet_cells.OrderedArmError(
            "artifact_receipt_refused",
            "publisher_untrusted: this pod cannot name its own endpoint or org")

    monkeypatch.setattr(fleet_cells, "arm_ordered", _raise)


def test_an_adopted_cell_that_will_not_arm_does_not_kill_the_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1122: THE COST. On master this ``OrderedArmError`` escaped setup, the hub logged ``worker_function_un..."""
    events = _Events(monkeypatch)
    _refusing_arm(monkeypatch)
    monkeypatch.setattr(
        fleet_cells, "enable_compiled",
        lambda *a, **k: fleet_cells.ArmOutcome(armed=False))
    monkeypatch.setattr(
        executor_mod.compile_cache,  # type: ignore[attr-defined]
        "mandatory_serving", lambda pipe: False)
    ex = _executor(tmp_path)
    order = executor_mod._ArmOrder(
        backend="aot_cell", publisher_org=POD_ORG, adopt=_hit())

    outcome = ex._enable_compiled(object(), _Cfg(), tmp_path / "cell.tar.gz",
                                  None, order)

    assert outcome.armed is False
    assert outcome.eager_reason == cell_adopt.EagerPhase.ADOPTED_COMPILED_GRAPH_REFUSED

    # ...and it says so ON THE WIRE, under the kind that already carries the
    # rest of this journey, with the refusing gate named (pgw#1116's shape).
    assert "arm_refused" in events.phases(activity_mod.KIND_BOOT_ADOPT)
    detail = events.detail(activity_mod.KIND_BOOT_ADOPT, "arm_refused")
    assert "cause=artifact_receipt_refused" in detail
    assert "publisher_untrusted" in detail
    assert f"family={FAMILY}" in detail and "key=cg-key-v1-" in detail


def test_a_HUB_ordered_arm_stays_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1122: The other half, and the one this fix must not erode: when the HUB named an exact artifact, a su..."""
    _Events(monkeypatch)
    _refusing_arm(monkeypatch)
    ex = _executor(tmp_path)
    order = executor_mod._ArmOrder(backend="aot_cell", publisher_org=POD_ORG)

    with pytest.raises(fleet_cells.OrderedArmError) as exc:
        ex._enable_compiled(object(), _Cfg(), tmp_path / "cell.tar.gz",
                            None, order)
    assert exc.value.reason == "artifact_receipt_refused"


def test_a_mandatory_lane_still_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1122: A w8a8/w4a4 lane serves ONLY from a cell, so "boot as yesterday" is not available: degrading th..."""
    _Events(monkeypatch)
    _refusing_arm(monkeypatch)
    monkeypatch.setattr(
        executor_mod.compile_cache,  # type: ignore[attr-defined]
        "mandatory_serving", lambda pipe: True)
    ex = _executor(tmp_path)
    order = executor_mod._ArmOrder(
        backend="aot_cell", publisher_org=POD_ORG, adopt=_hit())

    with pytest.raises(fleet_cells.OrderedArmError):
        ex._enable_compiled(object(), _Cfg(), tmp_path / "cell.tar.gz",
                            None, order)


def test_the_degrade_reruns_the_ordinary_policy_with_no_delivered_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1122: "Boot as this pod booted yesterday" is a claim with a mechanism: the order is dropped and the F..."""
    _Events(monkeypatch)
    _refusing_arm(monkeypatch)
    monkeypatch.setattr(
        executor_mod.compile_cache,  # type: ignore[attr-defined]
        "mandatory_serving", lambda pipe: False)
    seen: List[Tuple[Any, ...]] = []

    def _policy(pipe: Any, cfg: Any, cache_dir: Any, artifact: Any,
                **kw: Any) -> Any:
        seen.append((artifact, kw.get("delivered_ref"),
                     kw.get("delivered_digest")))
        return fleet_cells.ArmOutcome(armed=True)

    monkeypatch.setattr(fleet_cells, "enable_compiled", _policy)
    ex = _executor(tmp_path)
    order = executor_mod._ArmOrder(
        backend="aot_cell", publisher_org=POD_ORG, adopt=_hit())

    outcome = ex._enable_compiled(object(), _Cfg(), tmp_path / "cell.tar.gz",
                                  None, order)

    assert outcome.armed is True
    assert seen == [(None, "", "")], (
        "the degrade re-offered the refused artifact to the fleet policy")


def test_arm_refused_is_in_the_boot_adopt_vocabulary() -> None:
    """pgw#1116's fence, extended: the journey's LAST terminus has to be enumerable too, or the event that promi..."""
    assert "arm_refused" in boot_adopt.REASONS


def _lint() -> Any:
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    try:
        import lint_credential_identity as lint
    finally:
        sys.path.remove(str(root / "scripts"))
    return lint


def test_the_fence_catches_a_new_unclassified_credential_read(
    tmp_path: Path,
) -> None:
    """pgw#1122: A check that cannot go red proves nothing."""
    lint = _lint()
    (tmp_path / "new_gate.py").write_text(
        "class Gate:\n"
        "    def decide(self, cfg):\n"
        "        return cfg.worker_jwt()\n",
        encoding="utf-8")

    sites = lint.scan(tmp_path)
    assert any(site.endswith("Gate.decide::worker_jwt") for _, site in sites)
    problems = lint.check(sites, {})
    assert problems and "UNCLASSIFIED worker-credential read" in problems[0]
    assert "worker_identity.viewer()" in problems[0]


def test_there_is_no_classification_that_means_identity() -> None:
    """pgw#1122: The class-closer."""
    lint = _lint()
    assert "IDENTITY" not in lint.CLASSIFICATIONS
    assert lint.RESOLVER_FILES == {"worker_identity.py"}


def test_the_live_tree_is_fully_classified() -> None:
    """pgw#1122: And the allowlist is exact in both directions: an unclassified read is red, a row matching noth..."""
    lint = _lint()
    allowed, errors = lint.load_allowlist()
    assert not errors
    assert not lint.check(lint.scan(), allowed)


# ============================================================================
# pgw#1271 — the gates that could not go red, and the caller each
#   now has.
# ============================================================================

def _class_hash(dim: int) -> str:
    """pgw#1271: One TCG class hash, in the only shape the memo accepts (16 lower hex)."""
    return hashlib.sha256(f"class-dim-{int(dim)}".encode()).hexdigest()[:16]


def _entry_block(*, dim: int = 64) -> Dict[str, Any]:
    """The `entry` block a minted artifact carries, as the seam reads it."""
    return {
        "name": "a",
        "target": "unet",
        "class_hash": _class_hash(dim),
    }


class _Cfg_pgw1271:
    """The parent's `CompileCell`, as `cfg_spec` reads it."""

    family = "tiny"
    targets = ("unet",)
    shapes = ((1024, 1024),)
    text_lens = (77,)
    guidance_scales = (7.5,)
    lora_bucket = 0


class _Pending:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.family = "tiny"


class _Row:
    def __init__(self, entry: str, block: Dict[str, Any]) -> None:
        self.entry = entry
        self.key = f"cg-{entry}"
        self.metadata = {GRAPH_CLASS_BLOCK: block}


class _Result:
    def __init__(self, rows: List[_Row]) -> None:
        self.entries = tuple(rows)


def _mint_task(tmp_path: Path) -> Any:
    return mint_supervisor.MintTask(
        pending=_Pending(tmp_path),
        pipe=object(),
        function="generate",
        modules=("endpoint",),
        slots={},
    )


@pytest.fixture()
def _runtime_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """pgw#1271: Make the runtime key-complete on a GPU-less box."""
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: {
        "sku": "l4", "sm": "sm_89", "torch": "2.13.0+cu130",
        "triton": "3.6.0", "cuda": "13.0",
        "image_digest": "sha256:" + "ab" * 32,
    })


def _write_memo(tmp_path: Path, hashes: Dict[str, str]) -> Any:
    """pgw#1271: The pod-local `cg-keyset-v1` row this machine would have answered with."""
    cfg = _Cfg_pgw1271()
    digest = keyset.closure_digest(
        "tiny", mint_supervisor.cfg_spec(cfg), function="generate", slots={},
        modules=_mint_task(tmp_path).modules)
    assert keyset_store.write_closure(
        tmp_path, digest, keyset_doc.closure_row(
            family="tiny", function="generate",
            tcg_version=keyset.tcg_version(),
            classes={
                name: keyset_doc.GraphClassRow(
                    graph_class=name, class_hash=class_hash,
                    ingress_digest="9" * 32, target="denoiser")
                for name, class_hash in hashes.items()}))
    return digest


def test_the_mint_publish_seam_rules_on_a_DISHONEST_boot_memo(
    tmp_path: Path, _runtime_key: None,
) -> None:
    """pgw#1271: THE headline."""
    digest = _write_memo(tmp_path, {"a": _class_hash(64)})

    reason = mint_supervisor.rule_on_boot_memo(
        _mint_task(tmp_path), _Cfg_pgw1271(),
        _Result([_Row("a", _entry_block(dim=128))]), declared=1)

    assert "DISHONEST" in reason and "a: cached" in reason
    # The entry is GONE: the next boot re-traces instead of answering from it.
    assert keyset_store.class_hashes(digest, cache_dir=tmp_path) == {}


def test_an_HONEST_boot_memo_is_silence_at_the_publish_seam(
    tmp_path: Path, _runtime_key: None,
) -> None:
    hashes = {"a": _class_hash(64)}
    digest = _write_memo(tmp_path, hashes)

    assert mint_supervisor.rule_on_boot_memo(
        _mint_task(tmp_path), _Cfg_pgw1271(),
        _Result([_Row("a", _entry_block(dim=64))]), declared=1) == ""
    # An honest memo SURVIVES — the whole economic point of having one.
    assert keyset_store.class_hashes(digest, cache_dir=tmp_path) == hashes


def test_a_PARTIAL_class_set_rules_on_nothing(
    tmp_path: Path, _runtime_key: None,
) -> None:
    """Coverage accretes (pgw#1176), so a mint that packed 1 of 2 declared classes cannot tell "the memo holds a..."""
    hashes = {"a": _class_hash(64), "b": _class_hash(128)}
    digest = _write_memo(tmp_path, hashes)

    assert mint_supervisor.rule_on_boot_memo(
        _mint_task(tmp_path), _Cfg_pgw1271(),
        _Result([_Row("a", _entry_block(dim=64))]), declared=2) == ""
    assert keyset_store.class_hashes(digest, cache_dir=tmp_path) == hashes


def test_the_dishonest_verdict_reaches_the_wire_as_a_TYPED_EVENT(
    tmp_path: Path, _runtime_key: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a log line: a hub-spawned pod's stdout goes nowhere (pgw#760), and a dishonest memo is a KEY-SPACE fa..."""
    from gen_worker import activity as activity_mod

    seen: List[Tuple[str, str, str]] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, **kw: seen.append(
            (kind, detail, str(kw.get("phase") or ""))))

    _write_memo(tmp_path, {"a": _class_hash(64)})
    mint_supervisor._rule_on_boot_memo(
        _mint_task(tmp_path), _Cfg_pgw1271(), _Result([_Row("a", _entry_block(dim=128))]),
        declared=1, family="tiny")

    assert [(k, p) for k, _d, p in seen] == [
        (activity_mod.KIND_BOOT_MEMO, "memo_dishonest")]
    assert "DISHONEST" in seen[0][1]


def test_the_SDK_is_not_its_own_fleet_line_authority(tmp_path: Path) -> None:
    """pgw#1271: `_collect_authorities` appended "gen-worker" to the chain, so the SDK certified the very torch ..."""
    with pytest.raises(rigcheck.FleetLineUnknown):
        rigcheck.resolve_fleet_line(start=tmp_path, endpoint_dists=())


def test_a_host_whose_DIAGNOSTIC_is_broken_still_refuses_to_measure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """pgw#1271: The refusal used to be gated on `env["driver"]` being readable, i.e."""
    monkeypatch.setattr(rigcheck, "resolve_fleet_line", lambda **_: rigcheck.FleetLine(
        torch=(2, 13, 0), cuda=(13, 0), authorities=()))
    monkeypatch.setattr(rigcheck, "resolve_environment", lambda: {
        "python": "3.12.0",
        "torch": "2.13.0+cu130",
        "cuda": "13.0",
        # nvidia-smi did not answer. The allocation still failed.
        "driver": "",
        "cuda_usable": False,
        "cuda_unusable_reason": "CUDA driver version is insufficient",
        "cuda_unusable_class": "driver_too_old",
    })

    with pytest.raises(rigcheck.CudaUnusable):
        rigcheck.assert_fleet_line("rig", start=tmp_path, stream=None)


def test_an_expired_presign_on_the_DIRECT_FINAL_leg_is_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """pgw#1271: The re-plan at `presigned_upload_file` catches `ArtifactTransferError` with `phase == "put"` an..."""
    def _boom(**_kw: Any) -> str:
        raise TransportError("presigned PUT 403", retryable=False, status_code=403)

    monkeypatch.setattr(presigned_upload, "upload_part_to_presigned_url", _boom)
    body = tmp_path / "render.png"
    body.write_bytes(b"x" * 32)

    with pytest.raises(ArtifactTransferError) as caught:
        presigned_upload._put_whole_object(
            url="https://r2.example/final",
            file_path=str(body),
            size_bytes=32,
            extra_headers={},
            on_progress=None,
            cancel_check=None,
            put_pool=None,
        )

    exc = caught.value
    # The exact predicate the re-plan loop tests. Asserted as the predicate,
    # not as two fields, because the fields only matter through it.
    assert exc.phase == "put" and getattr(exc, "status_code", None) == 403


def test_a_cancel_on_the_direct_final_leg_is_a_CANCEL(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from gen_worker.api.errors import CanceledError

    def _boom(**_kw: Any) -> str:
        raise InterruptedError("canceled")

    monkeypatch.setattr(presigned_upload, "upload_part_to_presigned_url", _boom)
    body = tmp_path / "render.png"
    body.write_bytes(b"x")

    with pytest.raises(CanceledError):
        presigned_upload._put_whole_object(
            url="https://r2.example/final", file_path=str(body), size_bytes=1,
            extra_headers={}, on_progress=None, cancel_check=None, put_pool=None)


def test_a_wedged_fabric_is_not_swallowed_as_a_free_vram_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1271: `topology.delivered_topology` raises `TopologyError` on peer access with 0.0 GB/s measured — ev..."""
    def _wedged(*_a: Any, **_kw: Any) -> Any:
        raise TopologyError(
            "topology_fabric_wedged_peer_access_zero_bandwidth",
            "peer access reported with 0.0 GB/s measured")

    monkeypatch.setattr(lifecycle, "delivered_topology", _wedged)
    with pytest.raises(TopologyError):
        lifecycle.free_vram_bytes()


def test_an_ordinary_topology_failure_is_still_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1271: The catch is NARROWED, not removed: a box with no CUDA at all must still report 0 rather than f..."""
    def _no_torch(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(lifecycle, "delivered_topology", _no_torch)
    assert lifecycle.free_vram_bytes() == 0


def _report(cosine: float, *, measured: bool = True) -> numerics_probe.CompiledGraphNumerics:
    thresholds = numerics_ladder.Thresholds(
        floor=0.90, warn=0.99, label="test",
        source=numerics_ladder.SOURCE_SDK_DEFAULT)
    comparison = numerics_ladder.Comparison(
        reference="eager", subject="unet", thresholds=thresholds,
        rows=(numerics_ladder.RowStat(
            name="out", elements=16, cosine=cosine, retention=1.0),),
        cosine=cosine, retention=1.0)
    axis = numerics_probe.ProbeAxis(entry="unet", target="unet")
    verdict = numerics_probe.AxisVerdict(
        axis=axis, comparison=comparison if measured else None,
        reason="" if measured else "compiled_graph_forward_failed")
    return numerics_probe.CompiledGraphNumerics(
        family="tiny", compiled_graph_key="cg-1", thresholds=thresholds,
        threshold_source=thresholds.source, verdicts=(verdict,), axes_total=1)


class _Subject:
    def __init__(self, pipe: Any) -> None:
        self._pipe = pipe

    def armed_pipeline(self) -> Any:
        return self._pipe


def test_a_DEGRADED_mint_report_does_not_report_PASS(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1271: `read_parity` set `passed = True` in the mint branch from the mere EXISTENCE of a report — the ..."""
    monkeypatch.setattr(aot_serve, "entry_states", lambda _p: {
        "unet": {"state": "armed", "target": "unet", "calls": 1}})

    parity = author_ci.read_parity(
        cast(Any, _Subject(object())), declaration=None, minted=_report(0.95))

    assert parity.passed is False
    assert parity.cosine == pytest.approx(0.95)


def test_a_HEALTHY_mint_report_still_reports_PASS(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aot_serve, "entry_states", lambda _p: {
        "unet": {"state": "armed", "target": "unet", "calls": 1}})
    assert author_ci.read_parity(
        cast(Any, _Subject(object())), declaration=None, minted=_report(0.999)).passed is True


def test_a_DE_ARMED_entry_fails_the_parity_whatever_the_cosine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1271: The cross-check."""
    monkeypatch.setattr(aot_serve, "entry_states", lambda _p: {
        "unet": {"state": "armed", "target": "unet", "calls": 1},
        "vae.decode": {"state": "de_armed", "target": "vae",
                       "reason": "ingress_contract"},
    })
    parity = author_ci.read_parity(
        cast(Any, _Subject(object())), declaration=None, minted=_report(0.999))
    assert parity.passed is False
    assert "NOT armed" in parity.detail


class _Expert:
    """A sharding candidate: declares a `_cp_plan` and `enable_parallelism`."""

    _cp_plan = {"hidden_states": object()}

    def __init__(self, heads: int) -> None:
        self.config = type("Cfg", (), {"num_attention_heads": heads})()

    def enable_parallelism(self, *_a: Any, **_k: Any) -> None:  # pragma: no cover
        raise AssertionError("the group-aware install replaces this")


class _Pipeline:
    def __init__(self, heads: int) -> None:
        self.transformer = _Expert(heads)


def test_an_indivisible_HEAD_COUNT_is_refused_before_the_pod_commits() -> None:
    """diffusers #12536: the all-to-all shards the head dimension."""
    comms = CpComms(pg=object(), rank=0, device="cuda:0")
    with pytest.raises(ContextParallelUnavailable, match="head count"):
        install_context_parallel(_Pipeline(heads=6), degree=4, comms=comms)


def test_a_divisible_head_count_gets_past_the_divisibility_gate() -> None:
    """pgw#1271: Proves the gate is not a blanket refusal: heads=8 at degree 4 clears it and the call proceeds i..."""
    comms = CpComms(pg=object(), rank=0, device="cuda:0")
    with pytest.raises(Exception) as caught:
        install_context_parallel(_Pipeline(heads=8), degree=4, comms=comms)
    assert "head count" not in str(caught.value)


def test_every_axis_a_pt2_is_pinned_to_is_stated_by_the_one_probe() -> None:
    """pgw#1271: `aot_serve.IDENTITY_AXES` names the facts an exported `.pt2` is pinned to."""
    probe = set(compile_cache.runtime_key())
    missing = [a for a in aot_serve.IDENTITY_AXES if a not in probe]
    assert missing == [], (
        f"{missing} pin an artifact but `compile_cache.runtime_key()` cannot "
        f"state them; it probes {sorted(probe)}")


class TestReceiptTrustGate:
    """Driven through the real gate against the real signing hub."""

    def test_the_platform_tier_path_makes_no_VACUOUS_trust_call(
        self, tmp_path: Path, hub: HubStub, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """pgw#1271: `verify_delivered_artifact` called `refuse_untrusted_publisher(receipt, "", "")` INSIDE tha..."""
        calls: List[Tuple[str, str]] = []
        real = receipts.refuse_untrusted_publisher

        def _record(receipt: Any, endpoint_id: str, org_id: str = "") -> None:
            calls.append((endpoint_id, org_id))
            real(receipt, endpoint_id, org_id)

        monkeypatch.setattr(receipts, "refuse_untrusted_publisher", _record)

        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(artifact, publisher_tier="platform")
        _configure(hub)

        receipt = receipts.verify_delivered_artifact(artifact, FAMILY)
        assert receipt.publisher_tier == "platform"
        assert calls == [], (
            "the platform branch made a trust call that can only ever return")


    def test_an_ORG_tier_cell_still_reaches_the_real_trust_gate(
        self, tmp_path: Path, hub: HubStub, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """pgw#1271: The other half, so the deletion above cannot have removed enforcement: an org-tier receipt ..."""
        calls: List[Tuple[str, str]] = []
        real = receipts.refuse_untrusted_publisher

        def _record(receipt: Any, endpoint_id: str, org_id: str = "") -> None:
            calls.append((endpoint_id, org_id))
            real(receipt, endpoint_id, org_id)

        monkeypatch.setattr(receipts, "refuse_untrusted_publisher", _record)

        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id="ep_someone_else",
            publisher_org_id="org_someone_else")
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        with pytest.raises(receipts.ReceiptError):
            receipts.verify_delivered_artifact(artifact, FAMILY)
        assert calls and calls[0][0] == SELF_ENDPOINT


# ============================================================================
# pgw#1098 — An UNREADABLE cell envelope must refuse BY NAME, not vanish.
# ============================================================================

def _cell(path: Path, meta: Dict[str, Any], *, pad_to: int = 0) -> Path:
    """pgw#1098: A tarball carrying `metadata.json` at the root, optionally padded so the member's DECLARED size..."""
    if pad_to:
        meta = dict(meta)
        blob = json.dumps(meta).encode()
        meta["_pad"] = "x" * max(0, pad_to - len(blob) - 16)
    payload = json.dumps(meta).encode()
    with tarfile.open(path, mode="w:gz") as tar:
        info = tarfile.TarInfo(artifact_meta.METADATA_NAME)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return path


_ROW7_META: Dict[str, Any] = {
    "format": 2,
    "kind": "aot-inductor",
    "family": "sdxl",
    "compiled_graph_key": "ck1-" + "a" * 56,
    "lora_bucket": 64,
    "entries": {
        "unet/adapter=true,cfg=true/B=2,H_lat=128,T_txt=77,W_lat=128": {
            "target": "unet",
        },
    },
}


def test_a_36_entry_sdxl_scale_envelope_is_readable(tmp_path: Path) -> None:
    """pgw#1098: RED pre-fix: 16 MiB refused row 7's envelope."""
    artifact = _cell(tmp_path / "cell.tar.gz", _ROW7_META, pad_to=20 << 20)

    meta = artifact_meta.read_metadata(artifact)

    assert meta["compiled_graph_key"] == _ROW7_META["compiled_graph_key"]
    assert meta["entries"]["unet/adapter=true,cfg=true/B=2,H_lat=128,"
                           "T_txt=77,W_lat=128"]["target"] == "unet"


def test_the_bound_is_a_memory_bound_not_the_declare_bound() -> None:
    """pgw#1098: The derivation, pinned."""
    from gen_worker import fleet_cells

    assert artifact_meta.MAX_METADATA_BYTES >= 16 * fleet_cells.CELL_DECLARE_MAX_BYTES
    # Still bounded, and still well under a decompression bomb's scale:
    # pgw#1013's OOM threat is real and must not be reopened.
    assert artifact_meta.MAX_METADATA_BYTES < (128 << 20)


def test_an_oversized_envelope_still_refuses_before_decompressing(
    tmp_path: Path,
) -> None:
    """The threat pgw#1013 closed stays closed, and names its bound."""
    artifact = _cell(
        tmp_path / "huge.tar.gz", _ROW7_META,
        pad_to=artifact_meta.MAX_METADATA_BYTES + (1 << 20))

    with pytest.raises(artifact_meta.ArtifactMetadataError) as excinfo:
        artifact_meta.read_metadata(artifact)

    assert str(artifact_meta.MAX_METADATA_BYTES) in str(excinfo.value)


def test_there_is_ONE_envelope_reader_and_it_is_BOUNDED(tmp_path: Path) -> None:
    """pgw#1098: RED pre-fix: `aot_serve.unpack_metadata` kept its own UNBOUNDED scan, so on row 7's cell the bo..."""
    from gen_worker import aot_serve

    artifact = _cell(tmp_path / "cell.tar.gz", _ROW7_META, pad_to=20 << 20)
    assert artifact_meta.read_metadata(artifact)["family"] == _ROW7_META["family"]

    over = _cell(
        tmp_path / "over.tar.gz", _ROW7_META,
        pad_to=artifact_meta.MAX_METADATA_BYTES + (1 << 20))
    # It refuses, and it does not answer "there are no facts here".
    with pytest.raises(artifact_meta.ArtifactMetadataError):
        artifact_meta.read_metadata(over)

    assert not hasattr(aot_serve, "unpack_metadata"), (
        "a second envelope reader is back; pgw#1098 is that asymmetry")


class _Cfg_pgw1098:
    family = "sdxl"
    lora_bucket = 64
    targets = ("unet",)


def test_arm_aot_refuses_a_declared_bucket_it_cannot_resolve_a_target_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1098: RED pre-fix: this returned `lifted_inputs_unbindable` with NO root."""
    from gen_worker import aot_serve
    from gen_worker.models import lora_lifted, provision

    class _Pipe:
        def __init__(self) -> None:
            self.unet = object()

    monkeypatch.setattr(provision, "arm_route", lambda mode: object())
    monkeypatch.setattr(lora_lifted, "branch_targets", lambda p: {"unet": p.unet})
    monkeypatch.setattr(
        aot_serve, "enable",
        lambda *a, **k: AdoptOutcome.miss(
            "lifted_inputs_unbindable",
            "artifact declares lifted adapter input(s) ['lora_a', 'lora_b'] "
            "but the module has no lifted binding to supply them"))

    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"not a tarball")   # => metadata unreadable, meta=None

    outcome = provision.arm_aot(_Pipe(), _Cfg_pgw1098(), None, artifact, 64, None)

    assert not outcome.armed
    # The gate that noticed is still named — it really did refuse...
    assert outcome.reason == "lifted_inputs_unbindable"
    # ...but the refusal now carries WHY the binding was never installed.
    assert "root:" in outcome.detail
    assert "no lifted target resolved" in outcome.detail
    assert "unreadable" in outcome.detail


def test_adopt_delegated_mint_refuses_an_unreadable_envelope_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED pre-fix: `meta=None` flowed past the pgw#1042 divergence check into an arm that could not succeed, an..."""
    from gen_worker import fleet_cells

    armed_calls: list = []

    def _record_arm(*a: Any, **k: Any) -> Any:
        armed_calls.append(a)
        return AdoptOutcome.miss("x", "y")

    monkeypatch.setattr(fleet_cells.provision, "arm_aot", _record_arm)

    target = tmp_path / "adopted.tar.gz"
    mint_root = tmp_path / "mint-root"
    mint_root.mkdir()
    produced = tmp_path / "cell.tar.gz"
    # Over the bound: readable bytes, refused envelope. The distinction the
    # pre-fix tree could not express.
    _cell(produced, _ROW7_META,
          pad_to=artifact_meta.MAX_METADATA_BYTES + (1 << 20))

    pending = fleet_cells.PendingSelfMint(
        family="sdxl", arm_token="arm1-" + "b" * 40,
        ref="repo#arm1", cfg=_Cfg_pgw1098(), target=target, mint_root=mint_root,
        publisher=None, cache_dir=tmp_path / "cache", arm_key=None)

    minted = fleet_cells.adopt_delegated_mint(object(), pending, [produced])

    assert minted is None
    reason, why = fleet_cells.adopt_refusal(pending)
    assert reason == "compiled_graph_envelope_unreadable"
    assert artifact_meta.METADATA_NAME in why
    # (b): refused BEFORE the arm, so no gate downstream of the read can be
    # blamed for a fact it was never given.
    assert armed_calls == []
