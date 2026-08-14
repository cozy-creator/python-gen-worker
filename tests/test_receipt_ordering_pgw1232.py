"""A signed receipt is typed before transport and stays typed through arm."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gen_worker import aot_delivery, cell_resolve, receipts


KEY = "cg-key-v1-" + "a" * 56
DIGEST = "sha256:" + "b" * 64
SNAPSHOT = "sha256:" + "c" * 64


def _receipt() -> receipts.Receipt:
    return receipts.Receipt(
        version=receipts.RECEIPT_VERSION,
        family="sdxl",
        compiled_graph_key=KEY,
        identity_axes=(("graph", "g"), ("sm", "s"), ("toolchain", "t")),
        owning_endpoint_id="endpoint-1",
        publisher="cozy",
        publisher_tier=receipts.PUBLISHER_TIER_PLATFORM,
        publisher_org_id="org-1",
        snapshot_digest=SNAPSHOT,
        artifact_path="graphs/model.tar.gz",
        artifact_digest=DIGEST,
        artifact_size_bytes=123,
        issued_at_unix=1,
    )


def _hit() -> dict[str, Any]:
    return {
        "status": "hit",
        "found": True,
        "family": "sdxl",
        "compiled_graph_key": KEY,
        "compiled_graph_ref": "repo#graph",
        "content_digest": DIGEST,
        "receipt": "header.payload.signature",
        "transport": {
            "snapshot_digest": SNAPSHOT,
            "files": [{
                "path": "graphs/model.tar.gz",
                "size_bytes": 123,
                "digest": DIGEST,
                "url": "https://objects.invalid/graph",
            }],
        },
    }


def test_resolve_verifies_every_signed_binding_before_returning_a_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _receipt()
    observed: dict[str, Any] = {}

    def verify(jws: str, **bindings: Any) -> receipts.Receipt:
        observed.update(jws=jws, **bindings)
        return verified

    monkeypatch.setattr(receipts, "verify_receipt", verify)
    answer = cell_resolve._answer_from(_hit(), KEY, "sdxl")

    assert answer.hit
    assert answer.compiled_graph is not None
    assert answer.compiled_graph.receipt is verified
    assert observed == {
        "jws": "header.payload.signature",
        "family": "sdxl",
        "compiled_graph_key": KEY,
        "snapshot_digest": SNAPSHOT,
        "artifact_path": "graphs/model.tar.gz",
        "artifact_digest": DIGEST,
        "artifact_size_bytes": 123,
    }


def test_bad_receipt_refuses_only_its_answer_before_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        receipts,
        "verify_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            receipts.ReceiptError("receipt_signature_invalid", "bad signature")
        ),
    )

    answer = cell_resolve._answer_from(_hit(), KEY, "sdxl")

    assert not answer.hit
    assert answer.status == cell_resolve.STATUS_INCOMPLETE
    assert answer.refusal_code == "compiled_graph_resolve_incomplete"
    assert "receipt_signature_invalid" in answer.detail


def test_materialize_threads_the_same_verified_object_to_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _receipt()
    hit = _hit()
    monkeypatch.setattr(receipts, "verify_receipt", lambda *_a, **_k: verified)
    answer = cell_resolve._answer_from(hit, KEY, "sdxl")
    assert answer.compiled_graph is not None
    observed: dict[str, Any] = {}

    def materialize(*_args: Any, **kwargs: Any) -> Path:
        observed.update(kwargs)
        return Path("/tmp/admitted.tar.gz")

    monkeypatch.setattr(aot_delivery, "materialize_named_artifact", materialize)
    result = cell_resolve.materialize(answer.compiled_graph, cache_dir=None)

    assert result == Path("/tmp/admitted.tar.gz")
    assert observed["receipt"] is verified


def test_delivery_binds_bytes_before_tcg_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _receipt()
    incoming = tmp_path / "aot-compiled-graphs" / ".incoming"
    incoming.mkdir(parents=True)
    artifact = incoming / ("b" * 64 + ".tar.gz")
    artifact.write_bytes(b"compiled graph")
    order: list[str] = []

    class _CAS:
        root = tmp_path

        def put_file(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    monkeypatch.setattr(aot_delivery, "open_worker_cas", lambda _root: _CAS())
    monkeypatch.setattr(
        receipts,
        "verify_delivered_artifact",
        lambda path, family, receipt: (
            order.append("receipt") or verified
        ),
    )

    monkeypatch.setattr(
        aot_delivery.compiled_graph_store,
        "store",
        lambda *_args, **_kwargs: (
            order.append("tcg_import") or SimpleNamespace(artifact=artifact)
        ),
    )

    got = aot_delivery._materialize_named_artifact(
        KEY,
        "sdxl",
        "repo#graph",
        DIGEST,
        None,
        receipt=verified,
        cache_dir=None,
        what="test",
    )

    assert got == artifact
    assert order == ["receipt", "tcg_import"]
