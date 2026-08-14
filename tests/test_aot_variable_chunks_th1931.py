"""AOT delivery consumes recorded HashRepo object lengths, never a fixed layout."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from hashrepo import LocalCAS, TransferGrant, TransferReport

from gen_worker import aot_delivery


def test_aot_delivery_materializes_small_variable_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pieces = (b"header", b"tensor-body")
    payload = b"".join(pieces)
    whole = "sha256:" + hashlib.sha256(payload).hexdigest()
    remotes = tuple(
        SimpleNamespace(
            sha256=hashlib.sha256(piece).hexdigest(),
            len=len(piece),
            url=f"https://objects.invalid/{index}",
        )
        for index, piece in enumerate(pieces)
    )
    presigned = SimpleNamespace(
        files=[
            SimpleNamespace(
                path="compiled-graph.tar.gz",
                size_bytes=len(payload),
                url="",
                chunks=remotes,
                # Deliberately unrelated to either exact object length.
                chunk_size_bytes=2048,
            )
        ]
    )
    by_digest = {
        "sha256:" + hashlib.sha256(piece).hexdigest(): piece for piece in pieces
    }

    def fake_download(
        grants: Sequence[TransferGrant], cas: LocalCAS
    ) -> TransferReport:
        assert [grant.size_bytes for grant in grants] == [len(piece) for piece in pieces]
        for grant in grants:
            cas.put_bytes(by_digest[str(grant.digest)], expected=grant.digest)
        return TransferReport(
            examined=len(grants),
            succeeded=len(grants),
            bytes_transferred=len(payload),
        )

    monkeypatch.setattr(aot_delivery, "download", fake_download)
    monkeypatch.setattr(
        aot_delivery.receipts,
        "verify_delivered_artifact",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        aot_delivery.compiled_graph_store,
        "store",
        lambda artifact, **_kwargs: SimpleNamespace(artifact=artifact),
    )

    output = aot_delivery._materialize_named_artifact(
        "cg-key-v1-" + "a" * 56,
        "sdxl",
        "repo/family#compiled-graph",
        whole,
        presigned,
        receipt=object(),  # type: ignore[arg-type]
        cache_dir=tmp_path / "cache",
        what="th#1931 variable-layout proof",
    )

    assert output.read_bytes() == payload
