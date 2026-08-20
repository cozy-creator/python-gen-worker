"""pgw#1013 wave 3 — the two unbounded reads that live INSIDE the artifact
verifier, which is the worst place in the repo for one.

`receipts.verify_delivered_artifact` is the integrity gate for every
hub-delivered compiled graph. Both of the reads it performs before it can compare
anything were sizeless:

* the packed `metadata.json` — read to obtain the envelope it will verify, off
  a gzip member whose declared size nothing looked at. A 50 GB zero-filled
  member costs a few MB on the wire, so the pod OOMs inside the verifier,
  against an artifact whose digest had not been checked yet;
* the hub JWKS `n` — `int.from_bytes` straight into `cryptography`, so a
  multi-MB modulus makes every LATER verification super-linear for the life of
  the cached key, with nothing downstream ever refusing it.

Both are exercised on real bytes: a real gzip tarball built here, and a real
JWKS document served through the real `_fetch_jwks` path.
"""

from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

import pytest

from gen_worker import artifact_meta, receipts


# ---------------------------------------------------------------------------
# the packed envelope
# ---------------------------------------------------------------------------


def _artifact(tmp_path: Path, meta_bytes: bytes, name: str = "graph.tar.gz") -> Path:
    """A real gzip-compressed artifact carrying `meta_bytes` as its
    metadata.json, plus a payload member so it is shaped like a compiled graph."""
    path = tmp_path / name
    with tarfile.open(path, mode="w:gz") as tar:
        info = tarfile.TarInfo(artifact_meta.METADATA_NAME)
        info.size = len(meta_bytes)
        tar.addfile(info, io.BytesIO(meta_bytes))
        payload = b"weights"
        pinfo = tarfile.TarInfo("payload.bin")
        pinfo.size = len(payload)
        tar.addfile(pinfo, io.BytesIO(payload))
    return path


def test_a_legitimate_envelope_still_reads(tmp_path: Path):
    meta = {"kind": "aot", "entries": {f"e{i}": {"shape": [1, 4, 128, 128]}
                                       for i in range(64)}}
    raw = json.dumps(meta).encode()
    assert len(raw) < artifact_meta.MAX_METADATA_BYTES
    assert artifact_meta.read_metadata(_artifact(tmp_path, raw)) == meta


def test_a_decompression_bomb_in_the_envelope_is_refused_before_it_is_read(
    tmp_path: Path,
):
    # 128 MiB of zeroes: ~128 KiB on disk after gzip, 8x the bound.
    bomb = b"\0" * (128 << 20)
    artifact = _artifact(tmp_path, bomb)
    on_disk = artifact.stat().st_size
    assert on_disk < 2 << 20, (
        f"the bomb must be cheap on the wire for the test to mean anything "
        f"({on_disk} bytes)")

    with pytest.raises(artifact_meta.ArtifactMetadataError) as exc:
        artifact_meta.read_metadata(artifact)
    assert str(artifact_meta.MAX_METADATA_BYTES) in str(exc.value)
    assert "refused before decompressing it" in str(exc.value)


def test_the_verifier_itself_refuses_the_bomb_with_a_typed_reason(
    tmp_path: Path,
):
    """The read that matters is the one INSIDE `verify_delivered_artifact`;
    it must surface as a named refusal, not as a dead pod."""
    # pgw#1098 raised the bound from 16 MiB to 64 MiB (the 16 MiB one refused
    # a REAL 36-entry sdxl envelope and cost a 92-minute mint). The bomb has
    # to stay above the bound for this test to mean anything; its size was
    # always incidental, the typed refusal is the point.
    artifact = _artifact(tmp_path, b"\0" * (128 << 20))
    with pytest.raises(receipts.ReceiptError) as exc:
        receipts._embedded_meta(artifact)
    assert exc.value.reason == "artifact_unreadable"
    assert str(artifact_meta.MAX_METADATA_BYTES) in str(exc.value)


def test_the_bound_is_exact_at_both_sides(tmp_path: Path):
    """A member EQUAL to the bound is admitted; one byte more is not — so the
    refusal cannot be satisfied by a smaller legitimate envelope drifting."""
    limit = artifact_meta.MAX_METADATA_BYTES
    pad_key = '{"pad":"'
    at = (pad_key + "x" * (limit - len(pad_key) - 2) + '"}').encode()
    assert len(at) == limit
    assert artifact_meta.read_metadata(_artifact(tmp_path, at, "at.tar.gz"))["pad"]

    over = (pad_key + "x" * (limit - len(pad_key) - 1) + '"}').encode()
    assert len(over) == limit + 1
    with pytest.raises(artifact_meta.ArtifactMetadataError):
        artifact_meta.read_metadata(_artifact(tmp_path, over, "over.tar.gz"))


# ---------------------------------------------------------------------------
# the hub JWKS
# ---------------------------------------------------------------------------


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _jwk(modulus_bits: int, kid: str = "hub-1") -> dict:
    n = (1 << (modulus_bits - 1)) | 1  # top bit set, odd
    return {
        "kty": "RSA", "kid": kid,
        "n": _b64u(n.to_bytes(modulus_bits // 8, "big")),
        "e": _b64u((65537).to_bytes(3, "big")),
    }


@pytest.mark.parametrize("bits", [2048, 3072, 4096, 8192])
def test_every_modulus_the_hub_could_legitimately_publish_is_accepted(bits: int):
    key = receipts._rsa_key_from_jwk(_jwk(bits))
    assert key is not None and key.key_size == bits


def test_a_multi_megabyte_modulus_is_a_typed_refusal_not_a_slow_pod():
    with pytest.raises(receipts.ReceiptError) as exc:
        receipts._rsa_key_from_jwk(_jwk(1 << 20))  # 1 Mibit = 128 KiB of `n`
    assert exc.value.reason == "jwks_modulus_oversized"
    assert "hub-1" in str(exc.value)


def test_an_oversized_exponent_is_refused_by_the_same_bound():
    jwk = _jwk(2048)
    jwk["e"] = _b64u(b"\x01" * (16 << 10))
    with pytest.raises(receipts.ReceiptError) as exc:
        receipts._rsa_key_from_jwk(jwk)
    assert exc.value.reason == "jwks_modulus_oversized"


def test_the_fetch_path_refuses_the_whole_document_rather_than_skipping(
    monkeypatch: pytest.MonkeyPatch,
):
    """A hub publishing an absurd key is a fact about the hub. Dropping it and
    keeping the siblings would leave the pod verifying against whichever key
    happened to parse — the vacuous shape this module refuses elsewhere."""

    class _Resp:
        status_code = 200

        def json(self) -> dict:
            return {"keys": [_jwk(2048, "good"), _jwk(1 << 20, "absurd")]}

    monkeypatch.setattr(receipts.requests, "get", lambda *a, **k: _Resp())
    cfg = receipts._Config(base_url="http://hub.invalid", worker_jwt=lambda: "")
    with pytest.raises(receipts.ReceiptError) as exc:
        receipts._fetch_jwks(cfg)
    assert exc.value.reason == "jwks_modulus_oversized"
