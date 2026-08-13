"""th#1303 phase 3: the CONVERSION producer (`publish_flavors`) publishes v2.

The mirror was producer class 1 (`clone.py`'s full-clone arm, shipped in
0.79.0). This is class 2 — every quantize / fuse / cast / distil job in the
conversion endpoint reaches the hub through `publish_flavors`, so while it
still called `commit()` the corpus repoint had a second, higher-volume tap
still filling the blake3 CAS behind it.

Driven end to end against the shared fake hub, which ENFORCES LIKE R2 on the
PUT (bytes that do not hash to the key are refused and nothing is stored) —
the same stub the mirror's own v2 tests use, so "the producer flipped" and
"the protocol still works" are one assertion, not two.

Revert-turns-red: put `client.commit(` back in `publish_flavors` and
`test_publish_flavors_declares_v2` fails on the empty `/publishes` state, and
`test_no_blake3_leaves_the_conversion_producer` fails on the resurrected
`commit_requests`.

    pytest tests/convert/test_producer_flip_th1303.py -q
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from gen_worker.convert.produced import ProducedFlavor
from gen_worker.convert.publish import publish_flavors

from fake_hub import _FakeHub


class _Ctx:
    """Just enough ctx for `HubClient.from_ctx` plus a log capture."""

    def __init__(self, base_url: str) -> None:
        self._file_api_base_url = base_url
        self._worker_capability_token = "cap-token"
        self.owner = "acme"
        self.lines: list[str] = []

    def log(self, message: str, **fields: Any) -> None:
        self.lines.append(message)


def _tree(tmp_path: Path) -> Path:
    """A produced flavor tree: two shards, one config."""
    out = tmp_path / "fp8"
    out.mkdir()
    (out / "diffusion.safetensors").write_bytes(b"\x11" * 5000)
    (out / "text_encoder.safetensors").write_bytes(b"\x22" * 3000)
    (out / "config.json").write_text('{"architectures": ["Fake"]}')
    return out


def _publish(ctx: _Ctx, tree: Path, **kw: Any) -> Any:
    return publish_flavors(
        ctx,
        [ProducedFlavor(
            path=str(tree), flavor="fp8-w8a8",
            attributes={"dtype": "fp8", "quantization_method": "w8a8",
                        "quantization_library": "llm-compressor"},
        )],
        destination_repo="acme/quant",
        tags=["prod"],
        **kw,
    )


def test_publish_flavors_declares_v2(fake_hub: Any, tmp_path: Path) -> None:
    """The conversion producer speaks the chunked-sha256 declare, and every
    declared digest is the real sha256 of the bytes on disk."""
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    tree = _tree(tmp_path)

    results = _publish(ctx, tree)

    assert len(results) == 1
    req = _FakeHub.state["publish_request"]
    declared = {f["path"]: f for f in req["files"]}
    assert set(declared) == {"diffusion.safetensors",
                             "text_encoder.safetensors", "config.json"}
    for name, entry in declared.items():
        want = hashlib.sha256((tree / name).read_bytes()).hexdigest()
        assert entry["digest"] == f"sha256:{want}", name
        assert entry["size_bytes"] == (tree / name).stat().st_size

    # The R2-shaped enforcement actually ran: the objects only exist because
    # bytes that hashed to the key were PUT to a checksum-signed URL.
    cas = _FakeHub.state["v2_cas"]
    assert len(cas) == 3
    for name in declared:
        assert hashlib.sha256((tree / name).read_bytes()).hexdigest() in cas

    assert results[0].checkpoint_id.startswith("sha256:")
    assert results[0].uploaded == 3


def test_no_blake3_leaves_the_conversion_producer(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """The point of the flip: nothing in this producer touches the v1 route or
    declares a blake3 digest any more. A partial flip is invisible without
    this — a v2 publish and a v1 commit both end in a checkpoint id."""
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    _publish(ctx, _tree(tmp_path))

    assert not _FakeHub.state.get("commit_requests"), \
        "the conversion producer must not reach the v1 /commits route"
    body = repr(_FakeHub.state["publish_request"])
    assert "blake3" not in body


def test_flavor_identity_and_provenance_survive_the_flip(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """The flip is a transport change, not a semantics change: the th#597
    selector row (flavor/dtype/tags) and the th#606 worker-addable provenance
    stamp ride v2 exactly as they rode v1."""
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    _publish(ctx, _tree(tmp_path))

    req = _FakeHub.state["publish_request"]
    # pgw#1159: `flavor` is GONE from the body; `dtype` is the axis the hub
    # actually records.
    assert "flavor" not in req
    assert req["dtype"] == "fp8"
    assert req["mode"] == "replace"          # th#597 C2 default, unchanged
    assert req["tags"] == [{"tag": "prod"}]
    assert req["provenance"] == {
        "quantization_method": "w8a8",
        "quantization_library": "llm-compressor",
    }
    # Orchestrator-derived lineage is never sendable from the worker (th#1331).
    assert "parents" not in req["provenance"]
    assert "derivation_op" not in req["provenance"]


def test_corrupt_bytes_cannot_be_published_under_a_clean_digest(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What v1 could not do. A producer that declares one digest and ships
    other bytes is refused BY THE STORE, and the object does not exist
    afterwards — so a lying conversion job cannot mint a checkpoint at all.
    Under `commit()` the same substitution was a server-side after-the-fact
    verification the client could outrun."""
    import gen_worker.models.chunk_upload as cu

    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    tree = _tree(tmp_path)

    # Substitute at the UPLOAD read only — the declaration already carries the
    # honest sha256, which is exactly the hostile shape (th#1305's class).
    real_upload = cu.upload_grants

    def swapped(grants: Any, source_for: Any, **kw: Any) -> Any:
        def corrupt(digest: str) -> Any:
            path, off, ln = source_for(digest)
            bad = tmp_path / "corrupt.bin"
            bad.write_bytes(b"\x99" * ln)
            return (bad, 0, ln)
        return real_upload(grants, corrupt, **kw)

    monkeypatch.setattr(cu, "upload_grants", swapped)

    from gen_worker.hubio.client import HubPublishError

    with pytest.raises(HubPublishError):
        _publish(ctx, tree)
    assert not _FakeHub.state.get("v2_cas"), \
        "R2 must refuse the substituted bytes; nothing may be stored"
