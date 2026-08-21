"""The CONVERSION producer (`publish_flavors`) publishes v2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from fake_hub import _FakeHub

from gen_worker.convert.produced import ProducedFlavor
from gen_worker.convert.publish import publish_flavors


class _Ctx:

    def __init__(self, base_url: str) -> None:
        self._file_api_base_url = base_url
        self._worker_capability_token = "cap-token"
        self.owner = "acme"
        self.lines: list[str] = []

    def log(self, message: str, **fields: Any) -> None:
        self.lines.append(message)


def _tree(tmp_path: Path) -> Path:
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
            path=str(tree),
            attributes={"dtype": "fp8", "precision_class": "fp8",
                        "quantization_method": "w8a8",
                        "quantization_library": "llm-compressor"},
        )],
        destination_repo="acme/quant",
        release=kw.pop("release", "r1"),
        **kw,
    )


def test_publish_flavors_declares_v2(fake_hub: Any, tmp_path: Path) -> None:
    """The conversion producer speaks the chunked-sha256 declare, and every declared digest is the real sha256 of the bytes on disk."""
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

    cas = _FakeHub.state["v2_cas"]
    assert len(cas) == 3
    for name in declared:
        assert hashlib.sha256((tree / name).read_bytes()).hexdigest() in cas

    assert results[0].checkpoint_id.startswith("sha256:")
    assert results[0].uploaded == 3


def test_no_blake3_leaves_the_conversion_producer(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """The point of the flip: nothing in this producer touches the v1 route or declares a blake3 digest any more."""
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    _publish(ctx, _tree(tmp_path))

    assert not _FakeHub.state.get("commit_requests"), \
        "the conversion producer must not reach the v1 /commits route"
    body = repr(_FakeHub.state["publish_request"])
    assert "blake3" not in body


def test_flavor_identity_and_provenance_survive_the_flip(
    fake_hub: Any, tmp_path: Path,
) -> None:
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    _publish(ctx, _tree(tmp_path))

    req = _FakeHub.state["publish_request"]
    assert "flavor" not in req
    assert req["dtype"] == "fp8"
    assert req["mode"] == "replace"
    assert "tags" not in req
    assert req["release"] == "r1"
    assert req["provenance"] == {
        "quantization_method": "w8a8",
        "quantization_library": "llm-compressor",
    }
    assert "parents" not in req["provenance"]
    assert "derivation_op" not in req["provenance"]
