"""One identity authority: signed receipt -> TCG admission -> worker arm.

The former worker ``ExpectedIdentity`` projection compared a second spelling
of facts TCG already derives. These tests drive real signed receipts and real
TCG artifacts instead: the receipt binds the requested family and exact
identity before transport, delivered bytes bind to that receipt, and TCG alone
admits the compiled-graph key.
"""

# Pytest fixtures are imported into this module and then named as parameters.
# Ruff's ordinary shadowing rule cannot distinguish that injection idiom.
# ruff: noqa: F811

from __future__ import annotations

import inspect
import shutil
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from torch_compiled_graphs import Engine, GraphClassDeclaration
from torch_compiled_graphs.artifact import build_metadata, pack_artifact
from torch_compiled_graphs.host_isa import _host_requirement
from torch_compiled_graphs.identity import from_artifact_metadata

from gen_worker import (
    aot_delivery,
    aot_serve,
    artifact_meta,
    cell_resolve,
    compiled_graph_store,
    receipts,
)
from gen_worker.models import provision
from gen_worker.models.cache_paths import open_worker_cas
from harness import receipt_hub
from harness.receipt_hub import hub, rsa_key  # noqa: F401


def _elf() -> bytes:
    """A parseable ELF carrying the section TCG's package verifier inspects."""

    names = b"\0.shstrtab\0.lrodata\0"
    section_offset = 64
    section_size = 64
    string_offset = section_offset + section_size * 3
    image = bytearray(string_offset + len(names))
    image[:4] = b"\x7fELF"
    image[4:7] = bytes((2, 1, 1))
    struct.pack_into("<Q", image, 0x28, section_offset)
    struct.pack_into("<HHH", image, 0x3A, section_size, 3, 1)
    struct.pack_into("<II", image, section_offset + section_size, 1, 3)
    struct.pack_into(
        "<QQ",
        image,
        section_offset + section_size + 0x18,
        string_offset,
        len(names),
    )
    struct.pack_into("<II", image, section_offset + 2 * section_size, 11, 1)
    struct.pack_into(
        "<QQ",
        image,
        section_offset + 2 * section_size + 0x18,
        len(image),
        0,
    )
    image[string_offset:] = names
    return bytes(image)


def _artifact(tmp_path: Path, witness: str) -> tuple[Path, dict[str, Any]]:
    graph_class = f"denoiser/{witness[:8]}"
    package = tmp_path / f"{witness[:8]}.pt2"
    wrapper = "AOTInductorModelBase(1, 1, 0, device_str, std::move(cubin_dir), false)"
    with zipfile.ZipFile(package, "w") as archive:
        root = f"data/aotinductor/{graph_class}"
        archive.writestr(f"{root}/model.wrapper.cpp", wrapper)
        archive.writestr(f"{root}/model.so", _elf())
    graph = {
        "v": 3,
        "constant_fqns": [],
        "lifted_inputs": [],
        "pytree": {"in": "leaf", "out": "leaf"},
        "specialization": {},
    }
    declaration = GraphClassDeclaration(
        graph_class=graph_class,
        target="denoiser",
        graph=graph,
        graph_witness=witness,
        range_digest="0123456789abcdef" * 2,
    )
    metadata = build_metadata(
        graph_class={
            "name": declaration.graph_class,
            "target": declaration.target,
            "class_hash": declaration.class_hash,
            "graph": dict(declaration.graph),
            "graph_witness": declaration.graph_witness,
            "range_digest": declaration.range_digest,
            "fork": [],
            "class_dims": [],
            "strict": True,
            "lora_bucket": 0,
            "literal_values": "",
            "literal_payload_values": "",
            "placement": list(declaration.placement),
            "constants": [],
        },
        sm="sm_89",
        toolchain={"torch": "torch-content", "triton": "triton-content"},
        host_isa=_host_requirement().facts(),
    )
    path = pack_artifact(
        package,
        tmp_path / f"{witness[:8]}-compiled-graph.tar.gz",
        metadata,
    )
    return path, metadata


def _receipt(hub: Any, artifact: Path, metadata: dict[str, Any]) -> receipts.Receipt:
    jws = hub.serve_receipt_for(
        artifact,
        family=receipt_hub.FAMILY,
        publisher_tier=receipts.PUBLISHER_TIER_PLATFORM,
    )
    return receipts.verify_receipt(
        jws,
        family=receipt_hub.FAMILY,
        compiled_graph_key=str(metadata["compiled_graph_key"]),
        snapshot_digest=receipt_hub.SNAPSHOT,
        artifact_path=artifact.name,
        artifact_digest=receipt_hub.artifact_digest(artifact),
        artifact_size_bytes=artifact.stat().st_size,
    )


def _cache_artifact(root: Path, artifact: Path, digest: str) -> None:
    target = (
        root
        / compiled_graph_store.SIDECARS_DIRNAME
        / ".incoming"
        / f"{digest.partition(':')[2]}.tar.gz"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifact, target)


def test_worker_expected_identity_authority_is_deleted() -> None:
    root = Path(aot_serve.__file__).parent
    assert not (root / "aot_identity.py").exists()
    for module in (aot_serve, aot_delivery, cell_resolve, provision):
        source = inspect.getsource(module)
        assert "ExpectedIdentity" not in source
        assert "expected_identity_mismatch" not in source
    assert "expected" not in inspect.signature(aot_serve.stage_artifact).parameters
    assert "expected" not in inspect.signature(aot_serve.arm_entry).parameters
    assert "expected" not in inspect.signature(provision.arm_aot).parameters


def test_real_receipt_and_tcg_engine_are_the_complete_identity_path(
    tmp_path: Path,
    hub: Any,
) -> None:
    artifact, metadata = _artifact(tmp_path, "a" * 16)
    identity = from_artifact_metadata(artifact_meta.read_metadata(artifact))
    assert identity.value == metadata["compiled_graph_key"]
    verified = _receipt(hub, artifact, metadata)
    assert verified.identity_axes == identity.axes

    cache = tmp_path / "cas"
    _cache_artifact(cache, artifact, verified.artifact_digest)
    materialized = aot_delivery.materialize_named_artifact(
        identity.value,
        receipt_hub.FAMILY,
        f"root/family-{receipt_hub.FAMILY}#{identity.value}",
        verified.artifact_digest,
        None,
        receipt=verified,
        cache_dir=cache,
        what="identity integration proof",
    )

    assert materialized.is_file()
    resolved = Engine(open_worker_cas(cache)).resolve(
        identity.value,
        tmp_path / "resolved-after-restart",
    )
    assert resolved is not None
    assert resolved.key == identity.value
    assert from_artifact_metadata(resolved.metadata).axes == verified.identity_axes


def test_signed_identity_mismatch_refuses_before_tcg_import(
    tmp_path: Path,
    hub: Any,
) -> None:
    original, original_meta = _artifact(tmp_path, "b" * 16)
    changed, changed_meta = _artifact(tmp_path, "c" * 16)
    original_identity = from_artifact_metadata(original_meta)
    changed_identity = from_artifact_metadata(changed_meta)
    assert original_identity.value != changed_identity.value
    jws = hub.serve_receipt_for(
        changed,
        family=receipt_hub.FAMILY,
        compiled_graph_key=original_identity.value,
        identity_axes=original_identity.as_dict(),
        publisher_tier=receipts.PUBLISHER_TIER_PLATFORM,
    )
    verified = receipts.verify_receipt(
        jws,
        family=receipt_hub.FAMILY,
        compiled_graph_key=original_identity.value,
        snapshot_digest=receipt_hub.SNAPSHOT,
        artifact_path=changed.name,
        artifact_digest=receipt_hub.artifact_digest(changed),
        artifact_size_bytes=changed.stat().st_size,
    )
    cache = tmp_path / "cas"
    _cache_artifact(cache, changed, verified.artifact_digest)

    with pytest.raises(aot_delivery.NamedArtifactUnavailable) as refused:
        aot_delivery.materialize_named_artifact(
            original_identity.value,
            receipt_hub.FAMILY,
            f"root/family-{receipt_hub.FAMILY}#{original_identity.value}",
            verified.artifact_digest,
            None,
            receipt=verified,
            cache_dir=cache,
            what="tampered identity proof",
        )
    assert refused.value.reason == "artifact_receipt_refused"
    assert "receipt_identity_mismatch" in str(refused.value)
    assert Engine(open_worker_cas(cache)).resolve(
        original_identity.value,
        tmp_path / "must-not-resolve",
    ) is None


def test_unsigned_batch_family_cannot_replace_the_requested_family(
    tmp_path: Path,
    hub: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, metadata = _artifact(tmp_path, "d" * 16)
    key = str(metadata["compiled_graph_key"])
    digest = receipt_hub.artifact_digest(artifact)
    wrong_family = "another-family"
    jws = hub.serve_receipt_for(
        artifact,
        family=wrong_family,
        publisher_tier=receipts.PUBLISHER_TIER_PLATFORM,
    )
    answer = {
        "status": cell_resolve.STATUS_HIT,
        "found": True,
        "compiled_graph_key": key,
        "compiled_graph_ref": f"root/family-{wrong_family}#{key}",
        "content_digest": digest,
        "receipt": jws,
        "transport": {
            "snapshot_digest": receipt_hub.SNAPSHOT,
            "files": [{
                "path": artifact.name,
                "size_bytes": artifact.stat().st_size,
                "digest": digest,
                "url": "https://cas.invalid/compiled-graph",
            }],
        },
    }
    response = SimpleNamespace(
        status_code=200,
        text="",
        json=lambda: {
            "object": cell_resolve.RESOLVE_BATCH_OBJECT,
            "family": wrong_family,
            "answers": [answer],
        },
    )
    monkeypatch.setattr(cell_resolve.broker, "request", lambda *_a, **_k: response)

    (result,) = cell_resolve.resolve_batch(receipt_hub.FAMILY, [key])

    assert result.status == cell_resolve.STATUS_INCOMPLETE
    assert result.compiled_graph is None
    assert result.refusal_code == "compiled_graph_resolve_incomplete"
    assert "receipt_family_mismatch" in result.detail
