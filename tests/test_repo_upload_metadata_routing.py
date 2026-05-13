from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import gen_worker.presigned_upload as presigned_upload_module
from gen_worker.conversion.dispatch import _finalize_produced_variants
from gen_worker.conversion.produced import ProducedFlavor
from gen_worker.request_context import ConversionContext


def test_repo_cas_complete_payload_is_parts_only(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    ctx = ConversionContext(
        request_id="req-1",
        job_id="job-1",
        file_api_base_url="http://tensorhub",
        worker_capability_token="cap-token",
    )
    ctx._repo_job_upload_scope = lambda: ("owner", "repo", "job-1")  # type: ignore[method-assign]
    ctx._checkpoint_revision_id = lambda _owner, _repo: "revision-1"  # type: ignore[method-assign]

    def fake_presigned_upload_file(**kwargs: Any) -> presigned_upload_module.PresignedUploadResult:
        captured.update(kwargs)
        return presigned_upload_module.PresignedUploadResult(
            meta={
                "ref": "weights.safetensors",
                "size_bytes": kwargs["size_bytes"],
                "blake3": kwargs["blake3_hex"],
                "blob_digest": f"blake3:{kwargs['blake3_hex']}",
            }
        )

    monkeypatch.setattr(presigned_upload_module, "presigned_upload_file", fake_presigned_upload_file)

    stream = ctx.open_checkpoint_stream(
        "weights.safetensors",
        format="safetensors",
        flavor="bnb-nf4",
        attributes={"scheme": "nf4", "flavor": "bnb-nf4"},
    )
    stream.write(b"not really safetensors")
    out = stream.finalize()

    assert out.blob_digest
    assert captured["endpoint_path"] == "/api/v1/repos/owner/repo/revisions/revision-1/uploads"
    assert captured["create_payload"]["path"] == "weights.safetensors"
    assert captured["complete_extra"] is None


def test_conversion_dispatch_filters_checkpoint_metadata_and_writes_lineage_metadata(tmp_path: Path) -> None:
    out_dir = tmp_path / "tmplpt00vsg"
    out_dir.mkdir()
    (out_dir / "model.safetensors").write_bytes(b"weights")

    class FakeContext:
        request_id = "req-1"
        job_id = "job-1"
        destination = {"ref": "owner/repo", "tags": []}
        source = {"ref": "source/repo", "checkpoint_id": "source-checkpoint"}

        def __init__(self) -> None:
            self.saved: list[dict[str, Any]] = []
            self.published: dict[str, Any] | None = None

        def save_checkpoint(self, ref: str, local_path: str, **kwargs: Any) -> Any:
            self.saved.append({"ref": ref, "local_path": local_path, **kwargs})
            return SimpleNamespace(
                blob_digest="blake3:" + "a" * 64,
                size_bytes=Path(local_path).stat().st_size,
            )

        def publish_repo_revision(self, **kwargs: Any) -> dict[str, Any]:
            self.published = kwargs
            return {"ok": True}

    ctx = FakeContext()
    variant = ProducedFlavor(
        path=out_dir,
        flavor="nf4",
        attributes={"scheme": "nf4", "quantization_library": "bitsandbytes"},
    )

    _finalize_produced_variants(ctx, [variant], kind="quantization")  # type: ignore[arg-type]

    assert ctx.saved
    assert all("attributes" in item for item in ctx.saved)
    assert ctx.published is not None
    metadata = ctx.published["metadata"]
    flavor = metadata["checkpoint_flavors"][0]
    assert flavor["flavor"] == "nf4"
    assert "attributes" not in flavor
    assert flavor["metadata"]["scheme"] == "nf4"
    assert flavor["metadata"]["quantization_library"] == "bitsandbytes"
    assert "flavor" not in flavor["metadata"]
    assert "flavors" not in flavor["metadata"]
    assert "produced_by_job_id" not in flavor["metadata"]
    assert "produced_by_kind" not in flavor["metadata"]
    assert flavor["lineage_metadata"]["quantization_method"] == "nf4"
    assert flavor["lineage_metadata"]["quantization_library"] == "bitsandbytes"
