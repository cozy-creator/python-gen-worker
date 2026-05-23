from __future__ import annotations

from pathlib import Path

from gen_worker.conversion import ingest
from gen_worker.conversion.ingest import (
    CivitaiImportCandidate,
    CivitaiResolvedIdentity,
)


def _resolved_identity() -> CivitaiResolvedIdentity:
    candidate = CivitaiImportCandidate(
        file_id=99,
        name="model.safetensors",
        rel_path="model.safetensors",
        download_url="https://civitai.com/api/download/models/99",
        size_bytes=5,
        size_bytes_exact=True,
        primary=True,
        hashes={},
        fingerprint="fp-99",
    )
    return CivitaiResolvedIdentity(
        source_ref="civitai:model_version_id=123",
        source_revision="rev",
        source_manifest_sha256="sha",
        model_version_id=123,
        model_id=1,
        base_model="SDXL",
        base_model_type="Checkpoint",
        air="",
        model_family="sdxl",
        model_family_variant="unknown",
        pipeline_hint="diffusers-single-file",
        selected_file_id=99,
        selected_files=[candidate],
        file_fingerprints={"99": "fp-99"},
    )


def test_civitai_download_passes_api_key_to_cas(monkeypatch, tmp_path: Path) -> None:
    """The per-invocation Civitai API key is threaded to source_url_to_cas as
    the Bearer auth token, scoped to Civitai's own hosts (not the CDN redirect).
    """
    seen: dict[str, object] = {}

    def fake_source_url_to_cas(source_url, output_path, **kwargs):
        seen["source_auth_token"] = kwargs.get("source_auth_token")
        seen["source_auth_hosts"] = kwargs.get("source_auth_hosts")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"12345")
        return {"output_path": str(output_path), "sha256": "", "size_bytes": 5}

    monkeypatch.setattr(ingest, "source_url_to_cas", fake_source_url_to_cas)

    ingest.download_civitai_model_version_files(
        123,
        tmp_path / "out",
        resolved_identity=_resolved_identity(),
        civitai_api_key="invoker-key",
    )

    assert seen["source_auth_token"] == "invoker-key"
    assert seen["source_auth_hosts"] == ingest._CIVITAI_SOURCE_AUTH_HOSTS


def test_civitai_download_unauthenticated_by_default(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_source_url_to_cas(source_url, output_path, **kwargs):
        seen["source_auth_token"] = kwargs.get("source_auth_token")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"12345")
        return {"output_path": str(output_path), "sha256": "", "size_bytes": 5}

    monkeypatch.setattr(ingest, "source_url_to_cas", fake_source_url_to_cas)

    ingest.download_civitai_model_version_files(
        123,
        tmp_path / "out",
        resolved_identity=_resolved_identity(),
    )

    assert seen["source_auth_token"] == ""
