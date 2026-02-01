from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

from gen_worker.hf_downloader import HuggingFaceHubDownloader
from gen_worker.model_refs import HuggingFaceRef


def test_hf_downloader_errors_when_hub_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    d = HuggingFaceHubDownloader()
    with pytest.raises(RuntimeError) as e:
        d.download(HuggingFaceRef(repo_id="org/repo"))
    assert "huggingface_hub is required" in str(e.value)


def test_hf_downloader_builds_allow_patterns_and_skips_optional_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fake huggingface_hub module
    hub = types.ModuleType("huggingface_hub")

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_files(self, repo_id: str, repo_type: str, revision: str | None):
            assert repo_type == "model"
            return [
                "model_index.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.fp16.safetensors",
                "text_encoder/config.json",
                "text_encoder/model.fp16.safetensors",
                "tokenizer/vocab.json",
                "tokenizer/merges.txt",
                "scheduler/scheduler_config.json",
                # optional components present, but should be skipped by default
                "safety_checker/config.json",
                "safety_checker/model.fp16.safetensors",
                "feature_extractor/preprocessor_config.json",
            ]

    captured: dict[str, object] = {}

    def fake_hf_hub_download(*, repo_id: str, revision: str | None, filename: str, cache_dir=None, token=None):
        assert filename == "model_index.json"
        p = tmp_path / "model_index.json"
        p.write_text(
            json.dumps(
                {
                    "_class_name": "StableDiffusionPipeline",
                    "unet": ["diffusers", "UNet2DConditionModel"],
                    "vae": ["diffusers", "AutoencoderKL"],
                    "text_encoder": ["transformers", "CLIPTextModel"],
                    "tokenizer": ["transformers", "CLIPTokenizer"],
                    "scheduler": ["diffusers", "PNDMScheduler"],
                    "safety_checker": ["diffusers", "StableDiffusionSafetyChecker"],
                    "feature_extractor": ["transformers", "CLIPImageProcessor"],
                }
            ),
            encoding="utf-8",
        )
        return str(p)

    def fake_snapshot_download(*, repo_id: str, revision: str | None, **kwargs):
        captured["kwargs"] = kwargs
        return str(tmp_path / "snapshot")

    hub.HfApi = FakeApi
    hub.hf_hub_download = fake_hf_hub_download
    hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    monkeypatch.delenv("COZY_HF_FULL_REPO_DOWNLOAD", raising=False)
    monkeypatch.delenv("COZY_HF_COMPONENTS", raising=False)
    monkeypatch.delenv("COZY_HF_INCLUDE_OPTIONAL_COMPONENTS", raising=False)
    monkeypatch.delenv("COZY_HF_WEIGHT_PRECISIONS", raising=False)
    monkeypatch.delenv("COZY_HF_ALLOW_ROOT_JSON", raising=False)

    d = HuggingFaceHubDownloader(hf_home=str(tmp_path))
    out = d.download(HuggingFaceRef(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5"))
    assert out.local_dir.name == "snapshot"

    kwargs = captured["kwargs"]
    assert "allow_patterns" in kwargs
    allow = list(kwargs["allow_patterns"])  # type: ignore[index]

    # Required components should be present.
    assert "unet/config.json" in allow
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in allow
    assert "vae/diffusion_pytorch_model.fp16.safetensors" in allow
    assert "text_encoder/model.fp16.safetensors" in allow
    assert "tokenizer/vocab.json" in allow
    assert "tokenizer/merges.txt" in allow
    assert "scheduler/scheduler_config.json" in allow

    # Optional components should NOT be in the allowlist by default.
    assert not any(p.startswith("safety_checker/") for p in allow)
    assert not any(p.startswith("feature_extractor/") for p in allow)


def test_hf_downloader_errors_if_no_fp16_bf16_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = types.ModuleType("huggingface_hub")

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_files(self, repo_id: str, repo_type: str, revision: str | None):
            return [
                "model_index.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",  # fp32 only
            ]

    def fake_hf_hub_download(*, repo_id: str, revision: str | None, filename: str, cache_dir=None, token=None):
        p = tmp_path / "model_index.json"
        p.write_text(json.dumps({"_class_name": "X", "unet": ["diffusers", "UNet2DConditionModel"]}), "utf-8")
        return str(p)

    def fake_snapshot_download(*, repo_id: str, revision: str | None, **kwargs):
        return str(tmp_path / "snapshot")

    hub.HfApi = FakeApi
    hub.hf_hub_download = fake_hf_hub_download
    hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    monkeypatch.delenv("COZY_HF_FULL_REPO_DOWNLOAD", raising=False)
    monkeypatch.delenv("COZY_HF_COMPONENTS", raising=False)
    monkeypatch.delenv("COZY_HF_INCLUDE_OPTIONAL_COMPONENTS", raising=False)
    monkeypatch.setenv("COZY_HF_WEIGHT_PRECISIONS", "fp16,bf16")

    d = HuggingFaceHubDownloader(hf_home=str(tmp_path))
    # With hardcoded variant fallback order, we accept a default safetensors file
    # (no fp16/bf16 in the filename) as a last resort.
    out = d.download(HuggingFaceRef(repo_id="org/repo"))
    assert out.local_dir.name == "snapshot"


def test_hf_downloader_errors_if_no_safetensors_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = types.ModuleType("huggingface_hub")

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_files(self, repo_id: str, repo_type: str, revision: str | None):
            return [
                "model_index.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.bin",
            ]

    def fake_hf_hub_download(*, repo_id: str, revision: str | None, filename: str, cache_dir=None, token=None):
        p = tmp_path / "model_index.json"
        p.write_text(json.dumps({"_class_name": "X", "unet": ["diffusers", "UNet2DConditionModel"]}), "utf-8")
        return str(p)

    def fake_snapshot_download(*, repo_id: str, revision: str | None, **kwargs):
        return str(tmp_path / "snapshot")

    hub.HfApi = FakeApi
    hub.hf_hub_download = fake_hf_hub_download
    hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    monkeypatch.delenv("COZY_HF_FULL_REPO_DOWNLOAD", raising=False)
    monkeypatch.delenv("COZY_HF_COMPONENTS", raising=False)
    monkeypatch.delenv("COZY_HF_INCLUDE_OPTIONAL_COMPONENTS", raising=False)
    monkeypatch.delenv("COZY_HF_WEIGHT_PRECISIONS", raising=False)

    d = HuggingFaceHubDownloader(hf_home=str(tmp_path))
    with pytest.raises(RuntimeError) as e:
        d.download(HuggingFaceRef(repo_id="org/repo"))
    assert "missing required reduced-precision safetensors" in str(e.value)


def test_hf_downloader_includes_sharded_weights_from_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = types.ModuleType("huggingface_hub")

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_files(self, repo_id: str, repo_type: str, revision: str | None):
            assert repo_type == "model"
            return [
                "model_index.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.fp16.safetensors.index.json",
                "unet/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
                "unet/diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
            ]

    captured: dict[str, object] = {}

    def fake_hf_hub_download(*, repo_id: str, revision: str | None, filename: str, cache_dir=None, token=None):
        p = tmp_path / filename.replace("/", "__")
        if filename == "model_index.json":
            p.write_text(json.dumps({"_class_name": "X", "unet": ["diffusers", "UNet2DConditionModel"]}), "utf-8")
            return str(p)
        if filename == "unet/diffusion_pytorch_model.fp16.safetensors.index.json":
            p.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": "diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
                            "b": "diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
                        }
                    }
                ),
                "utf-8",
            )
            return str(p)
        raise AssertionError(f"unexpected hf_hub_download filename={filename}")

    def fake_snapshot_download(*, repo_id: str, revision: str | None, **kwargs):
        captured["kwargs"] = kwargs
        return str(tmp_path / "snapshot")

    hub.HfApi = FakeApi
    hub.hf_hub_download = fake_hf_hub_download
    hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    monkeypatch.delenv("COZY_HF_FULL_REPO_DOWNLOAD", raising=False)
    monkeypatch.delenv("COZY_HF_COMPONENTS", raising=False)
    monkeypatch.delenv("COZY_HF_INCLUDE_OPTIONAL_COMPONENTS", raising=False)
    monkeypatch.delenv("COZY_HF_WEIGHT_PRECISIONS", raising=False)
    monkeypatch.delenv("COZY_HF_VARIANT", raising=False)

    d = HuggingFaceHubDownloader(hf_home=str(tmp_path))
    _ = d.download(HuggingFaceRef(repo_id="org/repo"))

    allow = list(captured["kwargs"]["allow_patterns"])  # type: ignore[index]
    assert "unet/diffusion_pytorch_model.fp16.safetensors.index.json" in allow
    assert "unet/diffusion_pytorch_model.fp16-00001-of-00002.safetensors" in allow
    assert "unet/diffusion_pytorch_model.fp16-00002-of-00002.safetensors" in allow


def test_hf_downloader_errors_if_selection_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = types.ModuleType("huggingface_hub")

    class FakeRepoFile:
        def __init__(self, path: str, size: int):
            self.path = path
            self.size = size

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_files(self, repo_id: str, repo_type: str, revision: str | None):
            assert repo_type == "model"
            return [
                "model_index.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.fp16.safetensors",
            ]

        def list_repo_tree(self, repo_id: str, repo_type: str, revision: str | None, recursive: bool):
            assert recursive is True
            return [
                FakeRepoFile("model_index.json", 100),
                FakeRepoFile("unet/config.json", 100),
                # Make the selection exceed the 30GB hard limit.
                FakeRepoFile("unet/diffusion_pytorch_model.fp16.safetensors", 31_000_000_000),
            ]

    def fake_hf_hub_download(*, repo_id: str, revision: str | None, filename: str, cache_dir=None, token=None):
        p = tmp_path / "model_index.json"
        p.write_text(json.dumps({"_class_name": "X", "unet": ["diffusers", "UNet2DConditionModel"]}), "utf-8")
        return str(p)

    def fake_snapshot_download(*, repo_id: str, revision: str | None, **kwargs):
        raise AssertionError("snapshot_download should not be called when safety limit triggers")

    hub.HfApi = FakeApi
    hub.hf_hub_download = fake_hf_hub_download
    hub.snapshot_download = fake_snapshot_download
    hub.hf_hub_url = lambda **kwargs: "https://example.invalid/file"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    monkeypatch.delenv("COZY_HF_FULL_REPO_DOWNLOAD", raising=False)
    monkeypatch.delenv("COZY_HF_COMPONENTS", raising=False)
    monkeypatch.delenv("COZY_HF_INCLUDE_OPTIONAL_COMPONENTS", raising=False)
    monkeypatch.delenv("COZY_HF_WEIGHT_PRECISIONS", raising=False)
    monkeypatch.delenv("COZY_HF_ALLOW_ROOT_JSON", raising=False)

    d = HuggingFaceHubDownloader(hf_home=str(tmp_path))
    with pytest.raises(RuntimeError) as e:
        d.download(HuggingFaceRef(repo_id="org/repo"))
    assert "excessively large" in str(e.value)
