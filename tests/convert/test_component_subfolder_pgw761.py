"""A standalone diffusers component published in a SUBFOLDER."""

from __future__ import annotations

import fnmatch
import json
import shutil
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gen_worker.convert.classifier import (
    RepoRefusal,
    apply_source_include,
    classify_repo,
)
from gen_worker.convert.ingest import ingest_huggingface, plan_huggingface

_LISTING = json.loads(
    (Path(__file__).resolve().parents[1] / "testdata" / "prunavaed_hf_listing.json")
    .read_text(encoding="utf-8")
)
_PATHS = [f["path"] for f in _LISTING["files"]]
_SIZES = {f["path"]: int(f["size"]) for f in _LISTING["files"]}
_VAE_CONFIG = _LISTING["vae_config"]
_COMPONENT_CONFIGS = {"vae": _VAE_CONFIG}


def test_real_prunavaed_listing_classifies_as_a_subfolder_component() -> None:
    c = classify_repo(_PATHS, sizes=_SIZES, component_configs=_COMPONENT_CONFIGS)

    assert c.strategy == "diffusers_component"
    assert c.runtime_library == "diffusers"
    assert c.component_subfolder == "vae"
    assert c.attrs["architecture"] == "AutoencoderKLLTX2Video"
    assert set(c.allow_patterns) == {
        "README.md",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    }


def test_ambiguous_component_subfolders_still_refuse() -> None:
    """Two component subfolders and no root marker is a shape we do not guess at — fail closed with the same typed refusal as before."""
    paths = [
        "README.md",
        "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
        "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors",
    ]
    with pytest.raises(RepoRefusal) as excinfo:
        classify_repo(paths, component_configs={
            "vae": {"_class_name": "AutoencoderKLLTX2Video"},
            "transformer": {"_class_name": "LTX2VideoTransformer3DModel"},
        })
    assert excinfo.value.reason == "unknown_shape"

    c = classify_repo(
        apply_source_include(paths, ["vae/*"]),
        component_configs={"vae": {"_class_name": "AutoencoderKLLTX2Video"}},
    )
    assert (c.strategy, c.component_subfolder) == ("diffusers_component", "vae")


def test_ingest_leaves_a_root_component_repo_alone(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The root-config shape publishes exactly as it did before: diffusers library declared, single-file layout, nothing opted out."""
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "config.json").write_text(
        json.dumps({"_class_name": "AutoencoderKL"}), encoding="utf-8")
    (remote / "diffusion_pytorch_model.safetensors").write_bytes(_safetensors_bytes())
    monkeypatch.setattr("gen_worker.convert.ingest.hf", lambda: _fake_hf(remote))

    source = ingest_huggingface("madebyollin/sdxl-vae-fp16-fix", tmp_path / "snapshot")

    assert source.classification is not None
    assert source.classification.component_subfolder == ""
    assert source.repo_spec["library_name"] == "diffusers"
    assert source.repo_spec["class_name"] == "AutoencoderKL"
    assert source.attrs["file_layout"] == "single-file"
    assert "component_subfolder" not in source.metadata
    assert sorted(p.name for p in source.dir.rglob("*") if p.is_file()) == [
        "config.json", "diffusion_pytorch_model.safetensors"]


def _safetensors_bytes() -> bytes:
    header = json.dumps(
        {"decoder.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
    ).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + b"\x00\x02"


def _remote_repo(tmp_path: Path) -> Path:
    remote = tmp_path / "remote"
    (remote / "vae").mkdir(parents=True)
    (remote / "example").mkdir(parents=True)
    (remote / "README.md").write_text("# PrunaVAED\n", encoding="utf-8")
    (remote / "patch_diffusers.py").write_text("# demo\n", encoding="utf-8")
    (remote / "example" / "prompts.json").write_text("{}", encoding="utf-8")
    (remote / "vae" / "config.json").write_text(json.dumps(_VAE_CONFIG), encoding="utf-8")
    (remote / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(
        _safetensors_bytes())
    return remote


def _fake_hf(remote: Path) -> Any:

    def _files() -> list[Path]:
        return sorted(p for p in remote.rglob("*") if p.is_file())

    class _Api:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def repo_info(self, repo_id: str, revision: str | None = None) -> Any:
            return SimpleNamespace(sha="deadbeef" * 5)

        def list_repo_tree(self, repo_id: str, revision: str | None = None,
                           recursive: bool = False) -> Any:
            for p in _files():
                rel = p.relative_to(remote).as_posix()
                yield SimpleNamespace(
                    path=rel, size=p.stat().st_size,
                    lfs=SimpleNamespace(sha256=f"{abs(hash(rel)):064x}"[:64]),
                    blob_id="")

    def _download(repo_id: str, filename: str, revision: str | None = None,
                  token: str | None = None) -> str:
        return str(remote / filename)

    def _snapshot(repo_id: str, revision: str | None = None, local_dir: str = "",
                  allow_patterns: list[str] | None = None,
                  token: str | None = None) -> str:
        for p in _files():
            rel = p.relative_to(remote).as_posix()
            if allow_patterns and not any(
                    fnmatch.fnmatch(rel, pat) for pat in allow_patterns):
                continue
            dest = Path(local_dir) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        return str(local_dir)

    return SimpleNamespace(
        HfApi=_Api, hf_hub_download=_download, snapshot_download=_snapshot)


def test_ingest_mirrors_the_publisher_layout_unchanged(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    remote = _remote_repo(tmp_path)
    monkeypatch.setattr("gen_worker.convert.ingest.hf", lambda: _fake_hf(remote))

    plan = plan_huggingface("PrunaAI/PrunaVAED", source_include=["vae/*"])
    assert plan.classification.component_subfolder == "vae"
    assert [p for p, _, _ in plan.bank_files()] == [
        "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]

    source = ingest_huggingface(
        "PrunaAI/PrunaVAED", tmp_path / "snapshot", plan=plan)

    on_disk = sorted(p.relative_to(source.dir).as_posix()
                     for p in source.dir.rglob("*") if p.is_file())
    assert on_disk == ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
    assert (source.dir / "vae").is_dir()
    assert json.loads((source.dir / "vae" / "config.json").read_text())["_class_name"] \
        == "AutoencoderKLLTX2Video"
    assert source.repo_spec["class_name"] == "AutoencoderKLLTX2Video"
    assert source.repo_spec["library_name"] == ""
    assert source.repo_spec["kind"] == "model"
    assert source.attrs["dtype"] == "bf16"
    assert source.metadata["component_subfolder"] == "vae"
