"""Family normalization + dtype threading for the singlefile<->diffusers
repackage path (e2e tracker #112: launch-catalog ingest)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from cozy_convert.base_model_families import civitai_to_family
from cozy_convert.clone import _REPACKAGE_NORMALIZED_FAMILIES
from cozy_convert.ingest import _detect_snapshot_dtype
from cozy_convert.layout import canonical_model_family_from_variant
from cozy_convert.repackage import (
    _normalize_family,
    _repackage_torch_dtype,
    _singlefile_attempts_for_family,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sdxl", "sdxl"),
        ("sdxl-illustrious", "sdxl"),
        ("sdxl-pony", "sdxl"),
        ("sdxl-lightning", "sdxl"),
        ("sd15", "sd15_sd2"),
        ("sd14", "sd15_sd2"),
        ("sd2", "sd15_sd2"),
        ("flux1-dev", "flux"),
        ("flux1-schnell", "flux"),
        ("flux.2-klein-9b", "flux"),
        ("flux2-dev", "flux"),
        ("z-image", "zimage"),
        ("z-image-turbo", "zimage"),
        ("qwen-image", "unknown"),
        ("wan22", "wan"),
        ("ernie", "unknown"),
        ("anima", "unknown"),
        ("sd3-medium", "unknown"),
        ("", "unknown"),
    ],
)
def test_normalize_family(raw: str, expected: str) -> None:
    assert _normalize_family(raw) == expected


def test_repackage_families_all_have_singlefile_attempts() -> None:
    for family in _REPACKAGE_NORMALIZED_FAMILIES:
        assert _singlefile_attempts_for_family(family), family


def test_zimage_attempts_use_zimage_pipeline() -> None:
    attempts = _singlefile_attempts_for_family("zimage")
    assert attempts[0][0] == "ZImagePipeline"
    assert ("ZImagePipeline", "Tongyi-MAI/Z-Image-Turbo") in attempts


def test_repackage_torch_dtype() -> None:
    import torch

    assert _repackage_torch_dtype("fp16") is torch.float16
    assert _repackage_torch_dtype("fp32") is torch.float32
    assert _repackage_torch_dtype("bf16") is torch.bfloat16
    assert _repackage_torch_dtype("int4:nf4") is torch.bfloat16
    assert _repackage_torch_dtype(None) is torch.bfloat16


def test_variant_to_family_zimage_and_qwen_are_not_flux() -> None:
    assert canonical_model_family_from_variant("z_image") == "z-image"
    assert canonical_model_family_from_variant("qwen_image") == "qwen-image"
    assert canonical_model_family_from_variant("flux1") == "flux"


@pytest.mark.parametrize(
    ("base", "family"),
    [
        ("ZImageTurbo", "z-image-turbo"),
        ("ZImageBase", "z-image"),
        ("Ernie", "ernie"),
        ("Qwen", "qwen-image"),
        ("Anima", "anima"),
        ("Flux.2 Klein 9B", "flux.2-klein-9b"),
        ("Flux.2 Klein 9B-base", "flux.2-klein-9b"),
        ("Wan Video 2.2 I2V-A14B", "wan22"),
        ("NoobAI", "sdxl-illustrious"),
        ("Illustrious", "sdxl-illustrious"),
        ("Pony", "sdxl-pony"),
    ],
)
def test_civitai_to_family_launch_bases(base: str, family: str) -> None:
    assert civitai_to_family(base) == family


def _write_safetensors(path: Path, dtype: str, n: int = 4) -> None:
    import torch
    from safetensors.torch import save_file

    torch_dtype = {
        "F16": torch.float16, "BF16": torch.bfloat16,
        "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
    }[dtype]
    tensors = {
        f"w{i}": torch.zeros(4, dtype=torch.float32).to(torch_dtype) for i in range(n)
    }
    save_file(tensors, str(path))


def test_detect_snapshot_dtype(tmp_path: Path) -> None:
    _write_safetensors(tmp_path / "model.safetensors", "F16")
    assert _detect_snapshot_dtype(tmp_path) == "fp16"

    fp8 = tmp_path / "fp8"
    fp8.mkdir()
    _write_safetensors(fp8 / "dit.safetensors", "F8_E4M3", n=8)
    _write_safetensors(fp8 / "vae.safetensors", "BF16", n=2)
    assert _detect_snapshot_dtype(fp8) == "fp8"

    empty = tmp_path / "empty"
    empty.mkdir()
    assert _detect_snapshot_dtype(empty) == ""


def test_weight_groups_singlefile_multi_entry(tmp_path: Path) -> None:
    """Civitai bundles ship several root weight files (DiT + text encoder +
    VAE); every one must be its own group or a dtype pass drops all but the
    first (latent data loss, found preparing e2e #112)."""
    from cozy_convert.clone import _weight_groups

    _write_safetensors(tmp_path / "dit.safetensors", "F8_E4M3")
    _write_safetensors(tmp_path / "txt.safetensors", "BF16")
    _write_safetensors(tmp_path / "vae.safetensors", "BF16")
    groups = _weight_groups(tmp_path, "singlefile")
    assert [g[1].name for g in groups] == [
        "dit.safetensors", "txt.safetensors", "vae.safetensors"]
    assert all(comp == "" for comp, _ in groups)


def test_weight_groups_singlefile_sharded_index_wins(tmp_path: Path) -> None:
    from cozy_convert.clone import _weight_groups

    _write_safetensors(tmp_path / "model-00001-of-00002.safetensors", "BF16")
    _write_safetensors(tmp_path / "model-00002-of-00002.safetensors", "BF16")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }))
    groups = _weight_groups(tmp_path, "singlefile")
    assert [g[1].name for g in groups] == ["model.safetensors.index.json"]


def test_build_flavor_tree_source_dtype_passthrough(tmp_path: Path) -> None:
    """dtype='source' publishes every source file untouched and records the
    detected on-disk dtype (mixed-dtype bundles keep their majority label)."""
    from cozy_convert.clone import OutputSpec, build_flavor_tree
    from cozy_convert.ingest import IngestedSource

    src = tmp_path / "src"
    src.mkdir()
    _write_safetensors(src / "dit.safetensors", "F8_E4M3", n=8)
    _write_safetensors(src / "txt.safetensors", "BF16", n=2)
    (src / "workflow.json").write_text("{}")

    source = IngestedSource(
        provider="civitai", source_ref="1", source_revision="x", dir=src,
        layout="singlefile", model_family="ernie", model_family_variant="unknown",
        attrs={"dtype": _detect_snapshot_dtype(src), "file_layout": "singlefile"},
    )
    tree, attrs = build_flavor_tree(
        source,
        OutputSpec(dtype="source", file_layout="singlefile", file_type="safetensors"),
        tmp_path / "flavor",
    )
    assert attrs["dtype"] == "fp8"
    got = sorted(p.name for p in tree.rglob("*") if p.is_file())
    assert got == ["dit.safetensors", "txt.safetensors", "workflow.json"]
    assert (tree / "dit.safetensors").read_bytes() == (src / "dit.safetensors").read_bytes()


def test_build_flavor_tree_source_dtype_refuses_repackage(tmp_path: Path) -> None:
    from cozy_convert.clone import OutputSpec, build_flavor_tree
    from cozy_convert.ingest import IngestedSource

    src = tmp_path / "src"
    src.mkdir()
    _write_safetensors(src / "model.safetensors", "BF16")
    source = IngestedSource(
        provider="civitai", source_ref="1", source_revision="x", dir=src,
        layout="singlefile", model_family="sdxl", model_family_variant="unknown",
        attrs={"dtype": "bf16", "file_layout": "singlefile"},
    )
    with pytest.raises(ValueError, match="source"):
        build_flavor_tree(
            source,
            OutputSpec(dtype="source", file_layout="diffusers", file_type="safetensors"),
            tmp_path / "flavor",
        )


def test_detect_snapshot_dtype_ignores_garbage(tmp_path: Path) -> None:
    # A non-safetensors payload with a plausible header length must not crash.
    bogus = tmp_path / "bogus.safetensors"
    header = json.dumps({"not_a_tensor": "x"}).encode()
    bogus.write_bytes(struct.pack("<Q", len(header)) + header)
    assert _detect_snapshot_dtype(tmp_path) == ""


def test_civitai_base_model_beats_filename_tokens() -> None:
    """GonzaLomo regression (e2e #112): an SDXL 1.0 checkpoint whose FILENAME
    contains 'flux' must classify as sdxl via the structured baseModel."""
    from cozy_convert.layout import (
        detect_huggingface_source_layout,
        infer_model_family_variant_from_hint,
    )

    # The filename alone still reads as flux (that is the hazard)…
    assert infer_model_family_variant_from_hint(
        "gonzalomoXLFluxPony_v70PhotoXLDMD.safetensors") in ("flux1", "flux2")
    # the ingest-level precedence keeps baseModel authoritative.
    from cozy_convert.ingest import _resolve_civitai_family

    assert _resolve_civitai_family("sdxl", "flux") == "sdxl"
    assert _resolve_civitai_family("", "flux") == "flux"
    assert _resolve_civitai_family("other", "flux") == "flux"
    assert _resolve_civitai_family("other", "unknown") == "other"
    assert _resolve_civitai_family("", "unknown") == ""


def test_missing_component_error_parse() -> None:
    """moody-pro-mix-zit regression (e2e #112): DiT-only single-file checkpoints
    must source absent components from the family config repo. The component
    name+class ride in diffusers' SingleFileComponentError remedy snippet."""
    from cozy_convert.repackage import _MISSING_COMPONENT_RE

    msg = (
        "Failed to load Qwen3Model. Weights for this component appear to be "
        "missing in the checkpoint.\n"
        "Please load the component before passing it in as an argument to "
        "`from_single_file`.\n\n"
        "text_encoder = Qwen3Model.from_pretrained('...')\n"
        "pipe = ZImagePipeline.from_single_file(<checkpoint path>, "
        "text_encoder=text_encoder)\n\n"
    )
    m = _MISSING_COMPONENT_RE.search(msg)
    assert m is not None
    assert (m.group(1), m.group(2)) == ("text_encoder", "Qwen3Model")


def test_multi_weight_bundle_detection(tmp_path) -> None:
    """Anima regression (e2e #112): multi-component civitai bundles (distinct
    DiT/TE/VAE single-files) must publish with library_name unset — tensorhub
    finalize rejects them as diffusers/single-file
    (multiple_files_for_single_file_layout)."""
    from cozy_convert.ingest import _is_multi_weight_bundle

    d = tmp_path
    (d / "anima_dit.safetensors").write_bytes(b"x")
    assert not _is_multi_weight_bundle(d)
    # one logical model resharded HF-style is NOT a bundle
    (d / "anima_dit.safetensors").unlink()
    (d / "big-00001-of-00002.safetensors").write_bytes(b"x")
    (d / "big-00002-of-00002.safetensors").write_bytes(b"x")
    assert not _is_multi_weight_bundle(d)
    # a second distinct component IS a bundle
    (d / "qwen_image_vae.safetensors").write_bytes(b"x")
    assert _is_multi_weight_bundle(d)
