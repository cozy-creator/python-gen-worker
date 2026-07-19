"""gw#592: LTX-2.3 family detection.

Lightricks/LTX-2.3 has no model_index.json / config.json — it publishes
either one monolithic ``ltx-2*.safetensors`` at root, or the DiffSynth-Studio
LTX-2.3-Repackage per-submodule layout (transformer.safetensors,
video_vae_encoder.safetensors, ...). Neither carries a diffusers pipeline
manifest, so ``detect_huggingface_source_layout`` fell through to
model_family="unknown" and the clone's singlefile->diffusers repackage step
refused it. These tests are synthetic file-listing fixtures only — no real
LTX-2 weights are downloaded (this box never fetches real weights locally).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.convert.layout import (
    canonical_model_family_from_variant,
    detect_huggingface_source_layout,
    infer_model_family_variant_from_hint,
)


@pytest.mark.parametrize(
    "filename",
    [
        "ltx-2.3-13b-dev.safetensors",
        "ltx-2.3-i2v-13b-dev.safetensors",
        "LTX-2.3-distilled.safetensors",
        "ltx2-13b.safetensors",
    ],
)
def test_hint_detects_ltx2_from_monolith_filename(filename: str) -> None:
    assert infer_model_family_variant_from_hint(filename) == "ltx2"


def test_hint_does_not_confuse_ltx_video_v1_with_ltx2() -> None:
    # LTX-Video (v1) is a distinct, already-diffusers-native family — must
    # never be misdetected as ltx2 by a bare "ltx" substring match.
    assert infer_model_family_variant_from_hint("ltx-video-2b-v0.9.safetensors") == "unknown"


def test_canonical_family_from_ltx2_variant() -> None:
    assert canonical_model_family_from_variant("ltx2") == "ltx2"


def test_detect_layout_stamps_ltx2_for_monolith_singlefile(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    files = ["ltx-2.3-13b-dev.safetensors", "README.md"]
    for f in files:
        (repo_dir / f).write_bytes(b"\x00")

    info = detect_huggingface_source_layout(repo_dir=repo_dir, files=files)

    assert info.source_layout == "singlefile"
    assert info.model_family == "ltx2"
    assert info.model_family_variant == "ltx2"


def test_detect_layout_stamps_ltx2_for_repackage_layout(tmp_path: Path) -> None:
    """DiffSynth-Studio/LTX-2.3-Repackage: per-submodule ROOT files, none of
    which carry an "ltx" token individually — must be detected from repo
    structure (mirrors the te#70 trainer's own sentinel check). Root-only:
    classify_repo's current "multiple root safetensors" selection only scans
    top-level paths, so a real ingested repackage snapshot never carries the
    text_encoder/ subdir at this point either (a separate, out-of-scope gap)."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    files = [
        "transformer.safetensors",
        "video_vae_encoder.safetensors",
        "video_vae_decoder.safetensors",
        "audio_vae_decoder.safetensors",
        "audio_vocoder.safetensors",
        "text_encoder_post_modules.safetensors",
    ]
    for f in files:
        (repo_dir / f).write_bytes(b"\x00")

    info = detect_huggingface_source_layout(repo_dir=repo_dir, files=files)

    assert info.source_layout == "singlefile"
    assert info.model_family == "ltx2"


def test_detect_layout_repackage_requires_both_sentinel_files(tmp_path: Path) -> None:
    """A repo with only ONE of the two sentinel files (e.g. a lone VAE
    component repo) must not be misdetected as a full LTX-2 snapshot."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    files = ["video_vae_encoder.safetensors", "config.json"]
    for f in files:
        (repo_dir / f).write_bytes(b"\x00")

    info = detect_huggingface_source_layout(repo_dir=repo_dir, files=files)

    assert info.model_family != "ltx2"
