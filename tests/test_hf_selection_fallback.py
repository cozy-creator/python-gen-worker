"""Selection-policy regression tests for ``hf_selection`` (issue: HF
downloader selected ZERO weight files for sd1.5).

The planner must NEVER return an empty weight set for a repo that has
weights, must prefer 16-bit (bf16/fp16) over fp32, and must fall back
gracefully when dtype probing is unavailable.
"""

from __future__ import annotations

from typing import Optional, Set

import pytest

from gen_worker.models.hf_selection import (
    HFSelectionPolicy,
    plan_diffusers_download,
)


_MODEL_INDEX = {
    "_class_name": "StableDiffusionPipeline",
    "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}

# Default product policy: prefer 16-bit, no fp32.
_DEFAULT_POLICY = HFSelectionPolicy(weight_precisions=("bf16", "fp16"))
# A "bf16 flavor" should behave the same as 16-bit family default — bf16 OR
# fp16, whichever the repo has.
_BF16_FLAVOR_POLICY = HFSelectionPolicy(weight_precisions=("bf16", "fp16"))


def _unet_weights(plan) -> list[str]:
    return sorted(
        f for f in plan.selected_files if f.startswith("unet/") and f.endswith((".safetensors", ".bin", ".ckpt"))
    )


def _fp16_only_repo() -> list[str]:
    return [
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ]


# --------------------------------------------------------------------------- #
# (a) fp16-only repo under bf16 flavor AND under default both pick fp16 weights
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("policy", [_DEFAULT_POLICY, _BF16_FLAVOR_POLICY])
@pytest.mark.parametrize(
    "probe",
    [
        None,  # no probe callable
        lambda rel: None,  # probe returns unknown
        lambda rel: {"F16"} if rel.endswith(".safetensors") else None,  # real probe
    ],
)
def test_fp16_only_repo_selects_fp16_weights(policy, probe) -> None:
    plan = plan_diffusers_download(
        model_index=_MODEL_INDEX,
        repo_files=_fp16_only_repo(),
        policy=policy,
        probe_safetensors_dtypes=probe,
    )
    selected = _unet_weights(plan)
    assert selected == ["unet/diffusion_pytorch_model.fp16.safetensors"], selected
    # And it is genuinely non-empty.
    assert selected


# --------------------------------------------------------------------------- #
# (b) fp32 is NOT selected when a 16-bit weight exists
# --------------------------------------------------------------------------- #


def test_fp32_not_selected_when_16bit_exists() -> None:
    # sd1.5-shaped unet: plain (fp32) safetensors + fp16 variant + .bin files.
    repo = [
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.bin",
        "unet/diffusion_pytorch_model.fp16.bin",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.non_ema.safetensors",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ]

    def probe(rel: str) -> Optional[Set[str]]:
        if not rel.endswith(".safetensors"):
            return None
        return {"F16"} if "fp16" in rel else {"F32"}

    plan = plan_diffusers_download(
        model_index=_MODEL_INDEX,
        repo_files=repo,
        policy=_DEFAULT_POLICY,
        probe_safetensors_dtypes=probe,
    )
    assert _unet_weights(plan) == ["unet/diffusion_pytorch_model.fp16.safetensors"]
    # The fp32 master must NOT be present.
    assert "unet/diffusion_pytorch_model.safetensors" not in plan.selected_files

    # Same outcome even when probing is unavailable (filename fallback).
    plan2 = plan_diffusers_download(
        model_index=_MODEL_INDEX,
        repo_files=repo,
        policy=_DEFAULT_POLICY,
        probe_safetensors_dtypes=None,
    )
    assert _unet_weights(plan2) == ["unet/diffusion_pytorch_model.fp16.safetensors"]


# --------------------------------------------------------------------------- #
# (c) the planner never returns an empty weight set for a repo that has weights
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "repo,probe,expect_substr",
    [
        # fp32-only safetensors, default (no-fp32) policy: must still select it.
        (
            [
                "model_index.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",
            ],
            lambda rel: {"F32"} if rel.endswith(".safetensors") else None,
            "unet/diffusion_pytorch_model.safetensors",
        ),
        # fp32-only, no probe at all.
        (
            [
                "model_index.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",
            ],
            None,
            "unet/diffusion_pytorch_model.safetensors",
        ),
        # .bin-only component (no safetensors): last-resort .bin fallback.
        (
            [
                "model_index.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.bin",
                "unet/diffusion_pytorch_model.fp16.bin",
            ],
            lambda rel: None,
            "unet/diffusion_pytorch_model.fp16.bin",
        ),
    ],
)
def test_never_empty_weight_set(repo, probe, expect_substr) -> None:
    plan = plan_diffusers_download(
        model_index={"_class_name": "X", "unet": ["diffusers", "UNet2DConditionModel"]},
        repo_files=repo,
        policy=_DEFAULT_POLICY,
        probe_safetensors_dtypes=probe,
    )
    weights = _unet_weights(plan)
    assert weights, f"planner returned ZERO unet weights for {repo}"
    assert expect_substr in plan.selected_files


# --------------------------------------------------------------------------- #
# Optional components (safety_checker / feature_extractor) must come down so
# the snapshot is loadable by diffusers. Dropping them left empty folders,
# which made `from_pretrained` raise on the missing preprocessor_config.json
# and cascade into the misleading "no file named ...bin" error (the sd1.5
# failure this whole change set fixes).
# --------------------------------------------------------------------------- #


def test_optional_components_kept_when_typed_and_pruned_when_null() -> None:
    """Typed optional components (config-only feature_extractor + weighted
    safety_checker) come down so diffusers can load — fp16 weight, not fp32; a
    NULL-typed component is pruned. Dropping config-only dirs left the sd1.5
    failure ("no file named ...bin")."""
    def probe(rel: str):
        if not rel.endswith(".safetensors"):
            return None
        return {"F16"} if "fp16" in rel else {"F32"}

    # (a) typed optional components are kept (fp16, not fp32).
    plan = plan_diffusers_download(
        model_index={
            "_class_name": "StableDiffusionPipeline",
            "feature_extractor": ["transformers", "CLIPImageProcessor"],
            "safety_checker": ["stable_diffusion", "StableDiffusionSafetyChecker"],
            "unet": ["diffusers", "UNet2DConditionModel"],
        },
        repo_files=[
            "model_index.json",
            "feature_extractor/preprocessor_config.json",
            "safety_checker/config.json",
            "safety_checker/model.fp16.safetensors",
            "safety_checker/model.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
        ],
        policy=_DEFAULT_POLICY,
        probe_safetensors_dtypes=probe,
    )
    assert "feature_extractor/preprocessor_config.json" in plan.selected_files
    assert "safety_checker/config.json" in plan.selected_files
    assert "safety_checker/model.fp16.safetensors" in plan.selected_files
    assert "safety_checker/model.safetensors" not in plan.selected_files

    # (b) a NULL-typed component is pruned entirely.
    plan2 = plan_diffusers_download(
        model_index={
            "_class_name": "StableDiffusionPipeline",
            "safety_checker": [None, None],
            "unet": ["diffusers", "UNet2DConditionModel"],
        },
        repo_files=[
            "model_index.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ],
        policy=_DEFAULT_POLICY,
        probe_safetensors_dtypes=lambda r: {"F16"} if r.endswith(".safetensors") else None,
    )
    assert not any(f.startswith("safety_checker/") for f in plan2.selected_files)
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in plan2.selected_files
