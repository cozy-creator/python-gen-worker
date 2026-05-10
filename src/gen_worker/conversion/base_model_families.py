"""Base-model family enum + cross-format mapping tables.

Records the destination LoRA / PEFT checkpoint's lineage on two axes:
  - `base_model_family`: a finite canonical enum (`sdxl`, `flux2-klein-4b`,
    `llama-3-8b`, ...) — useful for compatibility checks ("base+LoRA must
    share family") and discovery filters
  - `base_model_repo`: the specific upstream repo id (`stabilityai/...`)
    — useful for deterministic load and "best base" suggestions

Resolution priority depends on the source format:
  - PEFT canonical: `adapter_config.json["base_model_name_or_path"]` is
    the strongest specific signal; family is mapped from the repo id
  - kohya / native LoRA: safetensors `__metadata__` block carries
    `ss_base_model_version` (family) + `ss_sd_model_name` (specific filename
    of the actual base used during training, often a community fine-tune)
  - HF README YAML frontmatter `base_model:` field
  - Civitai API `baseModel` enum
  - HF repo tags
"""

from __future__ import annotations

from typing import Mapping, Optional


# ---------------------------------------------------------------------------
# Canonical family enum
# ---------------------------------------------------------------------------

CANONICAL_FAMILIES: frozenset[str] = frozenset({
    # Image diffusion
    "sd14", "sd15", "sd2",
    "sd3-medium", "sd35-medium", "sd35-large", "sd35-large-turbo",
    "sdxl", "sdxl-turbo", "sdxl-pony", "sdxl-illustrious",
    "sdxl-distilled", "sdxl-lightning", "sdxl-hyper",
    "flux1-dev", "flux1-schnell", "flux1-kontext", "flux1-krea",
    "flux2-dev", "flux2-klein-4b", "flux2-klein-9b", "flux2-pro",
    "stable-cascade", "playground-v2", "pixart-alpha", "pixart-sigma",
    "lumina", "kolors", "hunyuan-1", "auraflow",
    "qwen-image", "z-image",

    # Video diffusion
    "svd", "svd-xt", "wan21", "wan22", "ltx-video",

    # LLMs (extend as encountered)
    "llama-3-1b", "llama-3-3b", "llama-3-8b", "llama-3-70b",
    "llama-3-2-1b", "llama-3-2-3b", "llama-3-2-11b", "llama-3-2-90b",
    "qwen-2-7b", "qwen-2-72b",
    "qwen-2-5-0-5b", "qwen-2-5-7b", "qwen-2-5-32b", "qwen-2-5-72b",
    "qwen-3-vl",
    "gemma-2-2b", "gemma-2-9b", "gemma-2-27b",
    "gemma-3-4b", "gemma-3-12b",
    "mistral-7b", "mistral-medium",
    "mixtral-8x7b", "mixtral-8x22b",
    "phi-3-mini", "phi-3-medium", "phi-3-5-mini",
    "deepseek-v2", "deepseek-v3", "deepseek-v4",

    # Embeddings / encoders
    "bge-base-en", "bge-large-en",
    "all-minilm-l6-v2", "all-mpnet-base-v2",
    "clip-vit-l-14", "clip-vit-bigg-14",
    "siglip-base", "siglip-large",

    # Sentinel
    "other", "unknown",
})


# ---------------------------------------------------------------------------
# kohya `ss_base_model_version` → canonical family
# ---------------------------------------------------------------------------

_KOHYA_TO_FAMILY: Mapping[str, str] = {
    # SD 1.x / 2.x
    "sd_v1.5": "sd15",
    "sd_v1": "sd15",
    "stable-diffusion-v1-5": "sd15",
    "stable-diffusion-1-5": "sd15",
    "stable-diffusion-v1": "sd15",
    "sd_v2": "sd2",
    "sd_v2.0": "sd2",
    "sd_v2.1": "sd2",
    "stable-diffusion-v2-1": "sd2",
    # SDXL 1.x
    "sdxl_base_v1": "sdxl",
    "sdxl_base_v0_9": "sdxl",
    "stable-diffusion-xl-1.0": "sdxl",
    "stable-diffusion-xl-base-1.0": "sdxl",
    # SDXL fine-tunes (community)
    "pony": "sdxl-pony",
    "pony-diffusion-xl-v6": "sdxl-pony",
    "illustrious": "sdxl-illustrious",
    # FLUX
    "flux1": "flux1-dev",
    "flux.1": "flux1-dev",
    "flux1-dev": "flux1-dev",
    "flux1-schnell": "flux1-schnell",
    "flux2": "flux2-klein-4b",
    "flux2-klein": "flux2-klein-4b",
    "flux2-klein-4b": "flux2-klein-4b",
    "flux2-klein-9b": "flux2-klein-9b",
    # SD 3.x
    "sd3": "sd3-medium",
    "sd3.5": "sd35-medium",
    "sd3.5-large": "sd35-large",
    # Video
    "wan21": "wan21",
    "wan22": "wan22",
    "ltx-video": "ltx-video",
}


# ---------------------------------------------------------------------------
# Civitai `baseModel` enum → canonical family
# ---------------------------------------------------------------------------

_CIVITAI_TO_FAMILY: Mapping[str, str] = {
    "SD 1.4": "sd14",
    "SD 1.5": "sd15",
    "SD 2.0": "sd2",
    "SD 2.1": "sd2",
    "SDXL 0.9": "sdxl",
    "SDXL 1.0": "sdxl",
    "SDXL Distilled": "sdxl-distilled",
    "SDXL Turbo": "sdxl-turbo",
    "SDXL Lightning": "sdxl-lightning",
    "SDXL Hyper": "sdxl-hyper",
    "Pony": "sdxl-pony",
    "Illustrious": "sdxl-illustrious",
    "Stable Cascade": "stable-cascade",
    "SVD": "svd",
    "SVD XT": "svd-xt",
    "Playground v2": "playground-v2",
    "PixArt a": "pixart-alpha",
    "PixArt α": "pixart-alpha",
    "PixArt Σ": "pixart-sigma",
    "PixArt sigma": "pixart-sigma",
    "Hunyuan 1": "hunyuan-1",
    "Lumina": "lumina",
    "Kolors": "kolors",
    "AuraFlow": "auraflow",
    "Flux.1 D": "flux1-dev",
    "Flux.1 S": "flux1-schnell",
    "Flux.2": "flux2-klein-4b",
    "SD 3": "sd3-medium",
    "SD 3.5": "sd35-medium",
    "SD 3.5 Medium": "sd35-medium",
    "SD 3.5 Large": "sd35-large",
    "SD 3.5 Large Turbo": "sd35-large-turbo",
    "Other": "other",
}


# ---------------------------------------------------------------------------
# HF repo id → canonical family (popular bases)
# ---------------------------------------------------------------------------

_REPO_TO_FAMILY: Mapping[str, str] = {
    # SD/SDXL
    "runwayml/stable-diffusion-v1-5": "sd15",
    "stabilityai/stable-diffusion-v1-4": "sd14",
    "stabilityai/stable-diffusion-2-1": "sd2",
    "stabilityai/stable-diffusion-xl-base-1.0": "sdxl",
    "stabilityai/stable-diffusion-xl-base-0.9": "sdxl",
    "stabilityai/stable-diffusion-xl-refiner-1.0": "sdxl",
    "stabilityai/sdxl-turbo": "sdxl-turbo",
    "ByteDance/SDXL-Lightning": "sdxl-lightning",
    "ByteDance/Hyper-SD": "sdxl-hyper",
    "stabilityai/stable-cascade": "stable-cascade",
    "stabilityai/stable-diffusion-3-medium": "sd3-medium",
    "stabilityai/stable-diffusion-3.5-medium": "sd35-medium",
    "stabilityai/stable-diffusion-3.5-large": "sd35-large",
    "stabilityai/stable-diffusion-3.5-large-turbo": "sd35-large-turbo",
    # FLUX
    "black-forest-labs/FLUX.1-dev": "flux1-dev",
    "black-forest-labs/FLUX.1-schnell": "flux1-schnell",
    "black-forest-labs/FLUX.1-Kontext-dev": "flux1-kontext",
    "black-forest-labs/FLUX.1-Krea-dev": "flux1-krea",
    "black-forest-labs/FLUX.2-dev": "flux2-dev",
    "black-forest-labs/FLUX.2-klein-4B": "flux2-klein-4b",
    "black-forest-labs/FLUX.2-klein-9B": "flux2-klein-9b",
    # Video
    "stabilityai/stable-video-diffusion-img2vid": "svd",
    "stabilityai/stable-video-diffusion-img2vid-xt": "svd-xt",
    "Wan-AI/Wan2.1-T2V-A14B": "wan21",
    "Wan-AI/Wan2.2-T2V-A14B": "wan22",
    "Lightricks/LTX-Video": "ltx-video",
    # LLMs
    "meta-llama/Llama-3-8B": "llama-3-8b",
    "meta-llama/Meta-Llama-3-8B": "llama-3-8b",
    "meta-llama/Llama-3-70B": "llama-3-70b",
    "meta-llama/Llama-3.2-1B": "llama-3-2-1b",
    "meta-llama/Llama-3.2-3B": "llama-3-2-3b",
    "Qwen/Qwen2.5-7B": "qwen-2-5-7b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen-2-5-7b",
    "Qwen/Qwen2.5-72B-Instruct": "qwen-2-5-72b",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen-2-5-0-5b",
    "google/gemma-2-2b": "gemma-2-2b",
    "google/gemma-2-9b": "gemma-2-9b",
    "google/gemma-3-4b-it": "gemma-3-4b",
    "mistralai/Mistral-7B-v0.3": "mistral-7b",
    # Embeddings
    "BAAI/bge-large-en-v1.5": "bge-large-en",
    "sentence-transformers/all-MiniLM-L6-v2": "all-minilm-l6-v2",
    "sentence-transformers/all-mpnet-base-v2": "all-mpnet-base-v2",
    "openai/clip-vit-large-patch14": "clip-vit-l-14",
}


# ---------------------------------------------------------------------------
# Tag-based family inference (weakest signal)
# ---------------------------------------------------------------------------

_TAG_TO_FAMILY: Mapping[str, str] = {
    "stable-diffusion": "sd15",
    "stable-diffusion-v1-5": "sd15",
    "stable-diffusion-1-5": "sd15",
    "stable-diffusion-2": "sd2",
    "stable-diffusion-xl": "sdxl",
    "stable-diffusion-xl-base": "sdxl",
    "sdxl": "sdxl",
    "pony": "sdxl-pony",
    "illustrious": "sdxl-illustrious",
    "flux": "flux1-dev",
    "flux-1": "flux1-dev",
    "flux-dev": "flux1-dev",
    "flux-schnell": "flux1-schnell",
    "flux-2": "flux2-klein-4b",
    "wan2-1": "wan21",
    "wan2-2": "wan22",
    "ltx-video": "ltx-video",
}


def kohya_to_family(value: str) -> Optional[str]:
    """Map a kohya `ss_base_model_version` value to the canonical family.
    Returns None for unrecognized inputs."""
    s = str(value or "").strip().lower().replace(" ", "_")
    if not s:
        return None
    if s in _KOHYA_TO_FAMILY:
        return _KOHYA_TO_FAMILY[s]
    # Heuristic fallback — substring match on common patterns
    if "sdxl" in s:
        return "sdxl"
    if "flux1" in s or "flux.1" in s:
        return "flux1-dev"
    if "flux2" in s or "flux.2" in s:
        return "flux2-klein-4b"
    if "sd_v1" in s or "sd-v1" in s or "sd1" in s:
        return "sd15"
    if "sd_v2" in s or "sd-v2" in s or "sd2" in s:
        return "sd2"
    return None


def civitai_to_family(value: str) -> Optional[str]:
    """Map a Civitai `baseModel` enum value to the canonical family.
    Returns None for unrecognized inputs."""
    s = str(value or "").strip()
    if not s:
        return None
    return _CIVITAI_TO_FAMILY.get(s)


def repo_to_family(repo_id: str) -> Optional[str]:
    """Map an HF repo id to the canonical family. Case-insensitive lookup;
    returns None for unrecognized inputs."""
    s = str(repo_id or "").strip()
    if not s:
        return None
    if s in _REPO_TO_FAMILY:
        return _REPO_TO_FAMILY[s]
    # Case-insensitive secondary
    sl = s.lower()
    for k, v in _REPO_TO_FAMILY.items():
        if k.lower() == sl:
            return v
    return None


def tags_to_family(tags) -> Optional[str]:
    """Map an HF YAML `tags:` list to the canonical family.
    Returns None when no tag matches a known family marker."""
    if not isinstance(tags, list):
        return None
    for t in tags:
        s = str(t or "").strip().lower()
        if s in _TAG_TO_FAMILY:
            return _TAG_TO_FAMILY[s]
    return None


def is_canonical_family(value: str) -> bool:
    """True when `value` is in the canonical family enum."""
    return str(value or "").strip().lower() in CANONICAL_FAMILIES


__all__ = [
    "CANONICAL_FAMILIES",
    "kohya_to_family",
    "civitai_to_family",
    "repo_to_family",
    "tags_to_family",
    "is_canonical_family",
]
