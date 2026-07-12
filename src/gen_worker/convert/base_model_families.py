"""Civitai `baseModel` -> canonical family mapping.

Used by the civitai ingest path (`convert/ingest.py`) to stamp
`base_model_family` on the destination LoRA/checkpoint from the Civitai API's
`baseModel` enum value.
"""

from __future__ import annotations

from typing import Mapping, Optional


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
    "Flux.2": "flux.2-klein-4b",
    "SD 3": "sd3-medium",
    "SD 3.5": "sd35-medium",
    "SD 3.5 Medium": "sd35-medium",
    "SD 3.5 Large": "sd35-large",
    "SD 3.5 Large Turbo": "sd35-large-turbo",
    "Flux.2 Klein 9B": "flux.2-klein-9b",
    "Flux.2 Klein 9B-base": "flux.2-klein-9b",
    "ZImageTurbo": "z-image-turbo",
    "ZImageBase": "z-image",
    "Qwen": "qwen-image",
    "Ernie": "ernie",
    "Anima": "anima",
    "NoobAI": "sdxl-illustrious",
    "SD 1.5 LCM": "sd15",
    "SDXL 1.0 LCM": "sdxl",
    "Wan Video 2.2 I2V-A14B": "wan22",
    "Wan Video 2.2 T2V-A14B": "wan22",
    "Wan Video 2.2 TI2V-5B": "wan22",
    "Other": "other",
}


def civitai_to_family(value: str) -> Optional[str]:
    """Map a Civitai `baseModel` enum value to the canonical family.
    Returns None for unrecognized inputs."""
    s = str(value or "").strip()
    if not s:
        return None
    return _CIVITAI_TO_FAMILY.get(s)


__all__ = ["civitai_to_family"]
