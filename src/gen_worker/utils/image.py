from typing import List, Union

import numpy as np
from PIL import Image


def _normalize_single(arr: np.ndarray) -> np.ndarray:
    """Sanitize and normalize a single (H, W, C) float array to [0, 1]."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    if arr.min() < -0.01:
        arr = (arr + 1.0) / 2.0
    return arr.clip(0.0, 1.0)


def pipeline_output_to_pil(
    arr: np.ndarray,
) -> Union[Image.Image, List[Image.Image]]:
    """
    Convert a raw diffusers pipeline output array to PIL Image(s).

    Handles the common failure modes in model outputs:
      - NaN / ±inf values (replaced with 0 / clamped)
      - [-1, 1] output range (some models, e.g. ZImagePipeline)
      - [0, 1] output range (standard diffusers output)
      - Batch output (N, H, W, C) — returns a list
      - Single output (H, W, C) — returns a single Image

    Args:
        arr: Float array of shape (H, W, C) or (N, H, W, C), typically float32.

    Returns:
        A single PIL Image for (H, W, C) input, or a list of PIL Images for
        (N, H, W, C) input.
    """
    if arr.ndim == 4:
        return [
            Image.fromarray((_normalize_single(frame) * 255).round().astype("uint8"))
            for frame in arr
        ]
    if arr.ndim == 3:
        return Image.fromarray((_normalize_single(arr) * 255).round().astype("uint8"))
    raise ValueError(f"Expected array with 3 or 4 dimensions, got shape {arr.shape}")
