"""HF/diffusers component-level loading dispatch for Component.as_hf_module()."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_component_module(component_dir: Path, config: dict, **kwargs: Any) -> Any:
    """Load a single diffusers / transformers component as its HF class.

    Two component-config conventions exist in the wild:

      - **diffusers**: ``config["_class_name"]`` (or sometimes
        ``"class_name"``) names the diffusers Python class (e.g.
        ``"FluxTransformer2DModel"``, ``"AutoencoderKLFlux2"``).
      - **transformers**: ``config["architectures"]`` is a list whose first
        entry names the transformers Python class (e.g.
        ``"Qwen3ForCausalLM"``, ``"CLIPTextModel"``). No ``_class_name``.

    Components like ``transformer`` / ``unet`` / ``vae`` follow the diffusers
    convention; text encoders (``text_encoder`` / ``text_encoder_2`` /
    ``text_encoder_3``) follow the transformers convention. We try both in
    order and fall back to ``transformers.AutoModel`` only when neither
    convention names a resolvable class — that fallback exists for
    forward-compat with components that ship without an explicit class hint.
    """
    diffusers_class = config.get("_class_name") or config.get("class_name")
    architectures = config.get("architectures") or []
    transformers_class = (
        architectures[0] if isinstance(architectures, list) and architectures else ""
    )

    # Diffusers path first — `_class_name` is the strongest hint.
    if diffusers_class:
        try:
            import diffusers
            cls = getattr(diffusers, diffusers_class, None)
            if cls is not None:
                return cls.from_pretrained(str(component_dir), **kwargs)
        except ImportError:
            pass
        try:
            import transformers
            cls = getattr(transformers, diffusers_class, None)
            if cls is not None:
                return cls.from_pretrained(str(component_dir), **kwargs)
        except ImportError:
            pass

    # Transformers path: `architectures[0]` is the canonical hint for
    # transformers-style configs.
    if transformers_class:
        try:
            import transformers
            cls = getattr(transformers, transformers_class, None)
            if cls is not None:
                return cls.from_pretrained(str(component_dir), **kwargs)
        except ImportError:
            pass

    # Last-resort generic loader. Works for most transformers-style configs
    # because AutoModel resolves the class via the config's `model_type` or
    # `architectures` field internally.
    if transformers_class or architectures:
        try:
            import transformers
            return transformers.AutoModel.from_pretrained(str(component_dir), **kwargs)
        except ImportError:
            pass

    if not (diffusers_class or transformers_class):
        raise ValueError(
            f"component {component_dir.name}: config has neither "
            f"`_class_name` (diffusers) nor `architectures` (transformers) — "
            f"can't pick a loader"
        )
    raise RuntimeError(
        f"component {component_dir.name}: could not resolve class "
        f"{diffusers_class or transformers_class!r} in diffusers or transformers"
    )


__all__ = ["load_component_module"]
