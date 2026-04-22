"""HF/diffusers component-level loading dispatch for Component.as_hf_module()."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_component_module(component_dir: Path, config: dict, **kwargs: Any) -> Any:
    """Load a single diffusers component as its HF class.

    Uses ``config['_class_name']`` (the diffusers convention) to pick the
    class. Falls back to ``transformers.AutoModel`` for components whose
    class isn't registered under diffusers (text encoders, image encoders).
    """
    class_name = config.get("_class_name") or config.get("class_name")
    if not class_name:
        raise ValueError(
            f"component {component_dir.name}: missing _class_name in config"
        )
    # Try diffusers first
    try:
        import diffusers
        cls = getattr(diffusers, class_name, None)
        if cls is not None:
            return cls.from_pretrained(str(component_dir), **kwargs)
    except ImportError:
        pass
    # Try transformers
    try:
        import transformers
        cls = getattr(transformers, class_name, None)
        if cls is not None:
            return cls.from_pretrained(str(component_dir), **kwargs)
        # Last resort: generic AutoModel
        return transformers.AutoModel.from_pretrained(str(component_dir), **kwargs)
    except ImportError:
        pass
    raise RuntimeError(
        f"component {component_dir.name}: could not resolve class {class_name!r} "
        "in diffusers or transformers"
    )


__all__ = ["load_component_module"]
