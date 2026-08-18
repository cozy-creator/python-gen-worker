"""Models layer: refs, download (ensure_local), memory decisions, residency.

**The package index is LAZY, and that is a serve-path guarantee (pgw#1331).**
An eager re-export of ``residency`` pulls in ``loading``, which constructs
diffusers and transformers objects — so *any* ``from .models import x`` would
execute the whole eager weight loader and the adopt-only serve role could not
be asserted model-free through a package root it never wanted. PEP 562:
importing this package costs nothing,
and asking for a name costs exactly the one submodule that defines it. Same
shape as ``gen_worker/__init__`` and ``gen_worker/model/__init__``, same
reason.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - the eager spelling, for type checkers only
    from .cache_paths import (
        tensorhub_cache_dir,
        tensorhub_cas_dir,
    )
    from .download import (
        build_provider_index_from_manifest,
        ensure_local,
        lookup_provider_for_ref,
        set_provider_index,
    )
    from .refs import (
        HuggingFaceRef,
        ParsedModelRef,
        TensorhubRef,
        WireRef,
        RefFragmentRemoved,
        RetiredTagRef,
        fold_ref,
        format_model_ref,
        normalize_model_ref,
        parse_model_ref,
        refuse_ref_fragment,
    )
    from .residency import (
        LoadedComponentKey,
        content_set_digest,
        Residency,
        Tier,
    )

#: Exported name -> the submodule that defines it. The eager import block it
#: replaces is reproduced verbatim under ``if TYPE_CHECKING`` above, so the two
#: cannot say different things without mypy noticing.
_EXPORTS: Final[dict[str, str]] = {
    "HuggingFaceRef": "refs",
    "Knob": "model_types",
    "SDXL": "model_types",
    "SamplerName": "model_types",
    "LoadedComponentKey": "residency",
    "ParsedModelRef": "refs",
    "RefFragmentRemoved": "refs",
    "Residency": "residency",
    "RetiredTagRef": "refs",
    "TensorhubRef": "refs",
    "Tier": "residency",
    "WireRef": "refs",
    "build_provider_index_from_manifest": "download",
    "content_set_digest": "residency",
    "ensure_local": "download",
    "fold_ref": "refs",
    "format_model_ref": "refs",
    "lookup_provider_for_ref": "download",
    "normalize_model_ref": "refs",
    "parse_model_ref": "refs",
    "refuse_ref_fragment": "refs",
    "set_provider_index": "download",
    "tensorhub_cache_dir": "cache_paths",
    "tensorhub_cas_dir": "cache_paths",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


__all__ = [
    "Knob",
    "SDXL",
    "SamplerName",
    "tensorhub_cache_dir",
    "tensorhub_cas_dir",
    "ensure_local",
    "set_provider_index",
    "lookup_provider_for_ref",
    "build_provider_index_from_manifest",
    "TensorhubRef",
    "HuggingFaceRef",
    "ParsedModelRef",
    "WireRef",
    "parse_model_ref",
    "format_model_ref",
    "normalize_model_ref",
    "fold_ref",
    "RefFragmentRemoved",
    "RetiredTagRef",
    "refuse_ref_fragment",
    "Residency",
    "Tier",
    "LoadedComponentKey",
    "content_set_digest",
]
