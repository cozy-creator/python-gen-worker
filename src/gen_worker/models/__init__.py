"""Models layer: refs, download (ensure_local), memory decisions, residency."""

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

__all__ = [
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
