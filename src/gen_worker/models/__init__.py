"""Models layer: refs, download (ensure_local), memory decisions, residency."""

from .cache_paths import (
    tensorhub_cache_dir,
    tensorhub_cas_dir,
    worker_local_model_cache_dir_default,
)
from .download import (
    build_provider_index_from_manifest,
    ensure_local,
    ensure_local_sync,
    lookup_provider_for_ref,
    set_provider_index,
)
from .refs import (
    HuggingFaceRef,
    ParsedModelRef,
    TensorhubRef,
    WireRef,
    flavor_token,
    fold_ref,
    format_model_ref,
    normalize_model_ref,
    parse_model_ref,
)
from .residency import (
    LoadedComponentKey,
    content_set_digest,
    Residency,
    Tier,
    build_function_owned_pipeline,
)

__all__ = [
    "tensorhub_cache_dir",
    "tensorhub_cas_dir",
    "worker_local_model_cache_dir_default",
    "ensure_local",
    "ensure_local_sync",
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
    "flavor_token",
    "Residency",
    "Tier",
    "LoadedComponentKey",
    "content_set_digest",
    "build_function_owned_pipeline",
]
