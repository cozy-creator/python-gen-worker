"""Model references, downloading, and caching."""

from .cache import ModelCache, ModelCacheStats, ModelLocation
from .cache_paths import (
    tensorhub_cache_dir,
    tensorhub_cas_dir,
    worker_local_model_cache_dir_default,
)
from .downloader import ModelDownloader
from .interface import ModelManagementInterface, ModelManager
from .refs import TensorhubRef, HuggingFaceRef, ParsedModelRef, parse_model_ref
from .shared_components import (
    LoadedComponentKey,
    SharedComponentCache,
    SharedComponentStats,
    build_function_owned_pipeline,
)

__all__ = [
    "ModelCache",
    "ModelCacheStats",
    "ModelLocation",
    "tensorhub_cache_dir",
    "tensorhub_cas_dir",
    "worker_local_model_cache_dir_default",
    "ModelDownloader",
    "ModelManagementInterface",
    "ModelManager",
    "TensorhubRef",
    "HuggingFaceRef",
    "ParsedModelRef",
    "parse_model_ref",
    "LoadedComponentKey",
    "SharedComponentCache",
    "SharedComponentStats",
    "build_function_owned_pipeline",
]
