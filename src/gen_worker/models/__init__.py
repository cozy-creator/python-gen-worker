"""Model references, downloading, and caching."""

from .cache import ModelCache, ModelCacheStats, ModelLocation
from .cache_paths import (
    tensorhub_cache_dir,
    tensorhub_cas_dir,
    worker_local_model_cache_dir_default,
    worker_model_cache_dir,
)
from .downloader import CozyHubDownloader, ModelDownloader
from .interface import ModelManagementInterface, ModelManager
from .refs import CozyRef, HuggingFaceRef, ParsedModelRef, parse_model_ref

__all__ = [
    "ModelCache",
    "ModelCacheStats",
    "ModelLocation",
    "tensorhub_cache_dir",
    "tensorhub_cas_dir",
    "worker_local_model_cache_dir_default",
    "worker_model_cache_dir",
    "CozyHubDownloader",
    "ModelDownloader",
    "ModelManagementInterface",
    "ModelManager",
    "CozyRef",
    "HuggingFaceRef",
    "ParsedModelRef",
    "parse_model_ref",
]
