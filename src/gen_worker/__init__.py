# Make src/gen_worker a Python package
from .decorators import ResourceRequirements, worker_function, worker_websocket
from .injection import ModelRef, ModelRefSource
from .worker import ActionContext, RealtimeSocket
from .errors import AuthError, RetryableError, FatalError
from .types import Asset
from .model_interface import ModelManager
from .downloader import ModelDownloader, CozyHubDownloader
from .project_validation import ProjectValidationResult, validate_project
from .model_cache import ModelCache, ModelCacheStats, ModelLocation

# Optional torch-dependent exports
try:
    from .pipeline_loader import (
        PipelineLoader,
        PipelineConfig,
        LoadedPipeline,
        PipelineLoaderError,
        ModelNotFoundError,
        CudaOutOfMemoryError,
    )
except ImportError:
    # torch not installed - pipeline_loader not available
    pass

__all__ = [
    # Core exports
    "worker_function",
    "worker_websocket",
    "ResourceRequirements",
    "ModelRef",
    "ModelRefSource",
    "ActionContext",
    "RealtimeSocket",
    "AuthError",
    "RetryableError",
    "FatalError",
    "Asset",
    "ModelManager",
    "ModelDownloader",
    "CozyHubDownloader",
    "ProjectValidationResult",
    "validate_project",
    # Model cache (always available)
    "ModelCache",
    "ModelCacheStats",
    "ModelLocation",
    # Pipeline loader (torch-dependent, may not be available)
    "PipelineLoader",
    "PipelineConfig",
    "LoadedPipeline",
    "PipelineLoaderError",
    "ModelNotFoundError",
    "CudaOutOfMemoryError",
]
