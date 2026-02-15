# Make src/gen_worker a Python package
from .decorators import ResourceRequirements, worker_function, worker_websocket
from .injection import ModelRef, ModelRefSource
from .worker import ActionContext, RealtimeSocket
from .errors import AuthError, RetryableError, FatalError
from .types import Asset
from .model_interface import ModelManager
from .downloader import ModelDownloader, CozyHubDownloader
from .endpoint_validation import EndpointValidationResult, validate_endpoint
from .model_cache import ModelCache, ModelCacheStats, ModelLocation
from .payload_constraints import Clamp

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
    "EndpointValidationResult",
    "validate_endpoint",
    # Model cache (always available)
    "ModelCache",
    "ModelCacheStats",
    "ModelLocation",
    "Clamp",
    # Pipeline loader (torch-dependent, may not be available)
    "PipelineLoader",
    "PipelineConfig",
    "LoadedPipeline",
    "PipelineLoaderError",
    "ModelNotFoundError",
    "CudaOutOfMemoryError",
]
