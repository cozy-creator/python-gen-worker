# Make src/gen_worker a Python package
from .api.decorators import ResourceRequirements, worker_function, worker_websocket
from .api.injection import ModelRef, ModelRefSource
from .request_context import RequestContext
from .worker import RealtimeSocket
from .api.errors import AuthError, RetryableError, FatalError
from .api.types import Asset, LoraSpec
from .models.interface import ModelManager
from .models.downloader import ModelDownloader, CozyHubDownloader
from .discovery.validation import EndpointValidationResult, validate_endpoint
from .models.cache import ModelCache, ModelCacheStats, ModelLocation
from .api.payload_constraints import Clamp
from .trainer import (
    StepContext,
    StepControlHints,
    StepResult,
    TrainingJobSpec,
)
from .api.streaming import iter_transformers_text_deltas
from .utils.image import image_output_sanitizer

# Optional torch-dependent exports
try:
    from .pipeline.loader import (
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
    "RequestContext",
    "RealtimeSocket",
    "AuthError",
    "RetryableError",
    "FatalError",
    "Asset",
    "LoraSpec",
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
    "StepContext",
    "StepControlHints",
    "StepResult",
    "TrainingJobSpec",
    "iter_transformers_text_deltas",
    # Pipeline loader (torch-dependent, may not be available)
    "PipelineLoader",
    "PipelineConfig",
    "LoadedPipeline",
    "PipelineLoaderError",
    "ModelNotFoundError",
    "CudaOutOfMemoryError",
]
