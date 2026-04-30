# Make src/gen_worker a Python package
from .api.decorators import (
    ResourceRequirements,
    inference_function,
    realtime_function,
)
# NOTE: @training_function lives in ``gen_worker.conversion`` (not re-exported
# at top level because it pulls in torch / diffusers / transformers via the
# Source + Dataset machinery it dispatches to). Tenants writing training
# endpoints import it directly:
#     from gen_worker.conversion import training_function
from .api.injection import ModelRef, ModelRefSource
from .request_context import RequestContext
from .worker import RealtimeSocket
from .api.errors import AuthError, RetryableError, FatalError, OutputTooLargeError
from .api.types import Asset, Compute, DatasetRef, DestinationRepo, LoraSpec, OutputSpec, SourceRepo, Tensors
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
from .utils.lora import load_loras
from .inference_memory import apply_low_vram_config, with_oom_retry

# Optional torch-dependent exports
try:
    from .pipeline.loader import (
        PipelineLoader,
        PipelineLoaderError,
        ModelNotFoundError,
    )
except ImportError:
    # torch not installed - pipeline_loader not available
    pass

__all__ = [
    # Core exports (training_function in gen_worker.conversion — see import note)
    "inference_function",
    "realtime_function",
    "ResourceRequirements",
    "ModelRef",
    "ModelRefSource",
    "RequestContext",
    "RealtimeSocket",
    "AuthError",
    "RetryableError",
    "FatalError",
    "OutputTooLargeError",
    "Asset",
    "Compute",
    "Tensors",
    "LoraSpec",
    "SourceRepo",
    "DestinationRepo",
    "DatasetRef",
    "OutputSpec",
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
    "PipelineLoaderError",
    "ModelNotFoundError",
    # Low-VRAM inference helpers
    "apply_low_vram_config",
    "with_oom_retry",
]
