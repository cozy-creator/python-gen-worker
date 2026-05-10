"""Worker-author API for gen-worker.

Keep this surface intentionally small. Endpoint code that needs advanced
subsystems should import their explicit modules, for example
``gen_worker.trainer`` or ``gen_worker.conversion``.
"""

from .api.decorators import (
    ResourceRequirements,
    inference_function,
    realtime_function,
)
from .api.injection import ModelRef, ModelRefSource
from .request_context import RequestContext
from .worker import RealtimeSocket
from .api.errors import (
    AuthError,
    CanceledError,
    FatalError,
    OutputTooLargeError,
    RefCompatibilitySurprise,
    ResourceError,
    RetryableError,
    ValidationError,
    WorkerError,
)
from .api.types import Asset, Compute, LoraSpec, Tensors
from .api.payload_constraints import Clamp
from .api.streaming import iter_transformers_text_deltas
from .utils.lora import load_loras
from .inference_memory import apply_low_vram_config, with_oom_retry


def __getattr__(name: str):
    if name == "clone":
        import importlib

        return importlib.import_module(".clone", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "inference_function",
    "realtime_function",
    "ResourceRequirements",
    "ModelRef",
    "ModelRefSource",
    "RequestContext",
    "RealtimeSocket",
    "AuthError",
    "CanceledError",
    "RetryableError",
    "ResourceError",
    "ValidationError",
    "FatalError",
    "OutputTooLargeError",
    "RefCompatibilitySurprise",
    "WorkerError",
    "Asset",
    "Compute",
    "Tensors",
    "LoraSpec",
    "Clamp",
    "iter_transformers_text_deltas",
    "load_loras",
    "apply_low_vram_config",
    "with_oom_retry",
    "clone",
]
