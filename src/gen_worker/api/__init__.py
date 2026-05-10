"""Public SDK surface: decorators, types, errors, and helpers for tenant code."""

from .decorators import (
    ResourceRequirements,
    inference_function,
    realtime_function,
)
from .errors import (
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
from .injection import ModelRef, ModelRefSource
from .payload_constraints import Clamp
from .streaming import iter_transformers_text_deltas
from .types import Asset, Compute, LoraSpec, Tensors

__all__ = [
    "ResourceRequirements",
    "inference_function",
    "realtime_function",
    "AuthError",
    "CanceledError",
    "FatalError",
    "OutputTooLargeError",
    "RefCompatibilitySurprise",
    "ResourceError",
    "RetryableError",
    "ValidationError",
    "WorkerError",
    "ModelRef",
    "ModelRefSource",
    "Clamp",
    "iter_transformers_text_deltas",
    "Asset",
    "Compute",
    "LoraSpec",
    "Tensors",
]
