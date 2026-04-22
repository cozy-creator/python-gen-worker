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
    ResourceError,
    RetryableError,
    ValidationError,
    WorkerError,
)
from .injection import InjectionSpec, ModelRef, ModelRefSource, parse_injection
from .payload_constraints import Clamp
from .streaming import iter_transformers_text_deltas
from .types import Asset, Tensors

__all__ = [
    "ResourceRequirements",
    "inference_function",
    "realtime_function",
    "AuthError",
    "CanceledError",
    "FatalError",
    "OutputTooLargeError",
    "ResourceError",
    "RetryableError",
    "ValidationError",
    "WorkerError",
    "InjectionSpec",
    "ModelRef",
    "ModelRefSource",
    "parse_injection",
    "Clamp",
    "iter_transformers_text_deltas",
    "Asset",
    "Tensors",
]
