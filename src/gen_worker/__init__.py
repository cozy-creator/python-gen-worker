"""Worker-author API for gen-worker.

Keep this surface intentionally small. Endpoint code that needs advanced
subsystems should import their explicit modules, for example
``gen_worker.trainer`` or ``gen_worker.conversion``.

Context types (issue #1: slim-request-context):
  - ``RequestContext`` — inference handlers (the base type).
  - ``ConversionContext`` — ``@training_function`` handlers that produce a
    new repo revision (format-conversion, quantization, fine-tuning, …).
    Adds ``publish_repo_revision`` / ``materialize_blob`` /
    ``read_repo_metadata`` / ``write_repo_metadata`` plus the conversion
    helper API (``mktemp``, ``open_output_writer``, …).
  - ``DatasetContext`` — ``@training_function(kind="dataset-generation")``
    handlers. Adds ``publish_dataset_revision`` / ``resolve_dataset``.
  - ``TrainingContext`` — trainer-class endpoints. Adds repo-metadata RPCs.

All three subclass ``RequestContext``; the kind-specific subclass is
constructed by the worker before dispatch based on the endpoint kind.
"""

from . import io
from .api.decorators import (
    ResourceRequirements,
    ScalingHints,
    inference_function,
)
from .api.injection import ModelRef, ModelRefSource
from .request_context import (
    ConversionContext,
    DatasetContext,
    RequestContext,
    TrainingContext,
)
from .api.errors import (
    AuthError,
    CanceledError,
    FatalError,
    InputTooLargeError,
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
    "ResourceRequirements",
    "ScalingHints",
    "ModelRef",
    "ModelRefSource",
    "RequestContext",
    "ConversionContext",
    "DatasetContext",
    "TrainingContext",
    "AuthError",
    "CanceledError",
    "RetryableError",
    "ResourceError",
    "ValidationError",
    "FatalError",
    "InputTooLargeError",
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
    "io",
]
