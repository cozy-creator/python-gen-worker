"""Worker-author API for gen-worker.

One decorator, four bindings, a slim request context::

    from gen_worker import endpoint, HF, Resources, RequestContext

    @endpoint
    def hello(ctx: RequestContext, data: In) -> Out: ...

Kind-specific contexts (``ConversionContext`` / ``DatasetContext`` /
``TrainingContext``) add the producer-contract surface (publish, mktemp,
dataset resolution) via plain inheritance; the worker constructs the right
subclass from ``@endpoint(kind=...)`` before dispatch.
"""

from . import io
from .api.binding import Civitai, HF, Hub, ModelRef, ModelScope
from .api.decorators import Compile, NoWarmup, Resources, endpoint, variant_of
from .api.model import Model, ModelChoice, ModelDefaults
from .api.slot import ResolvedSlot, Slot
from .families import FamilyDefaults
from .models.provision import arm_compile
from .api.errors import (
    CanceledError,
    ChildCallError,
    ChildCallRefusedError,
    ChildCallTimeoutError,
    ChildRequestCanceledError,
    ChildRequestFailedError,
    FatalError,
    RetryableError,
    ValidationError,
)
from .callout import ChildRequest
from .api.progress import diffusers_step_callback
from .api.streaming import (
    BatchItemDelta,
    Done,
    Error,
    IncrementalTokenDelta,
    StreamItem,
    StreamResult,
    TokenUsage,
    iter_transformers_text_deltas,
)
from .api.types import (
    Asset,
    AudioAsset,
    ExpectedOutput,
    ImageAsset,
    StringEnum,
    VideoAsset,
)
from .request_context import (
    ConversionContext,
    DatasetContext,
    RequestContext,
    TrainingContext,
    TrainingMetric,
)
from .subproc import run_process


__all__ = [
    # The decorator + bindings.
    "endpoint",
    "variant_of",
    "Resources",
    "Compile",
    "NoWarmup",
    "arm_compile",
    "HF",
    "Hub",
    "Civitai",
    "ModelRef",
    "ModelScope",
    # Curated model-selection (payload `model=` placement key).
    "Model",
    "ModelChoice",
    "ModelDefaults",
    # Hub-resolved model slots (pgw#520) + the per-family defaults vocabulary.
    "Slot",
    "ResolvedSlot",
    "FamilyDefaults",
    # Context types.
    "RequestContext",
    "ConversionContext",
    "DatasetContext",
    "TrainingContext",
    "TrainingMetric",
    # Per-step progress helper for diffusers pipelines.
    "diffusers_step_callback",
    # Delegated-trainer subprocess primitive.
    "run_process",
    # Errors.
    "CanceledError",
    "RetryableError",
    "ValidationError",
    "FatalError",
    # th#826 call-out primitive (ctx.call_endpoint / ctx.workflow_checkpoint).
    "ChildRequest",
    "ChildCallError",
    "ChildCallRefusedError",
    "ChildCallTimeoutError",
    "ChildRequestCanceledError",
    "ChildRequestFailedError",
    # Streaming signals.
    "BatchItemDelta",
    "Done",
    "Error",
    "IncrementalTokenDelta",
    "StreamItem",
    "StreamResult",
    "TokenUsage",
    "iter_transformers_text_deltas",
    # Payload + media helpers.
    "Asset",
    "AudioAsset",
    "ExpectedOutput",
    "ImageAsset",
    "StringEnum",
    "VideoAsset",
    "io",
]
