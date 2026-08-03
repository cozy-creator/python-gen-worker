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
from .api.compile_axis import AxisClass, CompileAxis
from .api.decorators import (
    Compile,
    ConfigParam,
    DynamicDim,
    NoWarmup,
    Resources,
    endpoint,
    variant_of,
    worker_function,
)
from .api.export_contract import (
    Arg,
    Dim,
    Fork,
    GraphClass,
    Input,
    import_export_declaration,
    register_export_declaration,
)
from .api.formula import RuntimeFormula
from .api.slot import OBJECTIVES, ObjectiveMismatchError, ResolvedSlot, Slot
from .families import GenerationDefaults
from .models.provision import arm_compile
from .text_pin import TextLengthExceededError, pad_text_sequence
from .url_fetch import FetchedUrl, fetch_bytes, fetch_image
from .view import for_request
from .api.errors import (
    CanceledError,
    ChildCallError,
    ChildCallRefusedError,
    ChildCallTimeoutError,
    ChildRequestCanceledError,
    ChildRequestFailedError,
    FatalError,
    IllegalCombination,
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
from .subproc import ProcessStalledError, run_process


__all__ = [
    # The decorators + bindings.
    "endpoint",
    "variant_of",
    "worker_function",
    "Resources",
    "Compile",
    "CompileAxis",
    "AxisClass",
    "DynamicDim",
    # pgw#739 export-declaration vocabulary.
    "Dim",
    "Fork",
    "GraphClass",
    "Input",
    "Arg",
    "register_export_declaration",
    "import_export_declaration",
    "ConfigParam",
    "NoWarmup",
    "arm_compile",
    # SDK v2 per-request views + text pinning.
    "for_request",
    "FetchedUrl",
    "fetch_bytes",
    "fetch_image",
    "pad_text_sequence",
    "TextLengthExceededError",
    "HF",
    "Hub",
    "Civitai",
    "ModelRef",
    "ModelScope",
    # Curated model-selection (payload `model=` placement key).
    # Hub-resolved model slots (pgw#520) + the per-family defaults vocabulary.
    "Slot",
    "ResolvedSlot",
    "GenerationDefaults",
    # pgw#654 objective vocabulary (checkpoint training-objective facts).
    "OBJECTIVES",
    "ObjectiveMismatchError",
    # Context types.
    "RequestContext",
    "ConversionContext",
    "DatasetContext",
    "TrainingContext",
    "TrainingMetric",
    # Per-step progress helper for diffusers pipelines.
    "diffusers_step_callback",
    # Delegated-trainer subprocess primitive.
    "ProcessStalledError",
    "run_process",
    # Errors.
    "CanceledError",
    "RetryableError",
    "ValidationError",
    "FatalError",
    "IllegalCombination",
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
