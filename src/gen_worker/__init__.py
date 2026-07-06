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
from .api.binding import Civitai, HF, Hub, ModelScope
from .api.decorators import Compile, Resources, endpoint
from .api.errors import (
    CanceledError,
    FatalError,
    RetryableError,
    ValidationError,
)
from .api.streaming import (
    BatchItemDelta,
    Done,
    Error,
    IncrementalTokenDelta,
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
from .diagnostics import emit_diagnostic_log
from .request_context import (
    ConversionContext,
    DatasetContext,
    RequestContext,
    TrainingContext,
)


__all__ = [
    # The decorator + bindings.
    "endpoint",
    "Resources",
    "Compile",
    "HF",
    "Hub",
    "Civitai",
    "ModelScope",
    # Context types.
    "RequestContext",
    "ConversionContext",
    "DatasetContext",
    "TrainingContext",
    # Errors.
    "CanceledError",
    "RetryableError",
    "ValidationError",
    "FatalError",
    # Streaming signals.
    "BatchItemDelta",
    "Done",
    "Error",
    "IncrementalTokenDelta",
    "iter_transformers_text_deltas",
    # Payload + media helpers.
    "Asset",
    "AudioAsset",
    "ExpectedOutput",
    "ImageAsset",
    "StringEnum",
    "VideoAsset",
    "emit_diagnostic_log",
    "io",
]
