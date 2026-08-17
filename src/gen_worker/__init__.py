"""Worker-author API for gen-worker.

One decorator, four bindings, a slim request context::

    from gen_worker import endpoint, HF, Resources, RequestContext

    @endpoint
    def hello(ctx: RequestContext, data: In) -> Out: ...

``JobContext`` (aliased for now as ``ConversionContext`` / ``DatasetContext``
/ ``TrainingContext``) adds the producer-contract surface (publish, mktemp,
dataset resolution) via plain inheritance; the worker constructs the right
subclass from ``@endpoint(kind=...)`` before dispatch.
"""

from . import io
from .api.binding import Binding, Civitai, HF, Hub, ModelRef, ModelScope
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
    AcceptsReferences,
)
from .api.export_contract import (
    Arg,
    Dim,
    Fork,
    GraphClass,
    Input,
    MintBlocker,
    import_export_declaration,
    register_export_declaration,
)
from .api.derive import (
    DeclarationMismatch,
    assert_blockers,
    assert_faithful,
    cfg_image_classes,
    class_set_delta,
    contract_delta,
    override_delta,
)
from .api.formula import RuntimeFormula
from .api.slot import OBJECTIVES, ObjectiveMismatchError, ResolvedSlot, Slot, resolve_slot
from .families import GenerationDefaults
from .models.provision import (arm_compile, report_applied_attention,
                               report_applied_lane, report_attention_backend)
from .text_pin import TextLengthExceededError, pad_text_sequence
from .geometry import (
    FamilyGeometry,
    FitMode,
    FitPlan,
    OutputSize,
    RestoreResult,
    fit_to_native,
    nearest_bucket,
    restore,
    set_upscaler,
)
from .url_fetch import FetchedUrl, fetch_bytes
from .view import for_request
from .api.errors import (
    AuthError,
    CanceledError,
    ChildCallError,
    ChildCallRefusedError,
    ChildCallTimeoutError,
    ChildRequestCanceledError,
    ChildRequestFailedError,
    FatalError,
    IllegalCombination,
    OutputTooLargeError,
    RefCompatibilitySurprise,
    ResourceError,
    RetryableError,
    SnapshotBuildFailedError,
    ValidationError,
    WorkerError,
)
from .hub_error import HubApiError, HubError, parse_hub_error, raise_for_hub_error
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
    MediaAsset,
    PromptRole,
    StringEnum,
    Tensors,
    VideoAsset,
)
from .request_context import (
    ConversionContext,
    DatasetContext,
    JobContext,
    RequestContext,
    TrainingContext,
    TrainingMetric,
)
from .api.jobs import job
from .subproc import ProcessStalledError, run_process


__all__ = [
    # The decorators + bindings.
    "endpoint",
    # pgw#1294: run-once submitted functions. Same (ctx, payload) -> Struct
    # contract as @endpoint, so one body promotes between them unchanged.
    "job",
    "variant_of",
    "worker_function",
    "AcceptsReferences",
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
    # pgw#1115: a mint refusal is DATA on the declaration.
    "MintBlocker",
    "register_export_declaration",
    "DeclarationMismatch",
    "assert_blockers",
    "assert_faithful",
    "cfg_image_classes",
    "class_set_delta",
    "contract_delta",
    "override_delta",
    "import_export_declaration",
    "ConfigParam",
    "NoWarmup",
    "arm_compile",
    # pgw#1104: the serve-time recipe reports the lane it APPLIED.
    "report_applied_attention",
    "report_applied_lane",
    # th#1871 P1: the attention KERNEL axis, which the sparsity reporter
    # correctly refuses to carry.
    "report_attention_backend",
    # SDK v2 per-request views + text pinning.
    "for_request",
    "FetchedUrl",
    "fetch_bytes",
    # pgw#664/ie#599 fit-to-native geometry: mechanism here, table in the family.
    "FamilyGeometry",
    "FitMode",
    "FitPlan",
    "OutputSize",
    "RestoreResult",
    "fit_to_native",
    "nearest_bucket",
    "restore",
    "set_upscaler",
    "pad_text_sequence",
    "TextLengthExceededError",
    "HF",
    "Hub",
    "Civitai",
    "Binding",
    "ModelRef",
    "ModelScope",
    # Curated model-selection (payload `model=` placement key).
    # Hub-resolved model slots (pgw#520) + the per-family defaults vocabulary.
    "Slot",
    "ResolvedSlot",
    "resolve_slot",
    "GenerationDefaults",
    # pgw#654 objective vocabulary (checkpoint training-objective facts).
    "OBJECTIVES",
    "ObjectiveMismatchError",
    # Context types.
    "RequestContext",
    "JobContext",
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
    "AuthError",
    "OutputTooLargeError",
    "RefCompatibilitySurprise",
    "ResourceError",
    "SnapshotBuildFailedError",
    "WorkerError",
    # pgw#1229: the hub's typed error envelope, for any endpoint HTTP call.
    "HubApiError",
    "HubError",
    "parse_hub_error",
    "raise_for_hub_error",
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
    "MediaAsset",
    "PromptRole",
    "StringEnum",
    "Tensors",
    "VideoAsset",
    "io",
]
