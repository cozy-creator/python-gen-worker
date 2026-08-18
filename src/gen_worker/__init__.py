"""Worker-author API for gen-worker.

One decorator, four bindings, a slim request context::

    from gen_worker import endpoint, HF, Resources, RequestContext

    @endpoint
    def hello(ctx: RequestContext, data: In) -> Out: ...

``JobContext`` adds the producer-contract surface (publish, mktemp, dataset
resolution) via plain inheritance. It is the ONLY producer context: every
``@job`` body and every producer-shaped ``@endpoint`` handler receives one,
and what a body may write comes from its declaration (``publishes`` /
``emits_media``), never from its kind.


THE PACKAGE INDEX IS LAZY, AND THAT IS A SERVE-PATH GUARANTEE (pgw#1331)
-------------------------------------------------------------------------
Every ``import gen_worker.anything`` executes this file. With an eager block
of re-exports that meant importing a graph binding — on an adopt-only pod that
will never register an endpoint, never load a checkpoint and cannot compile —
executed ``view``, ``models.provision``, ``models.loading``, ``api.streaming``
and the rest of the EAGER-CAPABLE worker's guts, all of which name a model
library inside a function. The serve role could not be asserted model-free
while its own package root dragged them in.

PEP 562 breaks that: importing this package costs nothing, and asking for a
name costs exactly the one submodule that defines it. The author surface is
byte-identical — ``from gen_worker import endpoint`` still works, and so does
``gen_worker.endpoint`` — because ``__getattr__`` resolves through the same
table the eager block used to spell out. ``if TYPE_CHECKING`` keeps the eager
spelling for type checkers, which never execute it.

``scripts/lint_serve_role_closure.py`` is what holds this: with the eager block
back, the whole serve role reaches ``diffusers``/``transformers`` and the fence
names the modules. ``gen_worker.model``'s own ``__init__`` is the same shape
for the same reason, one layer down.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - the eager spelling, for type checkers only
    from . import io
    from .api.binding import (
        Binding,
        Civitai,
        HF,
        Hub,
        ModelRef,
        ModelScope,
    )
    from .api.compile_axis import (
        AxisClass,
        CompileAxis,
    )
    from .api.decorators import (
        AcceptsReferences,
        Compile,
        ConfigParam,
        DynamicDim,
        NoWarmup,
        Resources,
        endpoint,
        variant_of,
        worker_function,
    )
    from .api.model_base import (
        Adapter,
        LoadContext,
        Model,
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
    from .api.formula import (
        RuntimeFormula,
    )
    from .api.jobs import (
        job,
    )
    from .api.progress import (
        diffusers_step_callback,
    )
    from .api.slot import (
        OBJECTIVES,
        ObjectiveMismatchError,
        ResolvedSlot,
        Slot,
        resolve_slot,
    )
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
        ImageFormat,
        MediaAsset,
        PromptRole,
        StringEnum,
        Tensors,
        VideoAsset,
    )
    from .callout import (
        ChildRequest,
    )
    from .families import (
        GenerationDefaults,
    )
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
    from .hub_error import (
        HubApiError,
        HubError,
        parse_hub_error,
        raise_for_hub_error,
    )
    from .models.provision import (
        arm_compile,
        report_applied_attention,
        report_applied_lane,
        report_attention_backend,
    )
    from .models.tensor_layout_contract import (
        LayoutDeclarationError,
        LayoutRequirements,
        RequirementTerms,
    )
    from .serving.endpoint import Endpoint
    from .request_context import (
        JobContext,
        RequestContext,
        TrainingMetric,
    )
    from .subproc import (
        ProcessStalledError,
        run_process,
    )
    from .text_pin import (
        TextLengthExceededError,
        pad_text_sequence,
    )
    from .url_fetch import (
        FetchedUrl,
        fetch_bytes,
    )
    from .view import (
        for_request,
    )

#: Every submodule re-exported as a MODULE rather than as a name in one.
_SUBMODULES: Final[tuple[str, ...]] = ("io",)

#: Exported name -> the submodule that defines it. THE package index, and the
#: reason it exists instead of a block of eager imports. The eager block is
#: reproduced verbatim under ``if TYPE_CHECKING`` above, so the two cannot say
#: different things without mypy noticing.
_EXPORTS: Final[dict[str, str]] = {
    "AcceptsReferences": "api.decorators",
    "Arg": "api.export_contract",
    "Asset": "api.types",
    "AudioAsset": "api.types",
    "AuthError": "api.errors",
    "AxisClass": "api.compile_axis",
    "BatchItemDelta": "api.streaming",
    "Binding": "api.binding",
    "CanceledError": "api.errors",
    "ChildCallError": "api.errors",
    "ChildCallRefusedError": "api.errors",
    "ChildCallTimeoutError": "api.errors",
    "ChildRequest": "callout",
    "ChildRequestCanceledError": "api.errors",
    "ChildRequestFailedError": "api.errors",
    "Civitai": "api.binding",
    "Compile": "api.decorators",
    "CompileAxis": "api.compile_axis",
    "ConfigParam": "api.decorators",
    "DeclarationMismatch": "api.derive",
    "Dim": "api.export_contract",
    "Model": "api.model_base",
    "Done": "api.streaming",
    "DynamicDim": "api.decorators",
    # pgw#1372's serving.endpoint base predates the ratified Model/entrypoint
    # split; it stays exported until that lane cuts over.
    "Endpoint": "serving.endpoint",
    "Adapter": "api.model_base",
    "LoadContext": "api.model_base",
    "Error": "api.streaming",
    "ExpectedOutput": "api.types",
    "FamilyGeometry": "geometry",
    "FatalError": "api.errors",
    "FetchedUrl": "url_fetch",
    "FitMode": "geometry",
    "FitPlan": "geometry",
    "Fork": "api.export_contract",
    "GenerationDefaults": "families",
    "GraphClass": "api.export_contract",
    "HF": "api.binding",
    "Hub": "api.binding",
    "HubApiError": "hub_error",
    "HubError": "hub_error",
    "IllegalCombination": "api.errors",
    "ImageAsset": "api.types",
    "ImageFormat": "api.types",
    "IncrementalTokenDelta": "api.streaming",
    "Input": "api.export_contract",
    "JobContext": "request_context",
    "LayoutDeclarationError": "models.tensor_layout_contract",
    "LayoutRequirements": "models.tensor_layout_contract",
    "MediaAsset": "api.types",
    "MintBlocker": "api.export_contract",
    "ModelRef": "api.binding",
    "ModelScope": "api.binding",
    "NoWarmup": "api.decorators",
    "OBJECTIVES": "api.slot",
    "ObjectiveMismatchError": "api.slot",
    "OutputSize": "geometry",
    "OutputTooLargeError": "api.errors",
    "ProcessStalledError": "subproc",
    "PromptRole": "api.types",
    "RefCompatibilitySurprise": "api.errors",
    "RequestContext": "request_context",
    "RequirementTerms": "models.tensor_layout_contract",
    "ResolvedSlot": "api.slot",
    "ResourceError": "api.errors",
    "Resources": "api.decorators",
    "RestoreResult": "geometry",
    "RetryableError": "api.errors",
    "RuntimeFormula": "api.formula",
    "Slot": "api.slot",
    "SnapshotBuildFailedError": "api.errors",
    "StreamItem": "api.streaming",
    "StreamResult": "api.streaming",
    "StringEnum": "api.types",
    "Tensors": "api.types",
    "TextLengthExceededError": "text_pin",
    "TokenUsage": "api.streaming",
    "TrainingMetric": "request_context",
    "ValidationError": "api.errors",
    "VideoAsset": "api.types",
    "WorkerError": "api.errors",
    "arm_compile": "models.provision",
    "assert_blockers": "api.derive",
    "assert_faithful": "api.derive",
    "cfg_image_classes": "api.derive",
    "class_set_delta": "api.derive",
    "contract_delta": "api.derive",
    "diffusers_step_callback": "api.progress",
    "endpoint": "api.decorators",
    "fetch_bytes": "url_fetch",
    "fit_to_native": "geometry",
    "for_request": "view",
    "import_export_declaration": "api.export_contract",
    "iter_transformers_text_deltas": "api.streaming",
    "job": "api.jobs",
    "nearest_bucket": "geometry",
    "override_delta": "api.derive",
    "pad_text_sequence": "text_pin",
    "parse_hub_error": "hub_error",
    "raise_for_hub_error": "hub_error",
    "register_export_declaration": "api.export_contract",
    "report_applied_attention": "models.provision",
    "report_applied_lane": "models.provision",
    "report_attention_backend": "models.provision",
    "resolve_slot": "api.slot",
    "restore": "geometry",
    "run_process": "subproc",
    "set_upscaler": "geometry",
    "variant_of": "api.decorators",
    "worker_function": "api.decorators",
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return import_module(f"{__name__}.{name}")
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return sorted((*_EXPORTS, *_SUBMODULES))


__all__ = [
    # The decorators + bindings.
    "Adapter",
    "LoadContext",
    "Model",
    "endpoint",
    # pgw#1294: run-once submitted functions. Same (ctx, payload) -> Struct
    # contract as @endpoint, so one body promotes between them unchanged.
    "job",
    "variant_of",
    "worker_function",
    "AcceptsReferences",
    "Resources",
    # pgw#1313 — the one requirement vocabulary, at both levels.
    "LayoutRequirements",
    "RequirementTerms",
    "LayoutDeclarationError",
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
    # pgw#1372: the required minimal endpoint base class.
    "Endpoint",
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
    "ImageFormat",
    "MediaAsset",
    "PromptRole",
    "StringEnum",
    "Tensors",
    "VideoAsset",
    "io",
]
