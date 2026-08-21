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

With the eager block back, the whole serve role would reach
``diffusers``/``transformers``. ``gen_worker.model``'s own ``__init__`` is the same shape
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
    from .api.progress import (
        diffusers_step_callback,
    )
    from .api.types import (
        Asset,
        AudioAsset,
        ExpectedOutput,
        ImageAsset,
        MediaAsset,
        PromptRole,
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
        report_applied_attention,
        report_applied_lane,
        report_attention_backend,
    )
    from .models.tensor_layout_contract import (
        LayoutDeclarationError,
        LayoutRequirements,
        RequirementTerms,
    )
    from .io import ImageFormat
    from .request_context import (
        JobContext,
        TrainingMetric,
    )
    from .serving.context import (
        Adapter,
        DistillationAdapter,
        LoadContext,
        RequestContext,
    )
    from .serving.engine_runtime import (
        EngineBootError,
        EngineHandle,
        EngineSpec,
        LlamaServer,
        VllmServer,
    )
    from .serving.deltas import (
        Delta,
        ItemDelta,
        TokenDelta,
        iter_transformers_text_deltas,
    )
    from .serving.entrypoints import entrypoint
    from .api.resources import Resources
    from .serving.model import Model
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
    "Asset": "api.types",
    "AudioAsset": "api.types",
    "AuthError": "api.errors",
    "Binding": "api.binding",
    "CanceledError": "api.errors",
    "ChildCallError": "api.errors",
    "ChildCallRefusedError": "api.errors",
    "ChildCallTimeoutError": "api.errors",
    "ChildRequest": "callout",
    "ChildRequestCanceledError": "api.errors",
    "ChildRequestFailedError": "api.errors",
    "Civitai": "api.binding",
    # pgw#1382: the Model/Endpoint split author surface. Model is the
    # stateful class; @entrypoint marks stateless module-level functions;
    # ctx splits into LoadContext (load moment) + RequestContext (request
    # moment); Adapter slots are explicit entrypoint parameters.
    "Adapter": "serving.context",
    # pgw#1421: the ENGINE-HOSTED tier's author surface. A spec DECLARES the
    # engine (`LlamaServer`/`VllmServer`); `ctx.engine(spec)` boots and
    # supervises it and hands back an `EngineHandle` with a `base_url`. This
    # is F3's eager-permanent world — external binaries only.
    "EngineBootError": "serving.engine_runtime",
    "EngineHandle": "serving.engine_runtime",
    "EngineSpec": "serving.engine_runtime",
    "LlamaServer": "serving.engine_runtime",
    "VllmServer": "serving.engine_runtime",
    "Resources": "api.resources",
    "DistillationAdapter": "serving.context",
    # pgw#1576: INCREMENTAL OUTPUT. `@entrypoint(streams=<type>)` declares the
    # chunk type, `ctx.emit(chunk)` puts one on the droppable JobProgress lane,
    # and the entrypoint still RETURNS its terminal struct on the authoritative
    # one — two wire channels, and the declaration names both.
    "Delta": "serving.deltas",
    "ItemDelta": "serving.deltas",
    "TokenDelta": "serving.deltas",
    "iter_transformers_text_deltas": "serving.deltas",
    "ExpectedOutput": "api.types",
    "FamilyGeometry": "geometry",
    "FatalError": "api.errors",
    "FetchedUrl": "url_fetch",
    "FitMode": "geometry",
    "FitPlan": "geometry",
    "GenerationDefaults": "families",
    "HF": "api.binding",
    "Hub": "api.binding",
    "HubApiError": "hub_error",
    "HubError": "hub_error",
    "IllegalCombination": "api.errors",
    "ImageAsset": "api.types",
    "ImageFormat": "io",
    "JobContext": "request_context",
    "LayoutDeclarationError": "models.tensor_layout_contract",
    "LoadContext": "serving.context",
    "LayoutRequirements": "models.tensor_layout_contract",
    "MediaAsset": "api.types",
    "Model": "serving.model",
    # pgw#1599 — the lane declaration surface. The demand TERM ALGEBRA lives
    # in `gen_worker.demand` (`from gen_worker.demand import const, per_mp_batch,
    # GiB, MiB`), deliberately namespaced: `const` is too common a word to own
    # at the package root.
    "DYNAMIC": "serving.lane_spec",
    "DeclaredLane": "serving.lane_spec",
    "LaneDeclarationError": "serving.lane_spec",
    "LaneSpec": "serving.lane_spec",
    "STATIC": "serving.lane_spec",
    "Structural": "serving.lane_spec",
    "lane": "serving.lane_spec",
    "ModelRef": "api.binding",
    "ModelScope": "api.binding",
    "OutputSize": "geometry",
    "OutputTooLargeError": "api.errors",
    "ProcessStalledError": "subproc",
    "PromptRole": "api.types",
    "RefCompatibilitySurprise": "api.errors",
    # pgw#1382: THE RequestContext is the serving one (request facts +
    # salvaged base surface); JobContext stays on the base module.
    "RequestContext": "serving.context",
    "RequirementTerms": "models.tensor_layout_contract",
    "ResourceError": "api.errors",
    "RestoreResult": "geometry",
    "RetryableError": "api.errors",
    "SnapshotBuildFailedError": "api.errors",
    "Tensors": "api.types",
    "TextLengthExceededError": "text_pin",
    "TrainingMetric": "request_context",
    "ValidationError": "api.errors",
    "VideoAsset": "api.types",
    "WorkerError": "api.errors",
    "diffusers_step_callback": "api.progress",
    "entrypoint": "serving.entrypoints",
    "fetch_bytes": "url_fetch",
    "fit_to_native": "geometry",
    "for_request": "view",
    "nearest_bucket": "geometry",
    "pad_text_sequence": "text_pin",
    "parse_hub_error": "hub_error",
    "raise_for_hub_error": "hub_error",
    "report_applied_attention": "models.provision",
    "report_applied_lane": "models.provision",
    "report_attention_backend": "models.provision",
    "restore": "geometry",
    "run_process": "subproc",
    "set_upscaler": "geometry",
}

def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return import_module(f"{__name__}.{name}")
    module = _EXPORTS.get(name)
    if module is None:
        # pgw#1373: a DELETED v1 name refuses by name, naming the migration.
        # A bare AttributeError here is the silent-absent state the tracker's
        # typed-refusal rule forbids — 27 endpoints would each read
        # "cannot import name 'endpoint'" and learn nothing.
        from .v1_deleted import REPLACEMENTS, refuse

        if name in REPLACEMENTS:
            raise refuse(name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return sorted((*_EXPORTS, *_SUBMODULES))


__all__ = [
    # pgw#1382: the Model/Endpoint split author surface.
    "Adapter",
    "DistillationAdapter",
    "ImageFormat",
    "LoadContext",
    "Model",
    "Resources",
    "entrypoint",
    # pgw#1576: incremental output — the declared chunk types + the
    # transformers text-delta helper.
    "Delta",
    "ItemDelta",
    "TokenDelta",
    "iter_transformers_text_deltas",
    # pgw#1421: the engine-hosted tier (external binaries only).
    "EngineBootError",
    "EngineHandle",
    "EngineSpec",
    "LlamaServer",
    "VllmServer",
    # The decorators + bindings.
    # pgw#1294: run-once submitted functions. Same (ctx, payload) -> Struct
    # contract as @endpoint, so one body promotes between them unchanged.
    # pgw#1313 — the one requirement vocabulary, at both levels.
    "LayoutRequirements",
    "RequirementTerms",
    "LayoutDeclarationError",
    # pgw#739 export-declaration vocabulary.
    # pgw#1115: a mint refusal is DATA on the declaration.
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
    "GenerationDefaults",
    # pgw#654 objective vocabulary (checkpoint training-objective facts).
    # Context types.
    "RequestContext",
    "JobContext",
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
    # Payload + media helpers.
    "Asset",
    "AudioAsset",
    "ExpectedOutput",
    "ImageAsset",
    "MediaAsset",
    "PromptRole",
    "DYNAMIC",
    "DeclaredLane",
    "LaneDeclarationError",
    "LaneSpec",
    "STATIC",
    "Structural",
    "lane",
    "Tensors",
    "VideoAsset",
    "io",
]
