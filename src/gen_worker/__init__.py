"""Worker-author API for gen-worker. The package index is lazy (PEP 562) on purpose: an eager re-export block would drag model libraries into the adopt-only serve role, which must be importable model-free."""

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

_SUBMODULES: Final[tuple[str, ...]] = ("io",)

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
    "Adapter": "serving.context",
    "EngineBootError": "serving.engine_runtime",
    "EngineHandle": "serving.engine_runtime",
    "EngineSpec": "serving.engine_runtime",
    "LlamaServer": "serving.engine_runtime",
    "VllmServer": "serving.engine_runtime",
    "Resources": "api.resources",
    "DistillationAdapter": "serving.context",
    # Incremental output: ctx.emit(chunk) rides the droppable JobProgress lane; the entrypoint's return value rides the authoritative channel — two wire channels, both named by the declaration.
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
        from .v1_deleted import REPLACEMENTS, refuse

        if name in REPLACEMENTS:
            raise refuse(name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return sorted((*_EXPORTS, *_SUBMODULES))


__all__ = [
    "Adapter",
    "DistillationAdapter",
    "ImageFormat",
    "LoadContext",
    "Model",
    "Resources",
    "entrypoint",
    "Delta",
    "ItemDelta",
    "TokenDelta",
    "iter_transformers_text_deltas",
    "EngineBootError",
    "EngineHandle",
    "EngineSpec",
    "LlamaServer",
    "VllmServer",
    "LayoutRequirements",
    "RequirementTerms",
    "LayoutDeclarationError",
    "report_applied_attention",
    "report_applied_lane",
    "report_attention_backend",
    "for_request",
    "FetchedUrl",
    "fetch_bytes",
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
    "GenerationDefaults",
    "RequestContext",
    "JobContext",
    "TrainingMetric",
    "diffusers_step_callback",
    "ProcessStalledError",
    "run_process",
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
    "HubApiError",
    "HubError",
    "parse_hub_error",
    "raise_for_hub_error",
    "ChildRequest",
    "ChildCallError",
    "ChildCallRefusedError",
    "ChildCallTimeoutError",
    "ChildRequestCanceledError",
    "ChildRequestFailedError",
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
