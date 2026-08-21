from __future__ import annotations

from typing import Final

MIGRATION: Final[str] = (
    "v1 SDK deleted, migrate to Model/@entrypoint, see se#757"
)


class V1SdkDeleted(ImportError):
    """A deleted v1 SDK name was imported or discovered."""


REPLACEMENTS: Final[dict[str, str]] = {
    "endpoint": "@entrypoint on a module-level function (gen_worker.entrypoint)",
    "job": (
        "@entrypoint(publishes=…, env=…, emits_media=…) — the same kwargs, "
        "and ctx: RequestContext carries mktemp/source/save_checkpoint; "
        "see pgw#1406"
    ),
    "worker_function": "@entrypoint",
    "variant_of": "one @entrypoint per variant; the request envelope picks",
    "Slot": "a Model subclass annotation on the @entrypoint parameter",
    "ResolvedSlot": "the model instance passed into the @entrypoint parameter",
    "resolve_slot": "the serve loop's residency lease (gen_worker.serving)",
    "OBJECTIVES": "checkpoint metadata; no author-side vocabulary",
    "ObjectiveMismatchError": "checkpoint metadata; no author-side vocabulary",
    "ConfigParam": "typed fields on the payload msgspec.Struct",
    "Compile": "ctx.compile marking inside Model.load",
    "CompileAxis": "the lane contract (tensorfs.contracts) on Model[T]",
    "AxisClass": "the lane contract (tensorfs.contracts) on Model[T]",
    "DynamicDim": "the lane contract (tensorfs.contracts) on Model[T]",
    "NoWarmup": "no successor — warmup is not an author declaration",
    "AcceptsReferences": "typed asset fields on the payload msgspec.Struct",
    "RuntimeFormula": "no successor — the catalog it fed is deleted",
    "GraphClass": "the publish-time instrumented derive (gen-worker release derive)",
    "Dim": "the publish-time instrumented derive (gen-worker release derive)",
    "Fork": "the publish-time instrumented derive (gen-worker release derive)",
    "Input": "the publish-time instrumented derive (gen-worker release derive)",
    "Arg": "the publish-time instrumented derive (gen-worker release derive)",
    "MintBlocker": "the publish-time instrumented derive (gen-worker release derive)",
    "register_export_declaration": "the publish-time instrumented derive",
    "import_export_declaration": "the publish-time instrumented derive",
    "DeclarationMismatch": "the publish-time instrumented derive",
    "assert_blockers": "the publish-time instrumented derive",
    "assert_faithful": "the publish-time instrumented derive",
    "cfg_image_classes": "the publish-time instrumented derive",
    "class_set_delta": "the publish-time instrumented derive",
    "contract_delta": "the publish-time instrumented derive",
    "override_delta": "the publish-time instrumented derive",
    "IncrementalTokenDelta": (
        "gen_worker.TokenDelta with @entrypoint(streams=TokenDelta) and "
        "ctx.emit(...) — the entrypoint still RETURNS its terminal struct "
        "(pgw#1576)"
    ),
    "BatchItemDelta": (
        "gen_worker.ItemDelta with @entrypoint(streams=ItemDelta) and "
        "ctx.emit(...); the binary `chunk` field is gone (msgpack framing "
        "could not survive the hub's JSON SSE surface) — text rides `text`, "
        "and the whole batch rides the returned struct (pgw#1576)"
    ),
    "TokenUsage": (
        "typed fields on the entrypoint's own returned struct — the terminal "
        "is the author's, so the platform folds nothing (pgw#1576)"
    ),
    "StreamResult": (
        "the entrypoint's own returned struct: a streaming entrypoint returns "
        "its terminal exactly like every other one (pgw#1576)"
    ),
    "StreamItem": "a field on the entrypoint's own returned struct (pgw#1576)",
    "Done": (
        "no successor — the stream ends when the body returns, and JobResult "
        "is the terminal (pgw#1576)"
    ),
    "Error": (
        "raise a typed gen_worker error; a failed ITEM in a batch is "
        "ItemDelta(error=...) plus the terminal struct (pgw#1576)"
    ),
    "GenerationDefaults": "gen_worker.models — the model type's Defaults struct",
    "StringEnum": "stdlib enum.StrEnum",
    "arm_compile": "ctx.compile inside Model.load",
    "report_applied_lane": "the lane is the Model's declared contract",
    "report_applied_attention": "the lane is the Model's declared contract",
    "report_attention_backend": "the lane is the Model's declared contract",
}


def refuse(name: str, *, context: str = "") -> V1SdkDeleted:
    """The typed refusal for one deleted name."""
    replacement = REPLACEMENTS.get(name)
    where = f" ({context})" if context else ""
    detail = f" Use {replacement}." if replacement else ""
    return V1SdkDeleted(
        f"gen_worker.{name} was deleted{where}: {MIGRATION}.{detail}"
    )


def refuse_module(module: str, attribute: str) -> V1SdkDeleted:
    """The typed refusal for a module that carried a v1 declaration."""
    return V1SdkDeleted(
        f"{module} carries {attribute!r} — a v1 @endpoint/@job declaration. "
        f"{MIGRATION}."
    )


__all__ = [
    "MIGRATION",
    "REPLACEMENTS",
    "V1SdkDeleted",
    "refuse",
    "refuse_module",
]
