"""The v1 SDK tombstone (pgw#1373).

Paul's hardcut ruling, 2026-08-18: *"you should hardcut the old SDK system
everywhere. no legacy support, no supporting both. It's fine if everything
breaks in the interim. Hardcut."*

The v1 author surface — ``@endpoint``, ``@job``, ``Slot``/``ResolvedSlot``,
``ConfigParam``, ``variant_of``, ``worker_function``, the 26-family model
catalog and its ``ModelSpec``/``GraphModelSpec`` declaration architecture, the
v1 ``functions[]``/``jobs[]`` manifest vocabulary and the boot-time keyset
ladder — is DELETED. ``Model[T]`` + ``@entrypoint`` is the only surface.

A deletion must not create a SILENT-ABSENT state (the tracker's typed-refusal
rule). A bare ``ImportError: cannot import name 'endpoint'`` across 27
unmigrated endpoint packages says nothing about what happened or where to go,
so every deleted name raises :class:`V1SdkDeleted` NAMING the migration —
here, at import, at the exact spelling the author wrote.
"""

from __future__ import annotations

from typing import Final

#: The one migration pointer. Every refusal in this module ends with it.
MIGRATION: Final[str] = (
    "v1 SDK deleted, migrate to Model/@entrypoint, see se#757"
)


class V1SdkDeleted(ImportError):
    """A deleted v1 SDK name was imported or discovered.

    An ``ImportError`` subclass deliberately: the failure IS an import
    failure, so ``except ImportError`` in a build wrapper still catches it,
    and the message it prints now names the migration instead of a symbol.
    """


#: Deleted name -> what replaces it. The whole v1 author surface, so a
#: refusal can be specific about the ONE name the author actually wrote.
REPLACEMENTS: Final[dict[str, str]] = {
    "endpoint": "@entrypoint on a module-level function (gen_worker.entrypoint)",
    "job": "@entrypoint — jobs have no v2 successor yet; see se#757",
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
    """The typed refusal for a module that carried a v1 declaration.

    Raised by discovery when it finds a v1 decorator's stamped attribute on
    an imported author module: the package still declares the old surface,
    which is a BUILD failure with a name attached, never an empty manifest.
    """
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
