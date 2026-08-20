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

THE RULE THIS TABLE ENFORCES, and it is a rule about MANIFEST FIELDS too
(pgw#1580's enumeration audit). Every field the hub DECODES may only be dropped
from the author surface together with a row here naming its successor. The
audit checked ``manifestFunction`` field by field against what
``discovery/entrypoints_v2.py`` emits and the partition was exact:

* every field with a row here — ``compile``/``compile_axes``, ``variant_of``,
  ``objectives``, ``runtime_formula``, ``config_params``,
  ``accepts_references`` — was a DECISION, and the successor holds;
* every field WITHOUT one was an ACCIDENT. There were three, all P0, all found
  by an endpoint breaking on them rather than by anything failing at publish:
  ``child_calls`` (pgw#1579 — the hub mints ``invoke_child`` only when it is
  set, so a workflow endpoint failed at its FIRST child call),
  ``incremental_output``/``delta_output_schema`` (pgw#1576 — emitted, but
  hardcoded ``False``), and ``expected_outputs`` (pgw#1580 — LIVE in
  production: every endpoint promoted to v2 silently stopped telling the
  platform what it was about to produce). ``handles`` was the fourth, ruled
  RESTORED rather than retired on the principle that capabilities and
  behavioral divergence are DECLARED, never inferred from code shape.

So: a hardcut that removes an author-side spelling owes either a row in
:data:`REPLACEMENTS` or a hub change that stops reading the field. Silence is
the one option that is not available, and
``tests/test_manifest_declarations_pgw1579_1580.py`` is the fence that says so.
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
    # pgw#1406 built the successor this line said did not exist. The three
    # `@job` kwargs the producer plane actually uses carry over VERBATIM, and
    # the one RequestContext carries the publisher surface, so the port is a
    # re-decoration — which is what the 27 conversion producers in
    # `cozy-creator/jobs` need to hear at the refusal (th#2173, jobs#297).
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
    # pgw#1576: these three were deleted BY OMISSION — no successor line, so an
    # author hit a bare ImportError and the gap read as an oversight rather
    # than a ruling. It was an oversight, and the successor exists now. The
    # third name is NOT here: `iter_transformers_text_deltas` came back
    # verbatim, same spelling, so it imports instead of refusing.
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
