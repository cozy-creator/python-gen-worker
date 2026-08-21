from __future__ import annotations

import msgspec

from gen_worker import ItemDelta, RequestContext, TokenDelta, entrypoint


class CompletionInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    max_tokens: int = 4


class Completion(msgspec.Struct):
    text: str
    tokens: int


class CaptionInput(msgspec.Struct, forbid_unknown_fields=True):
    items: list[str]


class CaptionOutput(msgspec.Struct):
    captions: list[str]
    failed: int


class SilentOutput(msgspec.Struct):
    ok: bool


@entrypoint(streams=TokenDelta)
async def complete(ctx: RequestContext, payload: CompletionInput) -> Completion:
    """Tokens live, the whole completion returned — both, from one body."""
    parts: list[str] = []
    for index in range(int(payload.max_tokens)):
        ctx.raise_if_cancelled()
        token = f"{payload.prompt}-{index} "
        parts.append(token)
        ctx.emit(TokenDelta(text=token))
        ctx.progress((index + 1) / payload.max_tokens, "decode", step=index + 1,
                     total=int(payload.max_tokens))
    return Completion(text="".join(parts), tokens=len(parts))


@entrypoint(streams=ItemDelta)
def caption(ctx: RequestContext, payload: CaptionInput) -> CaptionOutput:
    """A SYNC body streams with the identical call — no async anywhere."""
    total = len(payload.items)
    captions: list[str] = []
    failed = 0
    for index, item in enumerate(payload.items):
        if not item:
            failed += 1
            ctx.emit(ItemDelta(index=index, total=total, item_id=f"item-{index}",
                               error="blank item", finished=True))
            captions.append("")
            continue
        caption_text = f"a photo of {item}"
        ctx.emit(ItemDelta(index=index, total=total, item_id=f"item-{index}",
                           text=caption_text, finished=True))
        captions.append(caption_text)
    return CaptionOutput(captions=captions, failed=failed)


@entrypoint(streams=(TokenDelta, ItemDelta))
def narrate(ctx: RequestContext, payload: CaptionInput) -> CaptionOutput:
    """ONE handler, BOTH shapes: tokens while an item decodes, an item frame when it finishes."""
    total = len(payload.items)
    captions: list[str] = []
    for index, item in enumerate(payload.items):
        for word in item.split():
            ctx.emit(TokenDelta(text=word + " "))
        ctx.emit(ItemDelta(index=index, total=total, item_id=f"item-{index}",
                           text=item, finished=True))
        captions.append(item)
    return CaptionOutput(captions=captions, failed=0)


@entrypoint
def silent(ctx: RequestContext, payload: CaptionInput) -> SilentOutput:
    """Declares no chunk type; `emit` here must refuse rather than stream past a manifest that says `incremental_output: false`."""
    ctx.emit(TokenDelta(text="never"))
    return SilentOutput(ok=True)
