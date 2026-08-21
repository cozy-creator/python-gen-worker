from __future__ import annotations

import msgspec

from gen_worker import RequestContext, entrypoint


class TransformInput(msgspec.Struct, forbid_unknown_fields=True):
    text: str
    upper: bool = False
    repeat: int = 1


class TransformOutput(msgspec.Struct):
    text: str
    length: int


class ClosureGateInput(msgspec.Struct, forbid_unknown_fields=True):
    values: list[float]
    threshold: float = 0.5


class ClosureGateOutput(msgspec.Struct):
    passed: bool
    above: int


@entrypoint
def transform(ctx: RequestContext, payload: TransformInput) -> TransformOutput:
    text = (payload.text.upper() if payload.upper else payload.text) * payload.repeat
    ctx.warn(f"transformed {len(payload.text)} chars")
    return TransformOutput(text=text, length=len(text))


@entrypoint
def closure_gate(ctx: RequestContext, payload: ClosureGateInput) -> ClosureGateOutput:
    above = sum(1 for value in payload.values if value >= payload.threshold)
    return ClosureGateOutput(passed=above > 0, above=above)
