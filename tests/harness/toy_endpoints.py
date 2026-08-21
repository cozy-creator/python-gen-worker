"""The hub-double's default endpoint module — a WEIGHTLESS v2 surface."""

from __future__ import annotations

import msgspec

from gen_worker import RequestContext, ValidationError, entrypoint


class EchoIn(msgspec.Struct):
    text: str = ""


class EchoOut(msgspec.Struct):
    response: str


@entrypoint
def echo(ctx: RequestContext, data: EchoIn) -> EchoOut:
    """The plain dispatch contract: one payload in, one result out."""
    ctx.raise_if_cancelled()
    if (data.text or "").strip().lower() == "marco":
        return EchoOut(response="polo")
    raise ValidationError(f"expected 'marco', got {data.text!r}")


@entrypoint(emits_media=True)
def save_a_large_file(ctx: RequestContext, data: EchoIn) -> EchoOut:
    """Write one asset through the REAL upload path."""
    asset = ctx.save_bytes("out.bin", b"x" * (5 * 1024 * 1024))
    return EchoOut(response=asset.ref)
