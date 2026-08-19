"""The hub-double's default endpoint module — a WEIGHTLESS v2 surface.

pgw#1373 deleted the v1 file of this name (442 lines of ``@endpoint`` classes,
``Slot``, ``ConfigParam``, streaming and catalog fixtures) along with the SDK it
declared against. What it did NOT delete was ``hub_double``'s default
``modules=("harness.toy_endpoints",)``, so every test that stood a real
``Worker`` up without naming its own module failed at boot with
``ModuleNotFoundError`` — a harness default naming a module that does not exist
is a trap for every test written after it, not just the ones that were already
there (pgw#1438).

This is deliberately NOT a port of the old file. The v1 fixtures existed to
exercise the v1 executor's dispatch, GPU-slot yield, streaming and catalog
paths; those are the surfaces the hardcut removed, so porting them would mean
writing fixtures for machinery no longer under test. What survives is the
question "does a real worker boot, register, dispatch and SAVE" — three
functions' worth — and every consumer that needs more declares its own module.

WEIGHTLESS (pgw#1392: zero model slots is a valid declaration): the hub-double
tests measure the TRANSPORT, and a model slot would drag a checkpoint download
into a test about whether an event reached a socket.
"""

from __future__ import annotations

import msgspec

from gen_worker import RequestContext, ValidationError, entrypoint


class EchoIn(msgspec.Struct):
    text: str = ""


class EchoOut(msgspec.Struct):
    response: str


@entrypoint
def echo(ctx: RequestContext, data: EchoIn) -> EchoOut:
    """The plain dispatch contract: one payload in, one result out.

    Keeps the v1 fixture's marco/polo shape, including the refusal arm — a
    dispatch test that cannot show a REFUSAL reaching the client only proves
    the happy path.
    """
    ctx.raise_if_cancelled()
    if (data.text or "").strip().lower() == "marco":
        return EchoOut(response="polo")
    raise ValidationError(f"expected 'marco', got {data.text!r}")


@entrypoint(emits_media=True)
def save_a_large_file(ctx: RequestContext, data: EchoIn) -> EchoOut:
    """Write one asset through the REAL upload path.

    Over ``_SAVE_BYTES_INLINE_THRESHOLD`` (4 MiB) on purpose: under it a client
    that asked for ``Prefer: bytes=inline`` gets the bytes back inline and no
    upload happens at all, which would make an upload test pass without an
    upload. ``emits_media=True`` because the media grant is checked BEFORE a
    byte moves.
    """
    asset = ctx.save_bytes("out.bin", b"x" * (5 * 1024 * 1024))
    return EchoOut(response=asset.ref)
