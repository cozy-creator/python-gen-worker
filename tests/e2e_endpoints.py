"""Endpoint module served by the fake-scheduler e2e suite (not a test file)."""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import msgspec

from gen_worker import RequestContext, ValidationError, inference, invocable


class EchoIn(msgspec.Struct):
    text: str = ""


class EchoOut(msgspec.Struct):
    response: str


@inference()
class E2EEndpoint:
    def setup(self) -> None:
        pass

    @invocable(name="echo")
    def echo(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        ctx.raise_if_canceled()
        if (data.text or "").strip().lower() == "marco":
            return EchoOut(response="polo")
        raise ValidationError(f"expected 'marco', got {data.text!r}")

    @invocable(name="stream3")
    async def stream3(self, ctx: RequestContext, data: EchoIn) -> AsyncIterator[EchoOut]:
        for i in range(3):
            ctx.raise_if_canceled()
            yield EchoOut(response=f"chunk-{i}")

    @invocable(name="slow")
    async def slow(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        await asyncio.sleep(30.0)
        return EchoOut(response="late")

    @invocable(name="sleepy")
    def sleepy(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        time.sleep(0.5)
        return EchoOut(response="done")
