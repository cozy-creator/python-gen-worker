# source-hash bust: 2026-07-03 (#368: @endpoint API rewrite)
import asyncio
import time
from typing import AsyncIterator

import msgspec
from gen_worker import RequestContext, ValidationError, endpoint


class MarcoPoloInput(msgspec.Struct):
    text: str = ""


class MarcoPoloOutput(msgspec.Struct):
    response: str


@endpoint
class MarcoPolo:
    def marco_polo(self, ctx: RequestContext, data: MarcoPoloInput) -> MarcoPoloOutput:
        """Returns 'polo' when input is 'marco'; otherwise raises so the request fails."""
        # Deterministic minimal handler used for latency tests and for
        # exercising both billing branches: the marco->polo path succeeds
        # (capture), any other input raises ValidationError so the worker
        # reports a non-retryable FAILED job (hold release).
        ctx.raise_if_cancelled()
        time.sleep(0.3)  # sustain concurrency for the load test

        if str(data.text or "").strip().lower() == "marco":
            return MarcoPoloOutput(response="polo")

        raise ValidationError(f"expected 'marco', got {data.text!r}")

    async def marco_polo_slow(
        self, ctx: RequestContext, data: MarcoPoloInput
    ) -> MarcoPoloOutput:
        """Same marco->polo semantics, but ~15s of REAL asyncio waits so a
        large backlog can be held in-flight on one worker (#447)."""
        ctx.raise_if_cancelled()

        tick = 0.15
        for i in range(100):  # 100 * 0.15s = 15s
            if i % 10 == 0:
                ctx.raise_if_cancelled()
            await asyncio.sleep(tick)

        if str(data.text or "").strip().lower() == "marco":
            return MarcoPoloOutput(response="polo")

        raise ValidationError(f"expected 'marco', got {data.text!r}")

    async def marco_polo_stream(
        self, ctx: RequestContext, data: MarcoPoloInput
    ) -> AsyncIterator[MarcoPoloOutput]:
        """Streams 'p', 'po', 'pol', 'polo' as four progress chunks."""
        if str(data.text or "").strip().lower() != "marco":
            raise ValidationError(f"expected 'marco', got {data.text!r}")
        word = "polo"
        for i in range(1, len(word) + 1):
            ctx.raise_if_cancelled()
            await asyncio.sleep(0.05)
            yield MarcoPoloOutput(response=word[:i])
