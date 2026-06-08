# source-hash bust: 2026-06-08T01:00Z (#447: add marco_polo_slow ~15s async-I/O fn for failover-at-scale e2e)
import asyncio
import time

import msgspec
from gen_worker import RequestContext, ValidationError, inference, invocable


class MarcoPoloInput(msgspec.Struct):
    text: str = ""


class MarcoPoloOutput(msgspec.Struct):
    response: str


@inference()
class MarcoPolo:
    def setup(self) -> None:
        pass

    @invocable(name="marco_polo")
    def marco_polo(self, ctx: RequestContext, data: MarcoPoloInput) -> MarcoPoloOutput:
        """Returns 'polo' when input is 'marco'; otherwise raises so the request fails."""
        # Deterministic minimal handler used for latency tests and for
        # exercising both billing branches: the marco->polo path succeeds
        # (capture), any other input raises ValidationError so the worker
        # reports a non-retryable FAILED job (hold release).
        ctx.raise_if_canceled()
        time.sleep(0.3)  # sustain concurrency for the load test

        if str(data.text or "").strip().lower() == "marco":
            return MarcoPoloOutput(response="polo")

        raise ValidationError(f"expected 'marco', got {data.text!r}")

    @invocable(name="marco_polo_slow")
    async def marco_polo_slow(
        self, ctx: RequestContext, data: MarcoPoloInput
    ) -> MarcoPoloOutput:
        """Same marco->polo semantics as marco_polo, but each call takes ~15s
        using REAL asyncio I/O waits so a large backlog can be held in-flight
        while an orchestrator replica is killed (#447 failover-at-scale).

        The work is a loop of ``await asyncio.sleep(...)`` on the worker's shared
        asyncio loop. asyncio.sleep is a genuine non-blocking wait that RELEASES
        the GIL and yields the loop — so many concurrent calls overlap on the
        single loop instead of serializing, which is exactly what lets one worker
        hold hundreds of these in flight at once. We deliberately avoid a
        third-party HTTP client (httpx/requests) so the function needs ZERO extra
        dependencies in the locked tenant image — the wait is the I/O.

        Wall-time is ~15s (100 ticks * 150ms), with a cancel check between ticks
        so an interrupt during failover is honored promptly.
        """
        ctx.raise_if_canceled()

        tick = 0.15
        ticks = 100  # 100 * 0.15s = 15s
        for i in range(ticks):
            if i % 10 == 0:
                ctx.raise_if_canceled()
            await asyncio.sleep(tick)

        if str(data.text or "").strip().lower() == "marco":
            return MarcoPoloOutput(response="polo")

        raise ValidationError(f"expected 'marco', got {data.text!r}")
# bust 1780099200
