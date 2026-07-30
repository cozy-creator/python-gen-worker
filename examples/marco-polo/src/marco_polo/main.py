# source-hash bust: 2026-07-03 (#368: @endpoint API rewrite)
import asyncio
import os
import time
from typing import AsyncIterator

import msgspec
from gen_worker import RequestContext, ValidationError, endpoint
from gen_worker.api.types import Asset


class MarcoPoloInput(msgspec.Struct):
    text: str = ""


class MarcoPoloOutput(msgspec.Struct):
    response: str


class MarcoAttachInput(msgspec.Struct):
    image: Asset
    text: str = ""
    delay_s: float = 0.0


class MarcoAttachOutput(msgspec.Struct):
    response: str
    blake3: str
    size_bytes: int
    local_path: str


@endpoint(
    # th#1087: an org can only attach envs the RELEASE declares. GEN_WORKER_OOM_PROBE
    # is genuinely this endpoint's own switch, so declaring it is correct.
    # GEN_WORKER_PROCESS_SPLIT is deliberately absent: pgw#763 delta 0 moved it into
    # the platform-reserved namespace, so declaring it now earns a reserved_name
    # refusal — a tenant cannot decline the boundary that contains it.
    env=["GEN_WORKER_OOM_PROBE"],
)
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

    async def marco_polo_wedge(
        self, ctx: RequestContext, data: MarcoPoloInput
    ) -> MarcoPoloOutput:
        """th#965 layer-2 liveness probe: an ASYNC handler that BLOCKS the
        event loop with a synchronous sleep — the control loop that owes
        heartbeats goes silent while the gRPC transport threads keep
        answering keepalive pings. A th#965 hub must fire
        worker_heartbeat_lost within its miss window and recycle the pod;
        this handler never returns inside any sane enforcement window."""
        time.sleep(1800)
        return MarcoPoloOutput(response="unreachable")

    def marco_polo_oom(
        self, ctx: RequestContext, data: MarcoPoloInput
    ) -> MarcoPoloOutput:
        """pgw#763 stage 4: the UNCATCHABLE death, caused for real.

        Allocates host RAM in 256MB touched chunks until the kernel's cgroup
        OOM killer takes this process. There is no exception, no finally, and
        no Python left to report it — which is the entire premise of the
        control/compute split: the surviving control parent is the only thing
        that can tell the hub which request died and keep the pod serving.

        te#138's shape without te#138's call site (layer 2's host-RAM move
        guard now refuses the oversized `.to("cpu")` typed, so a REAL cgroup
        kill has to be provoked directly). Only reachable when
        GEN_WORKER_OOM_PROBE=1 is set on the endpoint version, so no tenant
        traffic can trip it.
        """
        if os.environ.get("GEN_WORKER_OOM_PROBE", "").strip() != "1":
            raise ValidationError(
                "marco-polo-oom is a pgw#763 resilience probe; it is refused "
                "unless the endpoint version sets GEN_WORKER_OOM_PROBE=1")
        ctx.log("pgw#763 OOM probe: allocating until the cgroup killer fires")
        chunks = []
        for i in range(4096):          # 1 TiB ceiling: the cgroup lands first
            chunks.append(bytearray(256 * 1024 * 1024))
            chunks[-1][::4096] = b"\x01" * (len(chunks[-1]) // 4096)  # touch
            if i % 4 == 0:
                ctx.log(f"pgw#763 OOM probe: {(i + 1) * 256} MB resident")
        return MarcoPoloOutput(response="unreachable: the cgroup never fired")

    async def marco_polo_attach(
        self, ctx: RequestContext, data: MarcoAttachInput
    ) -> MarcoAttachOutput:
        """th#886 v4 private-input probe: echoes the materialized attachment's
        BLAKE3/size/attempt-local path so the harness can verify exact bytes
        and post-terminal temp cleanup. delay_s>0 holds the job in flight for
        disconnect/retry chaos."""
        import blake3

        if str(data.text or "").strip().lower() != "marco":
            raise ValidationError(f"expected 'marco', got {data.text!r}")
        path = os.fspath(data.image)
        with open(path, "rb") as f:
            raw = f.read()
        waited = 0.0
        while waited < float(data.delay_s or 0.0):
            ctx.raise_if_cancelled()
            await asyncio.sleep(0.15)
            waited += 0.15
        return MarcoAttachOutput(
            response="polo",
            blake3=blake3.blake3(raw).hexdigest(),
            size_bytes=len(raw),
            local_path=path,
        )

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
