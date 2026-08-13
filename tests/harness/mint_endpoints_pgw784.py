"""Harness endpoints for pgw#784's liveness proof.

They exist to put a MINT on a real worker's real event loop, two ways, so the
same instrumentation can measure both:

* ``mint-out-of-process`` spawns the mint child exactly as ``mint_delegate``
  does and returns immediately — the shape pgw#784 ships;
* ``mint-in-process`` does the equivalent work on the worker's own loop — the
  shape th#1299 died of, kept reachable ONLY so the detector can be calibrated
  against a known-bad arm.

``eager-tick`` is the tenant traffic that has to keep completing throughout.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, List

import msgspec

from gen_worker import child_contract
from gen_worker import RequestContext, endpoint
from gen_worker import mint_process as mp

#: Set by the test before booting the worker.
WORKDIR = os.environ.get("PGW784_WORKDIR", "")

#: Recorded by the endpoints, read by the test after the run.
MINT_OUTCOMES: List[Any] = []
EAGER_COMPLETIONS: List[float] = []


class MintIn(msgspec.Struct):
    seconds: float = 1.0


class MintOut(msgspec.Struct):
    started: bool = True
    note: str = ""


class TickIn(msgspec.Struct):
    n: int = 0


class TickOut(msgspec.Struct):
    n: int = 0
    at: float = 0.0


def _request(workdir: Path) -> mp.MintRequest:
    workdir.mkdir(parents=True, exist_ok=True)
    return mp.MintRequest(
        function="gen", modules=("harness.toy_endpoints",),
        family="pgw784", arm_token="arm1-liveness",
        target=str(workdir / "cell.tar.gz"),
        work_root=str(workdir / "capture"),
        report=str(workdir / mp.REPORT_NAME),
        cfg=child_contract.CompileSpec(family="pgw784"),
    )


def _burn(seconds: float) -> int:
    """Long-running GIL-holding Python — a stand-in for inductor's
    orchestration layer, which is what actually starved the loop."""
    acc = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for i in range(200000):
            acc += i * i
    return acc


@endpoint
class MintLiveness:
    async def mint_out_of_process(
        self, ctx: RequestContext, payload: MintIn,
    ) -> MintOut:
        """Start the mint in its own OS process and return at once.

        This is the whole fix: the compile leaves this interpreter, so nothing
        it does can touch this loop's GIL, and the beat plus eager serving
        continue for the mint's entire duration.
        """
        workdir = Path(WORKDIR) / "oop"
        env = dict(os.environ)
        env["MINT_STUB_MODE"] = "busy"
        env["MINT_STUB_SECONDS"] = str(payload.seconds)

        async def _drive() -> None:
            MINT_OUTCOMES.append(await mp.run_mint(
                _request(workdir), workdir=workdir, env=env,
                observe_interval_s=1.0))

        asyncio.get_running_loop().create_task(_drive())
        return MintOut(note="child spawned")

    async def mint_in_process(
        self, ctx: RequestContext, payload: MintIn,
    ) -> MintOut:
        """th#1299's shape, for calibration only: the compile driver runs
        inside the serving process, on the loop."""
        _burn(payload.seconds)
        return MintOut(note="burned in-process")

    async def eager_tick(
        self, ctx: RequestContext, payload: TickIn,
    ) -> TickOut:
        """Tenant traffic. Trivial by design — its VALUE is the timestamp,
        which proves eager serving never stopped."""
        now = time.monotonic()
        EAGER_COMPLETIONS.append(now)
        return TickOut(n=payload.n, at=now)


def reset() -> None:
    MINT_OUTCOMES.clear()
    EAGER_COMPLETIONS.clear()
