"""pgw#658 (filed as pgw#654 in v0.58.2/v0.60.1 release notes — id race) fixture: a Slot endpoint whose setup outlasts the intent
registry's 2.0s unreported-wait grace — the shape of every real model
endpoint (multi-minute pipeline load), which no toy fixture had."""

from __future__ import annotations

import time

import msgspec

from gen_worker import Hub, RequestContext, Slot, endpoint
from gen_worker.families.base import GenerationDefaults, family


@family("harness-slowfam")
class _SlowDefaults(GenerationDefaults, frozen=True):
    steps: int = 7


class SlowIn(msgspec.Struct):
    text: str = ""
    model: str = ""


class SlowOut(msgspec.Struct):
    response: str


SLOW_DECLARED = Hub("harness/slow-pipeline", tag="prod")

# Longer than _UNREPORTED_WAIT_TIMEOUT_S=2.0 with margin, short enough for a
# unit test. Real endpoints take minutes here.
SETUP_SLEEP_S = 4.0


@endpoint(models={
    "pipeline": Slot(str, default_checkpoint=SLOW_DECLARED),
})
class SlowSlotEndpoint:
    def setup(self, pipeline: str) -> None:
        time.sleep(SETUP_SLEEP_S)
        self.pipeline_path = pipeline

    def slow_slot_echo(self, ctx: RequestContext, data: SlowIn) -> SlowOut:
        return SlowOut(response="ok")
