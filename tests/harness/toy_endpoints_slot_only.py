"""Isolated single-endpoint module for the pgw#606/th#938
``_boot_setup_watch`` pin (test_p3_slot_binding_precedence.py).

``_boot_setup_watch`` is ONE task shared across every function awaiting hub
delivery — loading it alongside ``harness.toy_endpoints``'s Hub()-bound
``model-echo`` (which legitimately awaits hub delivery and legitimately
spawns that task) would make the assertion meaningless. This module carries
only the Slot fn under test, so the watch's absence is attributable to it.
"""

from __future__ import annotations

from pathlib import Path

import msgspec

from gen_worker import Hub, RequestContext, Slot, endpoint
from gen_worker.families.base import FamilyDefaults, family


@family("harness-slot-only-testfam")
class _ToyDefaults(FamilyDefaults, frozen=True):
    steps: int = 7


class EchoIn(msgspec.Struct):
    text: str = ""


class EchoOut(msgspec.Struct):
    response: str


BOOT_UNREACHABLE_PIPELINE = Hub("harness/boot-precedence-pipeline", tag="prod")
BOOT_UNREACHABLE_VAE = Hub("harness/boot-precedence-vae", tag="prod")


@endpoint(models={
    "pipeline": Slot(str, default_checkpoint=BOOT_UNREACHABLE_PIPELINE, default_config=_ToyDefaults()),
    "vae": Slot(str, default_checkpoint=BOOT_UNREACHABLE_VAE, default_config=_ToyDefaults()),
})
class SlotBootPrecedenceEndpoint:
    def setup(self, pipeline: str, vae: str) -> None:
        self.pipeline_path = pipeline
        self.vae_path = vae

    def slot_boot_precedence(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        weights = Path(self.pipeline_path) / "model.safetensors"
        return EchoOut(response=weights.read_text())
