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
from gen_worker.families.base import GenerationDefaults, family


def _slot_weight_bytes(slot_dir: object) -> str:
    """What the slot's weights file HOLDS, on any tree shape.

    A harness endpoint is the closest thing this repo has to a pgw#1303 author
    slot: it is handed a DIRECTORY and reads raw weight bytes out of it. After
    pgw#1308 step ⑥ that directory is a projected tree, so the file at the
    path is a ~128 B pointer stub — which is exactly what a real third-party
    loader gets, and exactly why #1303 is a ruling rather than a cleanup.

    So these fixtures go through the SAME seam a gated production site now
    goes through (`models.materialized_view.third_party_dir`), which is what
    makes them evidence about the gate instead of a workaround for it.
    """

    from gen_worker.models.materialized_view import third_party_dir

    real = third_party_dir(Path(str(slot_dir)), why="harness author slot")
    return (real / "model.safetensors").read_text()


@family("harness-slot-only-testfam")
class _ToyDefaults(GenerationDefaults, frozen=True):
    steps: int = 7


class EchoIn(msgspec.Struct):
    text: str = ""


class EchoOut(msgspec.Struct):
    response: str


BOOT_UNREACHABLE_PIPELINE = Hub("harness/boot-precedence-pipeline", release="prod")
BOOT_UNREACHABLE_VAE = Hub("harness/boot-precedence-vae", release="prod")


@endpoint(models={
    "pipeline": Slot(str, default_checkpoint=BOOT_UNREACHABLE_PIPELINE),
    "vae": Slot(str, default_checkpoint=BOOT_UNREACHABLE_VAE),
})
class SlotBootPrecedenceEndpoint:
    def setup(self, pipeline: str, vae: str) -> None:
        self.pipeline_path = pipeline
        self.vae_path = vae

    def slot_boot_precedence(self, ctx: RequestContext[_ToyDefaults], data: EchoIn) -> EchoOut:
        return EchoOut(response=_slot_weight_bytes(self.pipeline_path))
