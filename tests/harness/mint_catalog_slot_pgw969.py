"""pgw#969: the sdxl SHAPE — a hub-CATALOG slot with no code default.

``sdxl/main.py`` declares::

    models={"pipeline": Slot(StableDiffusionXLPipeline, selected_by="model")}

No ``default_checkpoint=``. Every checkpoint it serves is a hub-catalog row,
so the decorator declares no ref at all and ``spec.models`` is EMPTY until the
parent binds it from the hub's pick.

pgw#828's harness endpoint (``WarmSlotEndpoint``) carries
``default_checkpoint=WARM_SLOT_PIPELINE``, which is why its regression test
was green on a shape that cannot exhibit this defect: a declared ref survives
rediscovery in the mint child, a catalog pick does not.

Two handlers on ONE class, like sdxl's ``generate``/``generate-turbo``: the
warm plan is class-scoped (pgw#654), and ``instance_key`` is a live property
over ``spec.models`` — so a fix that bound the invoked function alone would
move its key out from under its sibling and silently narrow the mint.

This module is deliberately isolated (the ``toy_endpoints_slot_only``
precedent): the mint child walks ``request.modules`` and re-runs discovery
over it, so what else lives beside it is part of the test.

pgw#1333: ``catalog_generate`` now declares sdxl's EXACT serving contract
(``objectives=("epsilon", "v_prediction"), distilled=False``). Without it this
module was the same trap it was written to escape one axis over — a catalog
slot whose function declared nothing, so the objective backstop never armed
and the whole "the child re-derives the facts from nothing" class was green by
fixture. Its sibling deliberately declares NOTHING, because the unrestricted
arm has to keep working on the same class.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import msgspec

from gen_worker import RequestContext, Resources, Slot, endpoint, worker_function
from gen_worker.api.binding import wire_ref
from gen_worker.families.base import GenerationDefaults, register_family


class CatalogDefaults(GenerationDefaults, frozen=True):
    steps: int = 5


register_family("harness-pgw969-catalog", CatalogDefaults)


class CatalogIn(msgspec.Struct):
    text: str = ""
    #: the ``selected_by="model"`` branch field
    model: str = ""


class CatalogOut(msgspec.Struct):
    response: str


#: One entry per handler call, base and turbo alike: the wire ref the
#: RESOLVED slot carried. The parent's serving path and the mint child's warm
#: forward both write here, which is what makes "the same checkpoint" an
#: assertion rather than a claim.
RESOLVED_REFS: List[str] = []


def _ref_of(ctx: RequestContext[CatalogDefaults]) -> str:
    wire = wire_ref(ctx.slots["pipeline"].ref)
    RESOLVED_REFS.append(wire)
    return wire


@endpoint(
    models={"pipeline": Slot(str, selected_by="model")},
    resources=Resources(gpu=True),
)
class CatalogSlotEndpoint:
    def setup(self, pipeline: str) -> None:
        self.pipeline_path = pipeline

    # sdxl `generate`'s declaration, character for character.
    @worker_function(objectives=("epsilon", "v_prediction"), distilled=False)
    def catalog_generate(
        self, ctx: RequestContext[CatalogDefaults], data: CatalogIn,
    ) -> CatalogOut:
        wire = _ref_of(ctx)
        weights = Path(self.pipeline_path) / "weights.txt"
        return CatalogOut(response=f"{wire}|{weights.read_text()}")

    # The sibling declares NOTHING: the unrestricted arm must survive on a
    # class whose other handler is governed.
    def catalog_generate_turbo(
        self, ctx: RequestContext[CatalogDefaults], data: CatalogIn,
    ) -> CatalogOut:
        return CatalogOut(response=f"turbo|{_ref_of(ctx)}")


#: The catalog row `harness/catalog-pick` stands for: `wai-illustrious`'s real
#: stamp, read three ways off the standing hub in pgw#1333's filing.
CATALOG_OBJECTIVE = "epsilon"
CATALOG_DISTILLED_STATUS = "classified"
