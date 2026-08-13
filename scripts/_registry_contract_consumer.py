"""A representative CONSUMER declaration module for the registry-contract gate.

the SDK ships registry MECHANISMS; every vocabulary is declared by the
endpoint that owns it. This module declares a synthetic family the way a real
endpoint does, across all six decorator/registration surfaces. The gate
(`check_registry_contract.py`) imports it via the documented
``load_declaration_module`` path and asserts every registration became visible.
"""

from __future__ import annotations

from gen_worker.api.decorators import Compile
from gen_worker.api.export_contract import (
    Dim,
    Fork,
    GraphClass,
    register_export_declaration,
)
from gen_worker.convert import (
    CIVITAI,
    ConversionCase,
    CorpusTensor,
    HintMatch,
    LayoutDeclaration,
    RenameRule,
    RepackageFamily,
    TopologyConversion,
    declare_foreign_family_map,
    register_layout,
    register_layout_conversion,
    register_repackage_family,
)
from gen_worker.models.tensor_layout_contract import (
    TOPOLOGY_COMFY_SPLITFILES,
    TOPOLOGY_DIFFUSERS_MULTIFILE,
)
from gen_worker.families import GenerationDefaults, family

FAMILY = "contractcheck"


@family(FAMILY)
class ContractCheckDefaults(GenerationDefaults, frozen=True):
    steps: int = 28
    guidance: float = 6.0


register_repackage_family(
    RepackageFamily(family=FAMILY, aliases=("cc",), alias_prefixes=("contractcheck-",))
)

register_layout(
    LayoutDeclaration(
        variant=FAMILY,
        family=FAMILY,
        order=10,
        hints=(HintMatch(any_tokens=(FAMILY,)),),
    )
)

declare_foreign_family_map(CIVITAI, {"ContractCheck 1.0": FAMILY})

# The SIXTH registry (§1.33). A topology edge is DATA — declared
# rename passes plus their inverse — and registration runs the round-trip
# admission proof over the declared corpus before the edge exists.
register_layout_conversion(
    TopologyConversion(
        from_id=TOPOLOGY_COMFY_SPLITFILES,
        to_id=TOPOLOGY_DIFFUSERS_MULTIFILE,
        version=1,
        rules=(RenameRule(
            kind="prefix", pairs=(("model.diffusion_model.", "transformer."),)),),
        inverse_rules=(RenameRule(
            kind="prefix", pairs=(("transformer.", "model.diffusion_model."),)),),
        corpus=(ConversionCase(
            name="contractcheck-dit",
            tensors={
                "model.diffusion_model.blocks.0.attn.to_q.weight":
                    CorpusTensor(dtype="BF16", shape=(4, 4)),
                "model.diffusion_model.blocks.0.attn.to_k.weight":
                    CorpusTensor(dtype="BF16", shape=(4, 4)),
            },
        ),),
        why="the registry-contract gate's representative topology edge",
    )
)

register_export_declaration(
    Compile(
        family=FAMILY,
        targets=("transformer",),
        text_len=512,
        dims=(
            Dim("H", carried_by=(("hidden_states", 2),), multiple_of=2),
            Dim("B", carried_by=(("hidden_states", 0),)),
        ),
        forks=(Fork("cfg", served=(False,), unserved=(True,),
                    reason="default_value"),),
        classes=(
            GraphClass(dims={"H": 90, "B": 1}, fork={"cfg": False}),
            GraphClass(dims={"H": 160, "B": 1}, fork={"cfg": False}),
        ),
        shape_strategy="dynamic-collapse",
        warm_changes_key=False,
    )
)
