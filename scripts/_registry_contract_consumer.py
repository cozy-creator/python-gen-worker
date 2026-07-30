"""A representative CONSUMER declaration module for the registry-contract gate.

pgw#740: the SDK ships registry MECHANISMS; every vocabulary is declared by the
endpoint that owns it. This module declares a synthetic family the way a real
endpoint does, across all five decorator/registration surfaces. The gate
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
    HintMatch,
    LayoutDeclaration,
    RepackageFamily,
    declare_foreign_family_map,
    register_layout,
    register_repackage_family,
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

register_export_declaration(
    Compile(
        family=FAMILY,
        targets=("transformer",),
        text_len=512,
        dims=(
            Dim("H", carried_by=(("hidden_states", 2),), multiple_of=2),
            Dim("B", carried_by=(("hidden_states", 0),)),
        ),
        forks=(Fork("cfg", served=(False,), unserved=(True,)),),
        classes=(
            GraphClass(dims={"H": 90, "B": 1}, fork={"cfg": False}),
            GraphClass(dims={"H": 160, "B": 1}, fork={"cfg": False}),
        ),
        shape_strategy="dynamic-collapse",
        warm_changes_key=False,
    )
)
