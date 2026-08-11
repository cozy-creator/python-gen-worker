"""The declaration a blocked family owns — importable WITHOUT detonating.

Split from :mod:`harness.blocked_declaration` (which raises at module scope,
the shape a declaration FILE could still fail in) so tests can build the same
real declaration to prove the family is BLOCKED, not malformed.
"""

from __future__ import annotations

from typing import Tuple

from gen_worker import Compile, Dim, GraphClass, Input, MintBlocker

FAMILY = "harness-blocked-family"

#: The blocker every refusal on this family must reproduce, verbatim.
BLOCKER = MintBlocker(
    id="B1-harness",
    what="the traced module's list container is not pytree-representable",
    evidence="harness fixture standing in for a real boundary-shape blocker.",
    resolves_when="the boundary is shrunk to tensors",
)

#: The sentence a declaration MODULE that refuses at import raises with. Only
#: :mod:`harness.blocked_declaration` uses it — a MINT refusal is data now
#: (:data:`BLOCKER`), and this is the other, still-real failure: a declaration
#: file whose module scope throws for any reason at all.
BLOCKER_TEXT = (
    f"family {FAMILY!r} has 1 UNRESOLVED mint blocker(s); the declaration is "
    f"complete and validated but refuses to mint:\n"
    f"  - {BLOCKER.id}: {BLOCKER.what}\n    RESOLVES WHEN: "
    f"{BLOCKER.resolves_when}"
)


def build_declaration(
    *, family: str = FAMILY, blockers: Tuple[MintBlocker, ...] = (),
) -> Compile:
    """A real, valid declaration in the pgw#739 vocabulary."""
    return Compile(
        family=family,
        targets=("transformer",),
        text_len=77,
        shapes=((1024, 1024),),
        dims=(Dim("B", carried_by=(("hidden_states", 0),)),),
        classes=(GraphClass(dims={"B": 1}),),
        inputs=(Input("hidden_states", shape=("B", 4, 128, 128),
                      dtype="bfloat16"),),
        shape_strategy="static-rows",
        warm_changes_key=False,
        blockers=blockers,
    )
