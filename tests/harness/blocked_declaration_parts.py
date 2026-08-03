"""The declaration a blocked family owns — importable WITHOUT detonating.

Split from :mod:`harness.blocked_declaration` (which raises at module scope,
the way ltx/qwen/z-image do today) so tests can build the same real
declaration to prove the family is BLOCKED, not malformed.
"""

from __future__ import annotations

from gen_worker import Compile, Dim, GraphClass, Input

FAMILY = "harness-blocked-family"

#: The exact sentence that must survive from the blocker to the typed event.
BLOCKER_TEXT = (
    f"family {FAMILY!r} has 1 UNRESOLVED mint blocker(s); the declaration is "
    f"complete and validated but refuses to mint:\n"
    f"  - B1-harness: the traced module's list container is not pytree-"
    f"representable\n    RESOLVES WHEN: the boundary is shrunk to tensors"
)


def build_declaration() -> Compile:
    """A real, valid declaration in the pgw#739 vocabulary."""
    return Compile(
        family=FAMILY,
        targets=("transformer",),
        text_len=77,
        shapes=((1024, 1024),),
        dims=(Dim("B", carried_by=(("hidden_states", 0),)),),
        classes=(GraphClass(dims={"B": 1}),),
        inputs=(Input("hidden_states", shape=("B", 4, 128, 128),
                      dtype="bfloat16"),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    )
