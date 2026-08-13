"""The rig's declaration registration must survive a registry reset.

This closes a recurrent CI flake: `test_graph_witness_pgw1031` errors at setup
with `AttributeError: 'NoneType' object has no attribute 'targets'`, from
`aot_mint.export_declaration(family)` returning `None`, while the SAME commit
passes on a plain re-run.

THE MECHANISM:

    [1] after first compile_cell : True
    [2] after reset              : False
    [3] after SECOND compile_cell: False     <- the flake

`rig_vehicles`' declaration registration is an import SIDE EFFECT
(`register_export_declaration` at module scope), so once the module sits in
`sys.modules` every later import is a no-op. Dozens of test modules call
`reset_export_declarations()` in their fixtures. Once both have happened in one
xdist worker, `compile_cell()` still returns a cell — it just names a family
nothing has declared any more, and the failure surfaces later and elsewhere.
Whether the two land in the same worker is a function of shard split, which is
why adding five unrelated test rows to a PR could summon it.

It is the same defect class the arm-state fence exists for: a
process-global registry fed by a convention the caller has to remember. The fix
is the same shape — ask the registry, do not trust the convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gen_worker.api.export_contract import (
    export_declaration, reset_export_declarations)

# Self-sufficient on purpose: depending on a sibling module to have put this on
# `sys.path` would make this row order-dependent, which is the exact property
# it exists to remove.
MICRO_SRC = Path(__file__).resolve().parent.parent / "examples" / "micro-diffusion" / "src"
if str(MICRO_SRC) not in sys.path:
    sys.path.insert(0, str(MICRO_SRC))

PAIR = ("micro-pad32", "micro-pad32-branchy")


@pytest.mark.parametrize("family", PAIR)
def test_the_rig_re_registers_after_a_reset_wipes_the_registry(family: str) -> None:
    """RED before the fix at step [3]: the import is a `sys.modules` no-op, so
    nothing re-registers and the cell names an undeclared family."""
    from harness import rig_vehicles

    veh = rig_vehicles.vehicle(family)

    assert veh.compile_cell() is not None
    assert export_declaration(family) is not None, "the first build must register"

    # Exactly what another test module's fixture does between two of ours.
    reset_export_declarations()
    assert export_declaration(family) is None, "the reset must really empty it"

    cell = veh.compile_cell()
    assert export_declaration(family) is not None, (
        "the rig must RE-register: an import that is a no-op cannot restore a "
        "registration a reset removed, and the cell it returns would name a "
        "family nothing has declared")
    assert cell.family == family


def test_the_cell_still_carries_the_declared_contract_after_a_reload() -> None:
    """The heal must not quietly change what it builds — a reloaded module has
    to produce a byte-identical cell, or the fix would trade a flake for a
    silent identity drift (the declaration feeds the cell key)."""
    from harness import rig_vehicles

    veh = rig_vehicles.vehicle("micro-pad32")
    before = veh.compile_cell()
    reset_export_declarations()
    after = veh.compile_cell()

    assert after.shapes == before.shapes
    assert after.text_len == before.text_len
    assert after.targets == before.targets
    assert after.family == before.family
