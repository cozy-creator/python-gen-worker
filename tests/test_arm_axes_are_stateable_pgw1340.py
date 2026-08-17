"""pgw#1340 / th#2098 — the handback seam compares axes a REAL cell cannot state.

MEASURED IN PRODUCTION (standing `master` hub `a35434eb14`, 2026-08-17), and it
is the whole of th#2098: every `tensorhub/sd15` pod bought for a burst spent
~25 minutes of paid L4 on an `inductor_compile` and threw it away, ~$1.00 of
GPU per burst for zero artifacts against 57.7 GPU-seconds of actual inference.
The wire says exactly which axis, on every single row::

    aot_entry_arm_failed  key_axis_divergence
      family: child cell states '', this runtime computed 'sd15'

224 occurrences on `sd15`, 28 on `ernie`, wheels 0.117.0 and 0.118.0 — every
self-mint that reached the seam, zero exceptions, zero publishes.

WHY `family` IS EMPTY. Since pgw#1270 TCG mints every artifact, and
`torchcg.artifact.validate_metadata` refuses any metadata whose field set is
not EXACTLY its closed vocabulary::

    compiled_graph_format  kind  compiled_graph_key  graph_class
    sm  toolchain  host_isa  package_constants_in_so  constant_folding_fenced

`family` is not in it. Neither are `weight_lane`/`lora_bucket` (the `lane`
axis) or `env_seal`. `arm_axis_divergence` still read all four out of the
metadata dict, so THREE of its six compared axes were structurally unstateable
and the comparison could only ever fail — `family` first, because it sorts
first in `ARM_ENVIRONMENT_FACTS`.

THIS IS THE UNFINISHED HALF OF pgw#1277. That issue found the same class one
block over (the worker read an `entry` block TCG never writes) and recorded the
reason it survived CI verbatim: *"CI stayed green because every fixture built
the obsolete shape."* `tests/test_handback_key_axes_pgw1042.py::_envelope` is
that fixture — it hand-writes `family`, `weight_lane`, `lora_bucket` and
`env_seal` into a dict no packed cell can contain. So the fence here is not
another assertion about the same fixture; it is the fixture built by
`torchcg.artifact.build_metadata`, through `tests/tcg_artifacts.py`, which has
existed since pgw#1283.

The two halves this file pins:

1. a cell TCG actually minted ARMS — the axes compared at the seam are exactly
   the axes a cell can state, and a real one agrees with its own runtime;
2. the divergence is knowable for $0 BEFORE the compile.
   :func:`fleet_cells.unstateable_arm_axes` is a pure function of two
   compile-time constants, so a seam that can never agree is a refusal at
   obligation-open — not a 25-minute receipt.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from gen_worker import aot_serve, env_seal, fleet_cells, graph_facts
from gen_worker._vendor.torchcg import identity as tcg_identity

import tcg_artifacts


def _real_cell_metadata(**kw: Any) -> Dict[str, Any]:
    """Metadata from TCG's own builder — the only shape a cell can carry."""
    return tcg_artifacts.metadata(**kw)


def _runtime_arm_key(meta: Dict[str, Any], family: str = "sd15") -> fleet_cells.ArmIdentity:
    """The parent's obligation identity for the runtime that cell was minted on.

    Every OBLIGATION fact is stated exactly as production states it: the family
    the release declares, the serving lane, this machine's env seal. The
    ENVIRONMENT facts are taken from the cell's own metadata, because that is
    the claim under test — a cell minted on this runtime describes it.
    """
    facts = {
        "family": family,
        aot_serve.COMPILED_GRAPH_FORMAT_KEY: str(aot_serve.COMPILED_GRAPH_FORMAT),
        "lane": "bf16-w16a16",
        "sm": str(meta["sm"]),
        "env_seal": env_seal.seal_digest(env_seal.effective_seal()),
        "toolchain": tcg_identity.toolchain_axis_digest(dict(meta["toolchain"])),
        "subject": graph_facts.subject_digest(()),
        "targets": "unet",
        "dynamic": "{}",
        "regional": "0",
    }
    return fleet_cells.ArmIdentity(facts=tuple(sorted(facts.items())))


def test_a_cell_TCG_MINTED_arms_on_the_runtime_that_minted_it() -> None:
    """THE PRODUCTION RED. Before pgw#1340 this returned
    ``family: child cell states '', this runtime computed 'sd15'`` — the exact
    string 224 `sd15` rows carry — and $1.00/burst of L4 went to a compile
    that could never arm.
    """
    meta = _real_cell_metadata()
    assert fleet_cells.arm_axis_divergence(_runtime_arm_key(meta), meta) == ""


@pytest.mark.parametrize("axis", ["family", "lane", "env_seal"])
def test_an_axis_no_cell_can_state_is_not_compared_at_this_seam(axis: str) -> None:
    """INVERTED, and stated rather than merely dropped (pgw#939: absence is a
    verdict, never a skipped check).

    These three are OBLIGATION facts, and they keep doing their whole job on
    THIS side of the boundary: they are in the arm token, so they split the
    pending, the child, and the local store's memo row — two families or two
    lanes never share a mint. What they cannot do is cross a boundary a TCG
    artifact has no field for. This is pgw#1176's own ruling about `envelope`,
    applied to the axes that cut was not swept for.
    """
    assert axis not in fleet_cells.ARM_ENVIRONMENT_FACTS
    assert axis in fleet_cells.ARM_OBLIGATION_FACTS


def test_every_compared_axis_is_stateable_by_a_cell() -> None:
    """The $0 preflight, evaluated on the real constants.

    A non-empty answer here means the seam is unsatisfiable by construction:
    every mint under it compiles for 20-45 minutes and refuses. That must be
    a refusal before the child is spawned, never a discovery after the bill.
    """
    assert fleet_cells.unstateable_arm_axes() == ()


def test_the_preflight_names_an_axis_that_left_the_vocabulary() -> None:
    """The fence proves it can go RED — this is what would have caught pgw#1270
    on a laptop instead of on eight L4s.
    """
    got = fleet_cells.unstateable_arm_axes(("sm", "family", "env_seal"))
    assert got == ("family", "env_seal")


@pytest.mark.parametrize("field,value,axis", [
    ("sm", "sm_80", "sm"),
    ("toolchain", {"torch": "a-different-compiler"}, "toolchain"),
])
def test_a_REAL_divergence_is_still_refused_by_name(
    field: str, value: Any, axis: str,
) -> None:
    """The positive control. Narrowing the compared set must not disarm it:
    a cell minted on another card or another compiler is exactly what this
    seam exists to refuse, and it still is.
    """
    meta = _real_cell_metadata()
    arm = _runtime_arm_key(meta)
    diverged = dict(meta, **{field: value})
    got = fleet_cells.arm_axis_divergence(arm, diverged)
    assert got.startswith(f"{axis}: "), got


def test_the_compared_set_is_not_empty() -> None:
    """A comparison narrowed to nothing would pass every cell and read green.

    The three that survive are the runtime axes TCG keys the artifact ON, so
    they are the three that can genuinely refuse a foreign cell.
    """
    assert set(fleet_cells.ARM_ENVIRONMENT_FACTS) == {
        aot_serve.COMPILED_GRAPH_FORMAT_KEY, "sm", "toolchain"}
