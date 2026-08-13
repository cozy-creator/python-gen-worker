"""pgw#1230 — the ``format`` arm axis has ONE derivation.

THE DEFECT, and why nothing caught it. Two different facts shared the name
``ARTIFACT_FORMAT`` in two modules:

  * ``aot_serve.ARTIFACT_FORMAT``   — the AOT cell metadata schema. The CHILD
    stamps it into every artifact (``aot_serve.entry_metadata``).
  * ``compile_cache.ARTIFACT_FORMAT`` — the torch-inductor-cache PRODUCER
    format (gw#391), an ingredient of the JIT semantic cache tag and nothing
    to do with cell metadata.

``fleet_cells.arm_identity`` computed its ``format`` fact from the SECOND one
and compared it against the first. Both were 2, so it passed by coincidence
for a year. pgw#1176 moved the cell schema to 3 and left the inductor format
at 2, and from that commit **every freshly minted cell failed to arm** —
`key_axis_divergence: format: child cell states '3', this runtime computed
'2'` — compiled fine, quarantined, published nothing. Observed on Cell Zero
across all 4 z-image graph classes.

Nothing was red, for two separate reasons, and both are the point of this
file:

  1. Six test files monkeypatch ``arm_axis_divergence`` to return ``""``. The
     seam that catches this is stubbed out wherever it would have run.
  2. The one file that DOES call it (``test_handback_key_axes_pgw1042``)
     hand-wrote ``"format": "2"`` on BOTH sides — it encoded the parent's
     wrong constant instead of deriving from the producer, so it agreed with
     the bug and stayed green through the 2->3 bump.

So the rows here are deliberately built from the REAL derivations on both
sides — ``fleet_cells.arm_identity`` for the parent, ``aot_serve``'s own
producer for the child — because a double on either side is what hid a P0.

The arm check itself was NOT at fault and is not weakened: it fired, named
the axis, named both values, and refused loudly. That is why this was
diagnosable at all.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, Dict

import pytest

from gen_worker import aot_serve, cell_key, compile_cache as cc, env_seal
from gen_worker import fleet_cells

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "gen_worker"

RUNTIME_KEY = {"sm": "sm_89", "sku": "l4", "torch": "2.13.0", "cuda": "12.8",
               "image_digest": ""}
TOOLCHAIN = (("libtorch.so", "cafe0123cafe0123"),)


class _Cfg:
    family = "micro-diffusion"
    shapes = ((64, 64),)
    text_lens = (7,)
    guidance_scales = (1.0,)
    lora_bucket = 0


@pytest.fixture()
def one_process_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parent and child in ONE process tree: the same runtime facts feed both
    derivations, which is the premise the arm comparison is written against.

    Only the DEVICE probe is stubbed — a CPU test box has no sm, and the axis
    under test is not the device. Every axis this file rules on is computed by
    production code from these same inputs.
    """
    monkeypatch.setattr(cc, "runtime_key", lambda: dict(RUNTIME_KEY))
    monkeypatch.setattr(cc, "toolchain_digest", lambda: TOOLCHAIN)
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: dict(RUNTIME_KEY))


def _child_meta() -> Dict[str, Any]:
    """What the CHILD stamps, through the producer that owns the envelope."""
    from tests.harness import exported_cell

    meta = exported_cell.metadata()
    meta.update(RUNTIME_KEY)
    meta["family"] = _Cfg.family
    meta["weight_lane"] = ""
    meta["lora_bucket"] = 0
    meta["toolchain"] = dict(TOOLCHAIN)
    meta[env_seal.SEAL_KEY] = dict(env_seal.effective_seal())
    return meta


# ---------------------------------------------------------------------------
# 1. THE ROW. stamp == computed, both sides real.
# ---------------------------------------------------------------------------


def test_the_stamp_and_the_computed_format_are_the_same_value(
    one_process_tree: None,
) -> None:
    """The whole P0 in one assertion.

    Goes red on the tree as it stood before this fix: the parent computed
    ``'2'`` from ``compile_cache`` while the child stamped ``3`` from
    ``aot_serve``.
    """
    parent = fleet_cells.arm_identity(_Cfg.family, "", 0, _Cfg())
    stamped = aot_serve.entry_metadata(
        family=_Cfg.family, precision="w8a8", cell_key="", name="x",
        entry=_child_meta()[cell_key.ENTRY_BLOCK_KEY],
    )
    assert parent.facts_dict()["format"] == str(stamped["format"])


def test_a_freshly_minted_cell_arms_in_the_tree_that_minted_it(
    one_process_tree: None,
) -> None:
    """The seam, end to end, with NOTHING stubbed on it — the exact call that
    quarantined Cell Zero's four classes."""
    parent = fleet_cells.arm_identity(_Cfg.family, "", 0, _Cfg())
    divergence = fleet_cells.arm_axis_divergence(parent, _child_meta())
    assert divergence == "", divergence


# ---------------------------------------------------------------------------
# 2. The refusal still fires — the fix must not buy agreement by deleting it
# ---------------------------------------------------------------------------


def test_a_divergent_format_is_still_refused_by_name(
    one_process_tree: None,
) -> None:
    """Credit where due: this check is what made the outage diagnosable. The
    fix removes the DISAGREEMENT, never the detection — a cell of a foreign
    schema must still be refused loudly rather than armed hopefully."""
    parent = fleet_cells.arm_identity(_Cfg.family, "", 0, _Cfg())
    foreign = dict(_child_meta(), format=aot_serve.ARTIFACT_FORMAT + 1)
    got = fleet_cells.arm_axis_divergence(parent, foreign)
    assert got.startswith("format: ")
    assert str(aot_serve.ARTIFACT_FORMAT + 1) in got
    assert str(aot_serve.ARTIFACT_FORMAT) in got


# ---------------------------------------------------------------------------
# 3. The twin cannot come back
# ---------------------------------------------------------------------------


def test_exactly_one_module_defines_an_artifact_format(
    one_process_tree: None,
) -> None:
    """A second constant of this name in a second module is the defect itself,
    not a step towards it — the values agreeing today is precisely how it hid
    for a year. One definition, or this is red.
    """
    definers = sorted(
        path.name
        for path in SRC.rglob("*.py")
        if re.search(r"^ARTIFACT_FORMAT\s*=", path.read_text(), re.M)
    )
    assert definers == ["aot_serve.py"], definers


def test_the_inductor_cache_format_is_not_called_a_format_of_the_artifact(
    one_process_tree: None,
) -> None:
    """The renamed constant keeps its VALUE — the semantic cache tag is a
    published-cache identity and must not churn on a rename — while losing the
    name that let it be read as the cell schema."""
    assert cc.SEMANTIC_TAG_FORMAT == 2
    assert not hasattr(cc, "ARTIFACT_FORMAT")
    assert aot_serve.ARTIFACT_FORMAT == 3


def test_no_module_computes_the_format_axis_from_a_second_source(
    one_process_tree: None,
) -> None:
    """The arm axis is built in exactly one place, from the producer's symbol.

    Stated structurally because the failure mode was structural: the parent
    was not WRONG about the format, it was asking a different module.
    """
    text = (SRC / "fleet_cells.py").read_text()
    assert '"format": str(aot_serve.ARTIFACT_FORMAT)' in text
    # CODE only — the comment above that line names the old symbol on purpose,
    # because a reader who does not know what it was cannot see why this one
    # is spelled the way it is.
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "cc.ARTIFACT_FORMAT" in ln]
