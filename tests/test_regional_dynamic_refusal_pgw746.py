"""pgw#746 (OQ-7), as pgw#817/D4 RELOCATED it: the ``regional=True`` +
``dynamic=(...)`` refusal moved off the DECLARATION and onto the one lane that
cannot honour it.

pgw#746 refused the combination in ``Compile.__post_init__`` on two premises,
both of which pgw#812 then measured away:

* *"regional never applies the declared marks, so the declaration is inert
  while the contract digest claims the dynamism"* — true of the DYNAMO branch
  only. Regional has an EXPORT counterpart (a block is exported with
  ``dynamic_shapes`` exactly like a whole graph), and the symbolic inner axis
  measured FREE on a conv-free region: +0.2% bf16 / 0.0% w8a8, against
  pgw#730's +7.2% for the same axis on sdxl's conv lane.
* *"regional is retiring in favour of whole-transformer export
  (LTX-AOT-DESIGN §3.2), so teaching a departing lane to mark is building on
  sand"* — regional is now 14.2x the whole-graph mint on the real sdxl w8a8
  cell with serve parity at +0.24%. It is not departing; it is the adoption.

What pgw#746 was RIGHT about survives, and this file pins it: the dynamo
regional branch still calls ``compile_repeated_blocks(dynamic=None)`` and
still cannot apply the marks, so it must never arm a graph that does not
implement the contract its cell key asserts (the pgw#716 failure class). It
now DECLINES BY NAME and the target falls through to the whole-forward
branch — which does mark — so the declaration is still served, just by the
other lane. A decline that SKIPPED the target would be a silent regression
(an uncompiled target), which is why the fall-through is asserted here.

Tested in both directions: the combination is admitted at declaration and
declined at the dynamo arm, and each half alone is untouched.
"""

from __future__ import annotations

from gen_worker import compile_cache as cc
from gen_worker.api.decorators import Compile, DynamicDim


class _ContractCfg:
    """Duck-typed contract source — bypasses Compile's validation so the test
    can address the digest directly."""

    def __init__(self, *, dynamic=(), regional=False):
        self.shapes = ((768, 768),)
        self.targets = ("transformer",)
        self.text_len = 0
        self.dynamic = dynamic
        self.regional = regional
        self.lora_bucket = 0
        self.guidance_scales = ()


_SEQ = DynamicDim(dim="sequence", min=2, max=64)


# ---------------------------------------------------------------------------
# the digest still distinguishes the two, and that is now CORRECT
# ---------------------------------------------------------------------------

def test_the_digest_distinguishes_a_declared_dynamic_regional_config() -> None:
    """pgw#746 read this difference as the harm — two artifacts that compile
    identically carrying different keys. Under D4 it is the point: the export
    lane implements the marks, so the two configs genuinely produce different
    artifacts and MUST key differently."""
    without = cc.declared_contract_facts(_ContractCfg(regional=True))
    with_dyn = cc.declared_contract_facts(
        _ContractCfg(regional=True, dynamic=(_SEQ,)))

    assert without["regional"] is True and with_dyn["regional"] is True
    assert without["dynamic"] == []
    assert with_dyn["dynamic"] == [{"dim": "sequence", "min": 2, "max": 64}]
    assert without != with_dyn


# ---------------------------------------------------------------------------
# direction 1: the DECLARATION admits it (the relocation)
# ---------------------------------------------------------------------------

def test_regional_with_declared_dynamic_is_admitted() -> None:
    cfg = Compile(shapes=((768, 768),), regional=True, dynamic=(_SEQ,))
    assert cfg.regional is True
    assert cfg.dynamic == (_SEQ,), (
        "the declaration must survive intact — the export lane implements it"
    )


def test_several_declared_dims_are_admitted_alongside_regional() -> None:
    dims = (DynamicDim(dim="batch", min=2, max=8), _SEQ)
    cfg = Compile(shapes=((768, 768),), regional=True, dynamic=dims)
    assert cfg.dynamic == dims


# ---------------------------------------------------------------------------
# direction 2: the DYNAMO ARM declines it, by name, naming every dim
# ---------------------------------------------------------------------------

def test_the_dynamo_regional_arm_declines_and_names_every_offending_dim() -> None:
    dims = (DynamicDim(dim="batch", min=2, max=8), _SEQ)
    cfg = Compile(shapes=((768, 768),), regional=True, dynamic=dims)

    reason = cc._regional_dynamic_decline(cfg, "transformer")
    assert reason, "the dynamo regional branch cannot honour declared marks"
    assert "transformer" in reason
    assert "batch" in reason and "sequence" in reason
    assert "compile_repeated_blocks" in reason, (
        "the reason must name the mechanism, not just the verdict"
    )
    assert "whole-forward" in reason, (
        "and it must say where the target goes instead — a decline that read "
        "as a skip would be an uncompiled target"
    )


def test_regional_without_declared_dynamic_does_not_decline() -> None:
    """The half pgw#746 always allowed: nothing to honour, nothing to decline."""
    cfg = Compile(shapes=((768, 768),), regional=True)
    assert cc._regional_dynamic_decline(cfg, "transformer") == ""


# ---------------------------------------------------------------------------
# direction 3: each half alone is untouched
# ---------------------------------------------------------------------------

def test_regional_without_declared_dynamic_is_accepted() -> None:
    cfg = Compile(shapes=((768, 768),), regional=True)
    assert cfg.regional is True
    assert cfg.dynamic == ()


def test_declared_dynamic_without_regional_is_accepted_and_kept() -> None:
    """The whole-graph branch is the one that DOES mark, so this stays legal
    and the declaration must survive validation intact."""
    cfg = Compile(shapes=((768, 768),), regional=False, dynamic=(_SEQ,))
    assert cfg.regional is False
    assert cfg.dynamic == (_SEQ,)


def test_neither_is_accepted() -> None:
    cfg = Compile(shapes=((768, 768),))
    assert cfg.regional is False and cfg.dynamic == ()
