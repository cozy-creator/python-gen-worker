"""pgw#746 (OQ-7): ``regional=True`` + ``dynamic=(...)`` is refused by name.

``_with_declared_marks`` is applied only on the whole-graph branch of
``compile_cache._arm``. The regional branch calls
``compile_repeated_blocks(dynamic=None)`` and returns before the marks are
ever reached — so the combination declares dynamism the trace never
implements, while ``declared_contract_facts`` folds that same declaration into
the contract digest. The key then asserts a contract the artifact does not
honor, which is the exact failure class pgw#716 exists to prevent.

The fix is refusal, not marking: regional is retiring (LTX-AOT-DESIGN.md §3.2
chose whole-transformer export), so teaching a departing lane to mark would be
building on sand. Refusal is honest until it goes.

Tested both directions — the combination is refused, and each half alone is
still accepted.
"""

from __future__ import annotations

import pytest

from gen_worker import compile_cache as cc
from gen_worker.api.decorators import Compile, DynamicDim


class _ContractCfg:
    """Duck-typed contract source — bypasses Compile's validation so the test
    can show what the digest WOULD have claimed."""

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
# the harm is real: the digest does fold in a declaration the trace ignores
# ---------------------------------------------------------------------------

def test_the_digest_would_have_claimed_dynamism_the_regional_trace_ignores() -> None:
    """Why this is refused rather than tolerated. Both configs compile the
    same way regionally — blocks, no marks — yet they carry DIFFERENT contract
    digests, so the key distinguishes artifacts that are byte-identical in
    behavior and claims a dynamism neither of them has."""
    without = cc.declared_contract_facts(_ContractCfg(regional=True))
    with_dyn = cc.declared_contract_facts(
        _ContractCfg(regional=True, dynamic=(_SEQ,)))

    assert without["regional"] is True and with_dyn["regional"] is True
    assert without["dynamic"] == []
    assert with_dyn["dynamic"] == [{"dim": "sequence", "min": 2, "max": 64}]
    assert without != with_dyn, (
        "the declaration reaches the digest even on the regional branch — "
        "which is precisely why it must not be declarable there"
    )


# ---------------------------------------------------------------------------
# direction 1: the combination is refused, by name
# ---------------------------------------------------------------------------

def test_regional_with_declared_dynamic_is_refused() -> None:
    with pytest.raises(ValueError) as excinfo:
        Compile(shapes=((768, 768),), regional=True, dynamic=(_SEQ,))

    message = str(excinfo.value)
    assert "regional" in message
    assert "dynamic" in message
    # names the offending dim, so the author knows what to drop
    assert "sequence" in message


def test_the_refusal_names_every_offending_dim() -> None:
    dims = (DynamicDim(dim="batch", min=2, max=8), _SEQ)
    with pytest.raises(ValueError) as excinfo:
        Compile(shapes=((768, 768),), regional=True, dynamic=dims)

    message = str(excinfo.value)
    assert "batch" in message and "sequence" in message


# ---------------------------------------------------------------------------
# direction 2: each half alone is untouched
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
