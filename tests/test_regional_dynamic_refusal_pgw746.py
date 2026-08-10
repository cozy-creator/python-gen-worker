"""pgw#746 (OQ-7) -> pgw#817/D4 -> **pgw#1078, which closes it**: the
``regional=True`` + ``dynamic=(...)`` combination is now SERVED by the dynamo
regional branch instead of declined by it.

pgw#746 refused the combination in ``Compile.__post_init__``; pgw#817/D4 moved
the refusal onto the dynamo arm, which then sent such a target to the
whole-forward branch. Both rested on one premise — *"the dynamo regional
branch cannot apply the declared marks"* — and ie#632 measured what it costs:
minimax-h3's 20.1B denoiser declares ``regional=True`` precisely because
whole-graph inductor planning is unaffordable for its class (ie#381), and the
decline compiled it whole-forward anyway. Its boot warmup compiled one shape;
every real request presented a different packed sequence, guard-missed, and
`_guarded` served eager for the life of the pod (`lane=bf16-w16a16+eager`,
257.7 s against a 131.7 s measured compiled wall).

The premise was simply wrong. ``compile_repeated_blocks`` compiles each
repeated BLOCK, so the block call is where this lane's graphs are traced and
where the marks belong — ``_mark_regional_blocks`` wraps each block's
``_compiled_call_impl`` with the same ``_with_declared_marks`` the
whole-forward branch uses. One declaration, two lanes, one meaning.

What pgw#746 was right about is unchanged and still pinned below: the contract
digest distinguishes the two configs, because they genuinely produce different
artifacts.
"""

from __future__ import annotations

import torch

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
    without = cc.declared_compile_facts(_ContractCfg(regional=True))
    with_dyn = cc.declared_compile_facts(
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
# direction 2: the DYNAMO REGIONAL ARM honours the marks at the block ingress
# ---------------------------------------------------------------------------

class _Block(torch.nn.Module):
    def forward(self, hidden_states):  # pragma: no cover - never called here
        return hidden_states


class _Owner(torch.nn.Module):
    _repeated_blocks = ["_Block"]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block(), _Block()])


def test_the_dynamo_regional_arm_marks_every_repeated_block() -> None:
    """The fix, stated as the behaviour: after ``compile_repeated_blocks``
    every repeated block's compiled call is wrapped so the DECLARED dims are
    marked before dynamo sees the inputs."""
    owner = _Owner()
    for block in owner.blocks:
        block._compiled_call_impl = block._call_impl

    marked = cc._mark_regional_blocks(
        owner, (DynamicDim(dim="batch", min=2, max=8), _SEQ))

    assert marked == 3, "every repeated block carries the declaration"
    for block in owner.blocks:
        assert getattr(block._compiled_call_impl, "_gw_declared_marks", False)


def test_marking_regional_blocks_is_idempotent() -> None:
    """A second arm on the same object must not stack mark wrappers."""
    owner = _Owner()
    for block in owner.blocks:
        block._compiled_call_impl = block._call_impl
    assert cc._mark_regional_blocks(owner, (_SEQ,)) == 3
    assert cc._mark_regional_blocks(owner, (_SEQ,)) == 0


def test_an_uncompiled_block_is_never_wrapped() -> None:
    """``compile_repeated_blocks`` is what installs ``_compiled_call_impl``;
    a block it did not compile has no ingress to mark."""
    owner = _Owner()
    assert cc._mark_regional_blocks(owner, (_SEQ,)) == 0


def test_the_declared_marks_reach_the_block_input() -> None:
    """Not decoration: the wrapper must actually mark the declared dim on the
    tensor the block is called with."""
    owner = _Owner()
    seen: list = []

    def _impl(hidden_states):
        seen.append(tuple(
            getattr(hidden_states, "_dynamo_dynamic_indices", set()) or ()))
        return hidden_states

    owner.blocks[0]._compiled_call_impl = _impl
    cc._mark_regional_blocks(owner, (_SEQ,))
    owner.blocks[0]._compiled_call_impl(torch.zeros(1, 8, 4))

    assert seen == [(1,)], (
        "`sequence` marks dim 1 of a rank-3 float tensor, exactly as the "
        "whole-forward branch marks it"
    )


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
