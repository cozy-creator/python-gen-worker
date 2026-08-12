"""pgw#1151: a declared range is a CONTRACT, not a strict dynamo constraint.

``_with_declared_marks`` used to forward the declaration into
``mark_dynamic(t, dim, min=, max=)``. Dynamo turns explicit bounds into a
``StrictMinMaxConstraint`` (``_dynamo/variables/builder.py``), and a strict
constraint makes any range NARROWING the compiler performs a hard
``ConstraintViolationError`` — which ``_guarded_regional`` can only answer by
degrading the target to eager for the life of the pod.

Compilers narrow ranges for reasons that carry no correctness content. The one
that cost us a product: inductor elects int32 indexing from the FIRST call's
size hint and then installs ``check_leq(numel, INT32_MAX)``
(``_inductor/codegen/simd.py::can_use_32bit_indexing``). On minimax-h3 the
DiT's largest planned buffer has a 28,672-element inner dim, so a cold 5 s
request (38,015 packed rows) pinned int32 and its guard —
``sequence <= 74,898`` — contradicted the declared max. Every width above that
took the refusal, which is the whole reason 11-15 s clips served 100% eager.
Nothing about wide sequences is uncompilable: with no strict upper bound
inductor simply picks int64 for the wide graph.

Marking without bounds yields a ``RelaxedUnspecConstraint`` instead: still no
specialization to a constant (the pgw#1082 failure ``_with_declared_marks``
exists to prevent), but the compiler may split the range and the wide call
RECOMPILES. Both tests below are RED on the pre-pgw#1151 code.
"""

from __future__ import annotations

import gc

import pytest
import torch

from gen_worker import compile_cache as cc


class _Dim:
    def __init__(self, dim: str, mn: int, mx: int) -> None:
        self.dim, self.min, self.max = dim, mn, mx


# A compiler that narrows the marked axis, in miniature: the branch installs a
# `sequence <= 1000` guard on the first (narrow) call, exactly the shape of
# inductor's `28672*sequence <= 2**31-1`.
_BLOCK_SRC = """
class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, hidden, index, rotary):
        cos, sin = rotary
        out = self.lin(hidden)
        if hidden.shape[1] <= 1000:
            out = out * 2.0
        scale = cos.index_select(0, index)
        return out * (1.0 + scale) + sin.index_select(0, index)
"""

_NARROWING_BOUND = 1000
_DECLARED_MAX = 4096


def _args(seq: int):
    return (
        torch.randn(1, seq, 8),
        torch.arange(seq),
        (torch.randn(seq, 8), torch.randn(seq, 8)),
    )


@pytest.fixture(autouse=True)
def _isolated_dynamo():
    gc.collect()
    torch._dynamo.reset()
    yield
    gc.collect()
    torch._dynamo.reset()


def _marked_block(dims):
    torch._dynamo.reset()
    ns = {"torch": torch}
    exec(_BLOCK_SRC, ns)
    block = ns["_Block"]().eval()
    block.compile(dynamic=None, fullgraph=True)
    block._compiled_call_impl = cc._with_declared_marks(
        block._compiled_call_impl, dims)
    return block


def test_a_compiler_narrowing_the_axis_recompiles_instead_of_degrading():
    """THE fix. RED before: the narrow call pinned a `sequence <= 1000` guard,
    the strict constraint from `max=4096` made that a
    ``ConstraintViolationError``, and the target degraded to eager. H3's
    real one was inductor's int32 index guard at 74,898 rows against a
    declared 116,126."""
    block = _marked_block((_Dim("sequence", 2, _DECLARED_MAX),))
    with torch.no_grad():
        block(*_args(96))                       # narrow: installs the guard
        out = block(*_args(_NARROWING_BOUND * 2))  # wide: must RECOMPILE
    assert out.shape == (1, _NARROWING_BOUND * 2, 8)
    audit = cc.graph_audit()
    assert audit.unique_graphs == 2, audit.summary()
    assert audit.graph_breaks == 0, audit.summary()


def test_the_declared_range_is_still_enforced_by_the_sdk_not_by_dynamo():
    """The guardrail. Dropping the bounds from ``mark_dynamic`` must not drop
    the CONTRACT: an out-of-range extent is still the typed, named refusal the
    executor turns into ``declared_range_exceeded``, and the marks dynamo
    receives must carry no bounds (that is what makes the constraint
    RELAXED — see ``_dynamo/variables/builder.py``)."""
    marks: list[dict] = []
    original = torch._dynamo.mark_dynamic
    torch._dynamo.mark_dynamic = lambda t, dim, **kw: marks.append(kw)
    try:
        wrapped = cc._with_declared_marks(
            lambda *a, **k: None, (_Dim("sequence", 2, _DECLARED_MAX),))
        wrapped(*_args(96))
    finally:
        torch._dynamo.mark_dynamic = original
    assert marks, "the declared axis must still be marked"
    assert all(kw == {} for kw in marks), marks

    block = _marked_block((_Dim("sequence", 2, _DECLARED_MAX),))
    with pytest.raises(cc.DeclaredRangeExceeded, match="sequence"):
        with torch.no_grad():
            block(*_args(_DECLARED_MAX + 1))
