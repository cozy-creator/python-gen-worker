"""pgw#1082: the regional lane armed, entered, never guard-missed — and ran
100% EAGER while every telemetry axis said compiled.

With routing correct (a ``regional=True`` + ``dynamic=(...)`` target takes the
dynamo regional branch), four defects remain in what it then EXECUTES:

1. ``_with_declared_marks`` marked one dim of one KIND of tensor. The sibling
   tensors carrying the same axis — the integer ``adaln_indices[S]``, the
   ``(cos[S, D], sin[S, D])`` rotary TUPLE — stayed static, dynamo specialized
   the symbol on them, and the mark on the float raised
   ``ConstraintViolationError`` on the FIRST call of every block.
2. The declared range was the TEXT bound (4,096) while the marked axis is the
   whole PACKED sequence (38,015), so it was out of range as well. That is an
   endpoint declaration defect and now says so by name.
3. ``_guarded_regional`` caught the raise, cleared the block compilations and
   served eager forever — but, unlike its ``_guarded`` twin, NEVER set
   ``failure_signal["degraded"]``. ``is_compile_armed`` reads exactly that, so
   the wire kept reporting ``serving_mode=jit_cell``,
   ``served_eager_fallback=false``, empty ``fallback_reason``.
4. ``emit_jit_compile_event``'s ``n_graphs`` had no caller, so nothing could
   see any of it.

Every test here is RED on the pre-pgw#1082 code.
"""

from __future__ import annotations

import gc
import threading

import pytest
import torch

from gen_worker import compile_cache as cc


class _Dim:
    def __init__(self, dim: str, mn: int, mx: int) -> None:
        self.dim, self.min, self.max = dim, mn, mx


_BLOCK_SRC = """
class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, hidden, index, rotary):
        # The H3 block's modulation, in miniature: a per-row table gathered by
        # an INTEGER index tensor of the sequence length, broadcast into the
        # rank-3 activation. Marking only `hidden` leaves the gather static,
        # which specializes the symbol and violates the mark.
        cos, sin = rotary
        out = self.lin(hidden)
        scale = cos.index_select(0, index)
        shift = sin.index_select(0, index)
        return out * (1.0 + scale) + shift
"""


def _fresh_block_cls():
    """The H3 block's shape in miniature: one float activation carrying the
    sequence axis, one INTEGER index tensor carrying it, and a rotary TUPLE
    carrying it — the three ways the old mapping could miss an axis.

    A FRESH code object per test: dynamo's caches are keyed on the code
    object, so two tests declaring different ranges over one class would
    otherwise inherit each other's guards.
    """
    ns = {"torch": torch}
    exec(_BLOCK_SRC, ns)
    return ns["_Block"]


def _args(seq: int):
    return (
        torch.randn(1, seq, 8),
        torch.arange(seq),
        (torch.randn(seq, 8), torch.randn(seq, 8)),
    )


@pytest.fixture(autouse=True)
def _isolated_dynamo():
    """`nn.Module.compile()` traces the SHARED `Module._call_impl` frame, so
    one test's declared range leaks into the next unless the caches are torn
    down with the objects that hold them."""
    gc.collect()
    torch._dynamo.reset()
    yield
    gc.collect()
    torch._dynamo.reset()


def _marked_block(dims, fullgraph: bool = True):
    torch._dynamo.reset()
    block = _fresh_block_cls()().eval()
    block.compile(dynamic=None, fullgraph=fullgraph)
    if dims:
        block._compiled_call_impl = cc._with_declared_marks(
            block._compiled_call_impl, dims)
    return block


def test_the_declared_axis_is_marked_on_every_argument_that_carries_it():
    """RED before: only dim 1 of rank>=3 FLOATS was marked, the index tensor
    and the rotary tuple stayed static, and dynamo raised
    ``ConstraintViolationError: you marked ... as dynamic but your code
    specialized it to be a constant``. ONE graph must serve two lengths."""
    from torch._dynamo.utils import counters

    counters.clear()
    block = _marked_block((_Dim("sequence", 2, 4096),))
    with torch.no_grad():
        block(*_args(96))
        block(*_args(128))
    audit = cc.graph_audit()
    assert audit.unique_graphs == 1, audit.summary()
    assert audit.graph_breaks == 0, audit.summary()


def test_an_extent_outside_the_declared_range_is_a_typed_named_refusal():
    """RED before: a dynamo-internal ``ConstraintViolationError`` that named
    no declaration and no endpoint. minimax-h3 declared max=4096 for an axis
    its own requests present at 38,015."""
    block = _marked_block((_Dim("sequence", 2, 4096),))
    try:
        with torch.no_grad():
            block(*_args(8192))
    except cc.DeclaredRangeExceeded as exc:
        assert "8192" in str(exc) and "4096" in str(exc)
        assert "sequence" in str(exc)
    else:
        raise AssertionError("an out-of-range extent must refuse, typed")


def test_the_marks_reach_the_integer_index_and_the_rotary_tuple():
    """THE fix, stated directly. RED before: the mapping was "dim 0 of every
    float for batch, dim 1 of every rank-3 float for sequence", so the
    ``adaln_indices[S]`` integer tensor and the ``(cos[S,D], sin[S,D])``
    rotary TUPLE — both carrying the SAME axis — were never marked, and
    dynamo specialized the symbol on them."""
    marked: list[tuple[tuple[int, ...], int]] = []

    def _spy(t, dim, **kw):
        marked.append((tuple(t.shape), dim))

    original = torch._dynamo.mark_dynamic
    torch._dynamo.mark_dynamic = _spy
    try:
        wrapped = cc._with_declared_marks(
            lambda *a, **k: None, (_Dim("sequence", 2, 262144),))
        wrapped(*_args(96))
    finally:
        torch._dynamo.mark_dynamic = original

    assert ((1, 96, 8), 1) in marked, marked      # the float activation
    assert ((96,), 0) in marked, marked           # the INTEGER index tensor
    assert marked.count(((96, 8), 0)) == 2, marked  # both rotary tuple members


def test_a_degraded_regional_target_stops_claiming_it_is_compiled():
    """THE LIE. RED before: ``_guarded_regional``'s permanent-degrade branch
    never set ``degraded``, which ``is_compile_armed`` reads — so a target
    that fell to eager on its first call reported ``serving_mode=jit_cell``
    with an empty ``fallback_reason`` for the life of the pod."""

    class _Owner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("regional block exploded")
            return "eager"

    owner = _Owner()
    signal = {
        "lock": threading.Lock(), "successful_calls": 0, "cache_hits": 0,
        "cache_misses": 0, "guard_misses": 0, "on_guard_miss": None,
        "callback": None, "router": None,
    }
    guarded = cc._guarded_regional(
        owner, owner.forward, "transformer", failure_signal=signal)
    pipeline = type("_P", (), {})()
    setattr(pipeline, cc._MARKER_ATTR, {"failure_signal": signal,
                                        "originals": (), "regional_mods": ()})
    assert cc.is_compile_armed(pipeline) is True
    assert guarded() == "eager"   # degrades to eager, serves the call
    assert signal.get("degraded") is True
    assert cc.is_compile_armed(pipeline) is False


def test_a_declared_range_refusal_is_readable_off_the_pipeline():
    """The executor turns this into the ``declared_range_exceeded`` eager
    posture, so the request row names the endpoint's declaration defect."""

    class _Owner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                raise cc.DeclaredRangeExceeded(
                    "declared dynamic axis 'sequence' has range [2, 4096]")
            return "eager"

    owner = _Owner()
    signal = {
        "lock": threading.Lock(), "successful_calls": 0, "cache_hits": 0,
        "cache_misses": 0, "guard_misses": 0, "on_guard_miss": None,
        "callback": None, "router": None,
    }
    guarded = cc._guarded_regional(
        owner, owner.forward, "transformer", failure_signal=signal)
    pipeline = type("_P", (), {})()
    setattr(pipeline, cc._MARKER_ATTR, {"failure_signal": signal,
                                        "originals": (), "regional_mods": ()})
    assert guarded() == "eager"
    assert "sequence" in cc.declared_range_refusal(pipeline)
    assert cc.is_compile_armed(pipeline) is False


def test_the_graph_audit_counts_what_no_caller_ever_populated():
    """RED before: ``emit_jit_compile_event``'s ``n_graphs`` had NO caller, so
    every ``jit_compile`` event on the platform read ``n_graphs=0``."""
    from torch._dynamo.utils import counters

    counters.clear()
    before = cc.graph_audit()
    block = _marked_block(())
    with torch.no_grad():
        block(*_args(64))
    delta = cc.graph_audit_delta(before)
    assert delta.unique_graphs >= 1
    assert "n_graphs=" in delta.summary() and "n_breaks=" in delta.summary()
