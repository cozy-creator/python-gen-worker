"""pgw#1156: the fp8 activation quantize is FUSED, and its failure is loud.

Measured on H100 at H3's real shapes: the eager op-by-op quantize costs six full
passes over the [M, K] activation while the GEMM scales with M*K*N, so on the
thin-N projections (`attn.to_out.0`, `ff.net.2`) the shipped fp8 forward ran
**0.82-0.93x bf16** — slower than the precision it replaces — at every duration.
Fusing the chain takes those classes to 1.21-1.39x.

Two halves, both CPU-only:

1. the fused path must compute EXACTLY what the pre-fix inline expression
   computed, on all three scale branches (rowwise dynamic / pertensor dynamic /
   static). The reference here is written out longhand rather than imported, so
   a semantic drift in the refactor is RED.
2. a host where the fusion cannot build must keep serving AND say so — an
   unfused pod keeps the fp8 memory saving and loses the speed, which is the
   pgw#824 silent-lane shape.
"""

from __future__ import annotations

from typing import Any, List

import pytest

torch = pytest.importorskip("torch")

from gen_worker import activity  # noqa: E402
from gen_worker.models import w8a8  # noqa: E402

FP8_MAX = 448.0


@pytest.fixture()
def events(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    captured: List[Any] = []
    monkeypatch.setattr(activity, "_emit", captured.append)
    return captured


def _reference(x2: Any, pertensor: bool, static: Any) -> tuple:
    """The expression the module carried BEFORE the fusion landed, verbatim."""
    if static is not None:
        sa = (static if pertensor
              else static.expand(x2.shape[0], 1).contiguous())
    elif pertensor:
        sa = (x2.abs().amax().float() / FP8_MAX).clamp(min=1e-12).reshape(1, 1)
    else:
        sa = (x2.abs().amax(dim=-1, keepdim=True).float()
              / FP8_MAX).clamp(min=1e-12)
    xq = (x2 * (1.0 / sa).to(x2.dtype)).clamp(
        -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return xq, sa


CASES = [
    ("rowwise_dynamic", False, False),
    ("pertensor_dynamic", True, False),
    ("rowwise_static", False, True),
    ("pertensor_static", True, True),
]


def _faithfulness(xq: Any, sa: Any, x2: Any) -> float:
    """Relative error of the quantized activation against the bf16 original."""
    deq = xq.float() * sa.float()
    return float((deq - x2.float()).norm() / x2.float().norm())


@pytest.mark.parametrize("name,pertensor,use_static", CASES)
def test_fused_quantize_is_the_same_recipe_and_no_less_faithful(
    name: str, pertensor: bool, use_static: bool,
) -> None:
    """NOT bitwise: fusing the chain lets inductor hold the reciprocal-multiply in
    fp32 instead of rounding it to bf16 between ops, so a handful of codes land one
    fp8 step away. That is the SAME divergence the compiled arm has always had
    against the eager one — the fix makes the two agree, it does not open a new
    gap — and it moves in the accurate direction. What must not drift is the
    recipe: the scale is a pure reduction and stays bit-exact, and the codes stay
    overwhelmingly identical."""
    torch.manual_seed(1156)
    x2 = torch.randn(64, 128, dtype=torch.bfloat16) * 7.5
    static = torch.tensor([[0.013]], dtype=torch.float32) if use_static else None

    quantize = w8a8._build_quantizer(torch)
    xq, sa = quantize(x2, pertensor, static)
    rq, rs = _reference(x2, pertensor, static)

    assert xq.dtype == torch.float8_e4m3fn
    # the GEMM's scale-shape contract: rowwise wants [M,1], pertensor wants [1,1]
    assert tuple(sa.shape) == ((1, 1) if pertensor else (x2.shape[0], 1))
    assert torch.equal(sa, rs), f"{name}: the scale recipe drifted"

    same = (xq.view(torch.uint8) == rq.view(torch.uint8)).float().mean().item()
    assert same > 0.95, f"{name}: {1 - same:.4f} of codes differ — not a rounding gap"
    # every disagreement is at most ONE representable e4m3 step (3 mantissa bits)
    f, r = xq.float() * sa, rq.float() * rs
    step = 2.0 ** -3 * 1.01
    assert int(((f - r).abs() > step * torch.maximum(f.abs(), r.abs())).sum()) == 0, name
    assert _faithfulness(xq, sa, x2) <= _faithfulness(rq, rs, x2) + 1e-6, name


def test_a_host_that_cannot_fuse_still_serves_and_confesses(
    events: List[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_inductor(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("inductor exploded")

    monkeypatch.setattr(torch, "compile", _no_inductor)
    torch.manual_seed(1156)
    x2 = torch.randn(32, 64, dtype=torch.bfloat16)

    quantize = w8a8._build_quantizer(torch)
    xq, sa = quantize(x2, False, None)
    rq, rs = _reference(x2, False, None)

    # BEHAVIOR unchanged — the lane is correct, only slow
    assert torch.equal(xq.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(sa, rs)

    got = [e for e in events if e.kind == activity.KIND_SERVE_DEGRADE]
    assert [e.phase for e in got] == ["w8a8_quant_unfused"]
    assert "inductor exploded" in got[0].detail

    # and it degrades ONCE, not on every activation
    quantize(x2, False, None)
    quantize(x2, False, None)
    assert len([e for e in events
                if e.kind == activity.KIND_SERVE_DEGRADE]) == 1


def test_inside_a_compiled_region_the_helper_nests_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The enclosing graph already fuses the chain; building a second inductor
    graph inside it is pure cost (and, under AOT export, a hazard)."""
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("torch.compile must not be called while compiling")

    monkeypatch.setattr(torch, "compile", _boom)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    x2 = torch.randn(16, 32, dtype=torch.bfloat16)
    quantize = w8a8._build_quantizer(torch)
    xq, sa = quantize(x2, False, None)
    rq, rs = _reference(x2, False, None)
    assert torch.equal(xq.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(sa, rs)
