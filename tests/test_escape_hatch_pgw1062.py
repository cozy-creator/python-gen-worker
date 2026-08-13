"""pgw#1062 — the escape hatch (pgw#1059 amendment 7), asserted where CI can see it.

The full escape-hatch cycle — a custom op, a ``triton_op`` kernel and a raw
``@triton.jit`` call minted, published, adopted and parity-checked — is the
``micro-escape`` gauntlet member and is LOCAL-ONLY (Triton has no CPU
backend). What does NOT need a card is every claim about the SURFACE itself,
and those are the ones that would rot silently:

* a custom op without a fake kernel must refuse at export with GUIDANCE, not
  crash obscurely — that refusal text is the authoring contract;
* a custom op WITH a fake kernel must trace under ``FakeTensorMode`` — the
  exact property pgw#1056's fake-weight mint stands on;
* an exported program carrying a custom op must survive the mint's own
  ``torch.export.save``/``load`` process hand-off;
* flex_attention does NOT survive that hand-off on torch 2.13 — the pinned
  upstream break behind pgw#1062's flex verdict. The pin is written to go
  red the day upstream fixes it, so the verdict gets revisited instead of
  fossilizing.

Each test is written so it goes RED if the property it names stops holding.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gen_worker.aot_mint import MintRefused, export_program


# ---------------------------------------------------------------------------
# The ops under test. Registered at module import, once per process.
# ---------------------------------------------------------------------------


@torch.library.custom_op("pgw1062_test::with_fake", mutates_args=())
def _with_fake(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0


@_with_fake.register_fake
def _with_fake_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("pgw1062_test::no_fake", mutates_args=())
def _no_fake(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0


class _WithFake(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.pgw1062_test.with_fake(x) + 1.0


class _NoFake(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.pgw1062_test.no_fake(x) + 1.0


# ---------------------------------------------------------------------------
# The refusal is typed and carries the fix
# ---------------------------------------------------------------------------


def test_a_custom_op_without_a_fake_kernel_refuses_with_guidance() -> None:
    """The mint's export wrapper must hand the author torch's own guidance.

    ``export_program`` wraps every export failure in :class:`MintRefused`;
    what matters here is that the wrapped text still NAMES the fix
    (``register_fake``), because that message is the entire authoring
    contract for the escape hatch's fake-kernel requirement.
    """
    with pytest.raises(MintRefused) as excinfo:
        export_program(_NoFake(), (torch.randn(4, 8),), {})
    assert "register_fake" in str(excinfo.value), (
        "the export refusal no longer carries torch's register_fake "
        "guidance — an author with a meta-less custom op now gets an "
        "unactionable error")


# ---------------------------------------------------------------------------
# The fake-kernel property pgw#1056 stands on
# ---------------------------------------------------------------------------


def test_a_custom_op_with_a_fake_kernel_traces_under_fake_mode() -> None:
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    with FakeTensorMode():
        x = torch.randn(4, 8)
        y = torch.ops.pgw1062_test.with_fake(x)
        assert isinstance(y, FakeTensor)
        ep = torch.export.export(_WithFake(), (x,), strict=True)
    ops = {str(node.target) for node in ep.graph.nodes
           if node.op == "call_function"}
    assert any("with_fake" in name for name in ops), sorted(ops)


# ---------------------------------------------------------------------------
# The process hand-off (torch.export.save/load — aot_compile_pool._stage)
# ---------------------------------------------------------------------------


def test_a_custom_op_program_survives_the_export_save_load_handoff(
    tmp_path,
) -> None:
    ep = torch.export.export(_WithFake(), (torch.randn(4, 8),), strict=True)
    torch.export.save(ep, tmp_path / "program.pt2")
    loaded = torch.export.load(tmp_path / "program.pt2")
    ops = {str(node.target) for node in loaded.graph.nodes
           if node.op == "call_function"}
    assert any("with_fake" in name for name in ops), (
        f"the custom-op node did not survive torch.export.save/load — the "
        f"mint's staged hand-off would drop it (graph ops: {sorted(ops)})")
    got = loaded.module()(torch.ones(4, 8))
    assert float((got - 3.0).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# The flex_attention pin — pgw#1062's measured upstream break
# ---------------------------------------------------------------------------


def test_flex_attention_still_breaks_at_the_save_handoff() -> None:
    """The PINNED upstream gap behind pgw#1062's flex verdict.

    Measured on 2.13.0 (cu126 and cpu identically): flex_attention EXPORTS,
    and AOTI compiles + serves it with parity 5.7e-7 when export and compile
    share a process — but ``torch.export.save`` refuses the flex HOP's
    block-mask tuple argument (``torch/_export/serde/serialize.py``:
    ``SerializeError: Unsupported list/tuple argument type``). The mint
    stages every compiled graph across exactly that save/load boundary
    (``aot_compile_pool._stage``), so flex cannot ride the pipeline today.

    IF THIS TEST FAILS because the save SUCCEEDED: upstream fixed HOP serde.
    That is good news, deliberately made loud — reopen pgw#1062's flex
    verdict (build the micro-flex gauntlet member) instead of deleting the
    test.
    """
    from torch._export.serde.serialize import SerializeError
    from torch.nn.attention.flex_attention import flex_attention

    def rel_bias(score, b, h, q_idx, kv_idx):
        return score + (q_idx - kv_idx) * 0.001

    class FlexBlock(nn.Module):
        def forward(self, q, k, v):
            return flex_attention(q, k, v, score_mod=rel_bias)

    q, k, v = (torch.randn(1, 2, 128, 32) for _ in range(3))
    ep = torch.export.export(FlexBlock(), (q, k, v), strict=True)

    import tempfile
    from pathlib import Path

    with pytest.raises(SerializeError):
        torch.export.save(
            ep, Path(tempfile.mkdtemp(prefix="pgw1062-flex-")) / "f.pt2")
