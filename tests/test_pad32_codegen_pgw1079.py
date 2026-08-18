"""The pad-to-32 expression class.

z-image pads its flattened latent to a multiple of 32, and with both latent
axes declared dynamic that padded length becomes an algebraic function of the
declared symbols. HOW it is spelled decides whether the exported program is
servable.

This file asks the algebra half locally, on a 1.1 MB toy that carries the exact
same expression class. The CODEGEN half is the gauntlet's `micro-pad32` member
(export -> AOTI -> load -> serve -> parity at three pad classes from one
artifact); this is the cheap unit that says WHY that member is the right shape.

What is pinned:

* the fixed spelling's shape env holds NO equality over the declared symbols —
  every `Eq` it records about them is REFUTED (`False`), the class the
  declared-range gate must EVALUATE rather than count;
* the upstream spelling carries `Eq(PythonMod(-L, 32), 0)` plus `Ne`/`>=` RANGE
  restrictions that exclude the pad-0 row;
* both spellings compute the same values as the unpadded base denoiser, at
  pads 0, 16 and 28.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"
if str(MICRO_SRC) not in sys.path:
    sys.path.insert(0, str(MICRO_SRC))

from micro_diffusion.model import MicroConfig, MicroDenoiser  # noqa: E402
from micro_diffusion.model_pad32 import (  # noqa: E402
    SEQ_MULTIPLE_OF,
    MicroPad32Branchy,
    MicroPad32Denoiser,
    padded_length,
)

#: (grid, pad) — the three classes the gauntlet member serves from ONE compiled graph.
PAD_CLASSES = ((12, 16), (14, 28), (16, 0))


def _feed(config: MicroConfig, grid: int):
    gen = torch.Generator().manual_seed(637)
    x = [torch.randn(config.in_channels, grid, grid, generator=gen)]
    t = torch.full((1,), 100.0)
    cond = [torch.randn(config.cond_len, config.cond_dim, generator=gen)]
    return x, t, cond


def _shape_env(exported):
    for node in exported.graph.nodes:
        val = node.meta.get("val")
        node_ = getattr(val, "node", None)
        env = getattr(node_, "shape_env", None)
        if env is not None:
            return env
    return None


def _export(cls, grid: int):
    module = cls(MicroConfig()).eval()
    args = _feed(module.config, grid)
    height = torch.export.Dim("H", min=4, max=64)
    width = torch.export.Dim("W", min=4, max=64)
    spec = ([{1: 2 * height, 2: 2 * width}],
            {0: torch.export.Dim.STATIC},
            [{0: torch.export.Dim.STATIC, 1: torch.export.Dim.STATIC}])
    return torch.export.export(module, args, dynamic_shapes=spec, strict=True)


def _equalities(env) -> dict:
    """Every `Eq(...)` the shape env recorded, mapped to what it decided."""
    return {str(k): str(v) for k, v in env.axioms.items()
            if str(k).startswith(("Eq(", "Eq "))}


def test_the_pad_lengths_this_family_declares_are_three_different_classes():
    """The premise. If every declared row padded the same amount, a graph that
    decided the pad once would serve them all and the question would be moot —
    which is exactly why z-image's declared grid was the counterexample."""
    pads = {padded_length(g * g) - g * g for g, _ in PAD_CLASSES}
    assert pads == {0, 16, 28}
    for grid, pad in PAD_CLASSES:
        assert padded_length(grid * grid) - grid * grid == pad
        assert padded_length(grid * grid) % SEQ_MULTIPLE_OF == 0


def test_the_fixed_spelling_holds_no_equality_over_the_declared_symbols():
    env = _shape_env(_export(MicroPad32Denoiser, 12))
    assert env is not None
    held = [expr for expr, value in _equalities(env).items()
            if value == "True" and "PythonMod" in expr]
    assert held == [], f"the fixed spelling pinned the pad: {held!r}"


def test_the_upstream_spelling_carries_ie637s_guard_verbatim():
    """The RED control, and the reason the fixed member's green means
    something. This is ie#566 G3's class: `Eq(PythonMod(-L, 32), 0)` — refuted
    here, which is what pgw#1077 taught the gate to see, and accompanied by
    `Ne`/inequality restrictions that EXCLUDE the pad-0 declared row."""
    env = _shape_env(_export(MicroPad32Branchy, 12))
    assert env is not None
    equalities = _equalities(env)
    assert any("PythonMod" in expr and "32" in expr and value == "False"
               for expr, value in equalities.items()), equalities
    guards = [str(g.expr) for g in env.guards]
    assert any(expr.startswith("Ne(PythonMod(") for expr in guards), guards


@pytest.mark.parametrize("grid,pad", PAD_CLASSES)
def test_both_spellings_agree_with_the_unpadded_base_at_every_pad(grid, pad):
    """The pad must be VALUE-neutral. A padded path that changed the answer
    would make every parity number downstream meaningless."""
    config = MicroConfig()
    base = MicroDenoiser(config).eval()
    args = _feed(config, grid)
    tokens = [g.reshape(g.shape[0], -1).transpose(0, 1) for g in args[0]]
    with torch.no_grad():
        want = base(tokens, args[1], args[2])
        want = want.transpose(1, 2).reshape(1, config.in_channels, grid, grid)
        for cls in (MicroPad32Denoiser, MicroPad32Branchy):
            module = cls(config).eval()
            module.load_state_dict(base.state_dict(), strict=True)
            got = module(*args)
            assert torch.allclose(got, want, atol=1e-5), (
                f"{cls.__name__} changed the value at pad {pad}")
