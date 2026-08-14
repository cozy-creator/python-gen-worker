"""A REFUTED equality is not a pin.

``ShapeEnv.axioms`` is ``{relation: sympy.true | sympy.false}`` and torch's
``symbolic_shapes.get_implications`` deposits ``Eq(a, b) => false`` plus its
commuted mirror for every ``Ne(a, b)`` the graph PROVES. Iterating that map by
KEY therefore reports every inequality the graph settled as an equality guard
PINNING the declared symbols, in mirrored pairs, and refuses a correct
declaration.

Every test here drives a real ``torch.export`` and asserts on the shipped gate.
The controls matter as much as the repro: a genuine ``Eq(h*w, N)`` pin must
still refuse — from the axioms source alone, not only from ``guards`` — and an
axiom whose value this gate cannot read stays refused, fail-closed.

Scope note: the branch that proves ``Ne`` narrows the served set only when the
refuted value is REACHABLE inside the declared hull. This gate has never
modelled that (a ``Ne`` guard is not an ``Eq``, so the ``guards`` source skips
it too), and the repro below is deliberately hull-clean — 102 is not a product
of two integers both in [8, 32] — so nothing is excluded by admitting it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gen_worker import aot_mint  # noqa: E402


_SPEC = {"x": {2: torch.export.Dim("h", min=8, max=32),
               3: torch.export.Dim("w", min=8, max=32)}}
_DIMS = (aot_mint.DynamicDim("x", 2, 8, 32),
         aot_mint.DynamicDim("x", 3, 8, 32))


class _ProvenInequality(torch.nn.Module):
    """Asks a question the tracer can only answer by PROVING ``h*w != 102``.

    102 = 2*3*17 has no factorization with both factors in [8, 32], so the
    inequality holds for every shape the declaration admits — the same
    situation as z-image's ``Mod(1, s18*s57) != 0``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, _c, h, w = x.shape
        if h * w != 102:
            return x.flatten(2).sum(-1)
        return x.flatten(2).mean(-1)


class _GenuinePin(torch.nn.Module):
    """The ie#566 G3 true positive: a static input length fixes ``h*w``."""

    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.reshape(b, c, h * w) + tokens.reshape(1, 1, -1)


def _proven_inequality_program() -> object:
    return torch.export.export(
        _ProvenInequality().eval(), (torch.randn(2, 4, 16, 24),), {},
        dynamic_shapes=_SPEC, strict=True)


def _genuine_pin_program() -> object:
    h, w = 16, 24
    return torch.export.export(
        _GenuinePin().eval(),
        (torch.randn(2, 4, h, w), torch.randn(h * w)), {},
        dynamic_shapes={"x": _SPEC["x"], "tokens": None}, strict=True)


def test_the_reproduction_really_is_a_REFUTED_axiom() -> None:
    """Guard the repro itself: if torch stops writing the refuted mirror the
    test below would pass vacuously, which is the wrong kind of green."""
    sympy = pytest.importorskip("sympy")
    env = aot_mint._shape_env(_proven_inequality_program())
    guards = [str(getattr(g, "expr", g)) for g in getattr(env, "guards", ())]
    assert guards == ["Ne(s37*s46, 102)"], guards
    axioms = {str(k): v for k, v in (getattr(env, "axioms", {}) or {}).items()}
    assert axioms["Eq(s37*s46, 102)"] is sympy.false, axioms
    assert axioms["Eq(102, s37*s46)"] is sympy.false, axioms


def test_a_proven_inequality_is_NOT_reported_as_a_pin() -> None:
    """The z-image residue, in twelve lines: the graph PROVED the relation
    does not hold, and the gate refused the mint for it anyway."""
    gaps = aot_mint.declared_range_gaps(_proven_inequality_program(), _DIMS)
    assert gaps == [], (
        "an axiom recorded as sympy.false is a refutation, not a pin: "
        f"{gaps!r}")


def test_a_genuine_pin_is_STILL_refused() -> None:
    """The primary control — ie#637's pre-fix arm, whose real pin arrives as a
    TRUE axiom. It must keep refusing, and now says so ONCE."""
    gaps = aot_mint.declared_range_gaps(_genuine_pin_program(), _DIMS)
    assert len(gaps) == 1, f"one relation, one refusal: {gaps!r}"
    assert "PINS" in gaps[0] and "Eq(s37*s46, 384)" in gaps[0]


def test_a_true_axiom_refuses_even_with_the_guards_source_empty() -> None:
    """The value filter must not amount to dropping ``axioms``: a pin present
    only as a TRUE axiom (ie#637's ``Eq(PythonMod(-s18*s57, 32), 0) => True``)
    still has to refuse."""
    program = _genuine_pin_program()
    env = aot_mint._shape_env(program)
    env.guards = []
    gaps = aot_mint.declared_range_gaps(program, _DIMS)
    assert len(gaps) == 1 and "PINS" in gaps[0], gaps


def test_an_unreadable_axiom_value_stays_refused() -> None:
    """Fail-closed, the same direction ``_is_tautology`` takes: only a
    recognised false admits."""
    program = _proven_inequality_program()
    env = aot_mint._shape_env(program)
    env.axioms = {key: object() for key in env.axioms}
    gaps = aot_mint.declared_range_gaps(program, _DIMS)
    assert gaps and "PINS" in gaps[0], gaps


def test_refuted_and_unreadable_values_discriminate_directly() -> None:
    sympy = pytest.importorskip("sympy")
    assert aot_mint._refuted(sympy.false)
    assert aot_mint._refuted(False)
    assert not aot_mint._refuted(sympy.true)
    assert not aot_mint._refuted(True)
    assert not aot_mint._refuted(object())
    assert not aot_mint._refuted(None)
