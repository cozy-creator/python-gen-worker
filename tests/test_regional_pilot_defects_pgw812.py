"""pgw#812 D1 + D2 — the two defects that block flux2's mint outright.

Both were found on CPU for $0 by the regional-compilation pilot's dry run,
before any pod was rented, and both are REAL-EXPORT reproductions here: every
test below drives ``torch.export`` over a module whose traced algebra matches
the one flux2 produces, then asserts on the shipped mint functions. Nothing is
stubbed except the direct :func:`_is_tautology` discrimination check, which
exists because a genuinely-not-always-true ``Mod`` guard cannot reach the gate
at all (export refuses it first — see
``test_D2_a_non_tautological_mod_never_even_reaches_the_gate``).

D1  ``dynamic_shapes_spec`` minted one torch symbol per (input, axis), so a
    declared ``Dim`` with two carriers became two INDEPENDENT symbols and
    strict export died. flux2 declares
    ``Dim("T_img", carried_by=(("hidden_states", 1), ("img_ids", 1)))``
    deliberately, so the most careful declaration in the fleet was the one
    that could not mint. ie#571 recorded it "READY — no open mint blockers".

D2  the ie#566 G3 range gate refused flux2 on
    ``Eq(Mod(3072*s + 1572864, 48*s + 24576), 0)``, which is
    ``Mod(64*X, X) == 0`` — identically true. A gate that cannot tell "pinned"
    from "trivially true" refuses correct mints.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gen_worker import aot_declaration, aot_mint  # noqa: E402
from gen_worker.api.decorators import (  # noqa: E402
    Compile, Dim, GraphClass,
)


# ---------------------------------------------------------------- D1

class _TwoCarrier(torch.nn.Module):
    """The flux2 shape: ONE logical token axis carried by TWO inputs.

    ``hidden_states[1]`` and ``img_ids[1]`` must always be equal — the edit
    lane grows both together (``torch.cat`` on the image-token axis for the
    latents and the matching ``img_ids`` concat) — and the forward here makes
    that a hard requirement rather than a coincidence, exactly as flux2's
    rope construction does.
    """

    def forward(self, hidden_states: torch.Tensor,
                img_ids: torch.Tensor) -> torch.Tensor:
        return hidden_states + img_ids.sum(-1, keepdim=True)


def _two_carrier_declaration() -> Compile:
    """flux2's binding, minimised: one Dim, two carriers, one collapsed class."""
    return Compile(
        family="two-carrier",
        targets=("transformer",),
        shape_strategy="dynamic-collapse",
        dims=(Dim("T_img", carried_by=(("hidden_states", 1), ("img_ids", 1))),),
        classes=(
            GraphClass(dims={"T_img": 4096}),
            GraphClass(dims={"T_img": 8160}),
        ),
    )


def _rows_for(decl: Compile) -> tuple:
    plans = aot_declaration.mint_plans(decl, "transformer")
    assert len(plans) == 1, "dynamic-collapse must produce ONE plan"
    return plans[0].rows


def test_D1_a_multi_carrier_dim_mints_ONE_shared_symbol() -> None:
    decl = _two_carrier_declaration()
    dims = aot_declaration.derived_dynamic(decl, "transformer", _rows_for(decl))
    assert len(dims) == 2, "one row per carrier"
    assert {d.input_name for d in dims} == {"hidden_states", "img_ids"}
    assert {d.dim for d in dims} == {"T_img"}, (
        "each row must carry the DECLARED dim name — that name is the only "
        "thing that can tell two carriers of one axis from two axes")

    spec = aot_mint.dynamic_shapes_spec(dims, ("hidden_states", "img_ids"))
    left = spec["hidden_states"][1]
    right = spec["img_ids"][1]
    assert left is right, (
        "both carriers must be the SAME torch symbol object; two symbols is "
        "the D1 defect and strict export refuses it")


def test_D1_the_declaration_exports_STRICT() -> None:
    """The regression test at the DECLARATION level, per pgw#812's own note.

    Asserting on ``dynamic_shapes_spec`` alone would let the next rewrite
    reintroduce the defect one layer up, so this runs the whole bridge —
    declaration -> mint_plans -> derived_dynamic -> dynamic_shapes_spec ->
    ``export_program(strict=True)`` — and demands a real export.
    """
    decl = _two_carrier_declaration()
    dims = aot_declaration.derived_dynamic(decl, "transformer", _rows_for(decl))
    spec = aot_mint.dynamic_shapes_spec(dims, ("hidden_states", "img_ids"))

    program = aot_mint.export_program(
        _TwoCarrier().eval(),
        (torch.randn(1, 4096, 8), torch.randn(1, 4096, 3)), {},
        dynamic_shapes=spec, strict=True)

    # The axis is genuinely symbolic and genuinely SHARED: one symbol governs
    # both placeholders, which is the property the declaration is asserting.
    shapes = [tuple(n.meta["val"].shape) for n in program.graph.nodes
              if n.op == "placeholder" and n.meta.get("val") is not None]
    assert len(shapes) == 2
    assert str(shapes[0][1]) == str(shapes[1][1]), shapes
    assert not str(shapes[0][1]).isdigit(), "the axis must not specialize"


def test_D1_independent_axes_of_one_input_still_get_their_own_symbols() -> None:
    """The empty-name fallback, which the hand-registered builder path needs.

    ``aot_inputs.latent_hw_dims`` emits latent H and W as two rows of ONE
    input with no declared name. They are genuinely independent axes and
    sharing a symbol would pin every artifact to square latents.
    """
    dims = (aot_mint.DynamicDim("sample", 2, 64, 160, multiple_of=8),
            aot_mint.DynamicDim("sample", 3, 64, 160, multiple_of=8))
    assert all(d.dim == "" for d in dims)
    spec = aot_mint.dynamic_shapes_spec(dims, ("sample",))
    assert str(spec["sample"][2]) != str(spec["sample"][3]), (
        "unnamed rows must keep one symbol each")


def test_D1_same_name_different_bounds_does_not_silently_widen() -> None:
    """Sharing is keyed on (name, multiple_of, min, max), not name alone."""
    dims = (aot_mint.DynamicDim("a", 1, 8, 64, dim="T"),
            aot_mint.DynamicDim("b", 1, 8, 128, dim="T"))
    spec = aot_mint.dynamic_shapes_spec(dims, ("a", "b"))
    assert spec["a"][1] is not spec["b"][1]


# ---------------------------------------------------------------- D2

class _VacuousMod(torch.nn.Module):
    """flux2's guard, reproduced in twelve lines and no weights.

    The image tokens are concatenated with a fixed-length text stream and the
    result is unflattened into heads and flattened back — attention's own
    shape algebra. That records ``Eq(Mod(3072*N, 48*N), 0)`` where ``N`` is
    the concatenated length: identically true, because ``3072 = 64 * 48``.
    """

    def forward(self, hidden_states: torch.Tensor,
                txt: torch.Tensor) -> torch.Tensor:
        x = torch.cat([txt, hidden_states], dim=1)
        n = x.shape[1]
        return x.reshape(1, 48 * n, 64).reshape(1, n, 3072)


def _vacuous_mod_program(lo: int = 64, hi: int = 4096):
    dims = (aot_mint.DynamicDim("hidden_states", 1, lo, hi, dim="T_img"),)
    spec = aot_mint.dynamic_shapes_spec(dims, ("hidden_states", "txt"))
    program = aot_mint.export_program(
        _VacuousMod().eval(),
        (torch.randn(1, 1024, 3072), torch.randn(1, 512, 3072)), {},
        dynamic_shapes=spec, strict=False)
    return program, dims


def test_D2_the_recorded_guard_really_is_the_flux2_one() -> None:
    """Guard the reproduction itself: if torch stops emitting this shape the
    test below would pass vacuously, which is the wrong kind of green."""
    program, _dims = _vacuous_mod_program()
    env = aot_mint._shape_env(program)
    guards = [str(getattr(g, "expr", g)) for g in getattr(env, "guards", ())]
    assert any(text.startswith("Eq(Mod(3072*") and "48*" in text
               for text in guards), guards


def test_D2_a_vacuous_mod_guard_is_ADMITTED() -> None:
    program, dims = _vacuous_mod_program()
    gaps = aot_mint.declared_range_gaps(program, dims)
    assert gaps == [], (
        "Mod(64*X, X) == 0 pins nothing; refusing it blocks flux2's mint "
        f"for a false reason: {gaps!r}")


def test_D2_a_genuine_pin_is_STILL_refused() -> None:
    """The ie#566 G3 true positive must survive the fix.

    ``reshape(b, c, h*w)`` against a static-length input forces the tracer to
    record ``Eq(h*w, N)`` — a constant on one side, unprovable as a tautology,
    so the gate must keep refusing it.
    """
    class GenuineDep(torch.nn.Module):
        def forward(self, x: torch.Tensor,
                    tokens: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            return x.reshape(b, c, h * w) + tokens.reshape(1, 1, -1)

    h, w = 16, 24
    program = torch.export.export(
        GenuineDep().eval(), (torch.randn(2, 4, h, w), torch.randn(h * w)), {},
        dynamic_shapes={"x": {2: torch.export.Dim("h", min=8, max=32),
                              3: torch.export.Dim("w", min=8, max=32)},
                        "tokens": None},
        strict=True)
    dims = (aot_mint.DynamicDim("x", 2, 8, 32),
            aot_mint.DynamicDim("x", 3, 8, 32))
    gaps = aot_mint.declared_range_gaps(program, dims)
    assert gaps and "PINS" in " ".join(gaps)


def test_D2_tautology_check_discriminates_on_mod() -> None:
    """Only a PROOF admits — an unprovable Mod stays refused.

    This is the one non-export assertion in the file, and it is here because
    a ``Mod`` guard that is not identically true cannot be exported at all
    (next test). Without it, "the gate still refuses a shape-dependent Mod"
    would be untested rather than merely unreachable.
    """
    import sympy

    from torch.utils._sympy.functions import Mod as TorchMod

    s = sympy.Symbol("s50", integer=True, positive=True)
    vacuous = sympy.Eq(TorchMod(3072 * s + 1572864, 48 * s + 24576), 0)
    genuine = sympy.Eq(TorchMod(384 * s, 5), 0)
    assert aot_mint._is_tautology(vacuous) is True
    assert aot_mint._is_tautology(genuine) is False


def test_D2_a_non_tautological_mod_never_even_reaches_the_gate() -> None:
    """Measured on this toolchain: export refuses first.

    Recorded so the scope of D2 is not over-read. A ``Mod`` guard that does
    not hold over the whole declared range is a CONSTRAINT VIOLATION at
    export time, so the only ``Mod`` equalities that can reach
    ``declared_range_gaps`` are ones the tracer already proved.
    """
    class NeedsMultipleOfFive(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.reshape(1, 5, -1)

    with pytest.raises(Exception) as excinfo:
        torch.export.export(
            NeedsMultipleOfFive().eval(), (torch.randn(1, 1000, 384),), {},
            dynamic_shapes={"x": {1: torch.export.Dim("T", min=10, max=4000)}},
            strict=False)
    assert "Constraints violated" in str(excinfo.value)
