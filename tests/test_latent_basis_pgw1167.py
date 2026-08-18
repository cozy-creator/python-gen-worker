"""The declared latent divisor is reconciled against the pipeline.

The class rows are derived by dividing declared PIXEL shapes by a latent
divisor the author passes to `derive.cfg_image_classes(latent_scale=…)`. An
unchecked divisor produces a whole compiled graph of correctly-shaped, permanently
unusable artifacts — silent, and paid for at full mint price.

WHY IT IS CHECKED IN THE COMPILE CHILD AND NOWHERE EARLIER

* Not at declaration time: `Compile` is built at endpoint import, no pipeline
  exists, and a latent divisor is a claim about the CHECKPOINT.
* Not as a derivation: §4.27/pgw#1089 requires `ck1` to derive from CODE ALONE,
  on fake/meta tensors, before any weight is resident, and the class rows feed
  the key. Reading the divisor off a loaded pipeline would make the key depend
  on runtime state and break boot-time adopt.
* Not at `aot_export_spec`: that seam has no declaration to reconcile. The
  compile child's `build_pipeline()` is the first current boundary where the
  loaded composition and declaration coexist, and it runs before the child
  hands either to the export loop.

THE CARRIER

The divisor is passed ONCE, to the deriver. `DerivedClasses` is a tuple
subclass that carries it out with the rows it produced, and
`Compile.__post_init__` transfers it to `Compile.latent_basis` BEFORE the row
coercion rebuilds `classes` as a plain tuple. Transport, then a compiled graph-level
home — never a second declaration by the author, and never a compiled graph-wide scalar
stamped on every row and read back off `rows[0]`: a label written beside a thing
cannot be told from one describing it.
"""

from __future__ import annotations

from gen_worker import aot_compile_child as child
from gen_worker.api import derive
from gen_worker.api.decorators import Compile
from gen_worker.api.export_contract import Dim, Fork
from gen_worker.child_preflight import PreflightRefused

_SHAPES = ((1024, 1024), (1152, 896))


def _decl(latent_scale: int = 8, *, derived: bool = True) -> Compile:
    rows = derive.cfg_image_classes(
        shapes=_SHAPES, latent_scale=latent_scale, text_len=77)
    return Compile(
        family="pgw1167-fam", targets=("transformer",), text_len=77,
        dims=(Dim("B", carried_by=(("x", 0),)),
              Dim("H_lat", carried_by=(("x", 2),)),
              Dim("W_lat", carried_by=(("x", 3),)),
              Dim("T_txt", carried_by=(("y", 1),))),
        forks=(Fork("cfg", served=(True, False)),),
        classes=rows if derived else tuple(rows),
        shape_strategy="static-rows", warm_changes_key=False)


class _Vae:
    def __init__(self, blocks: int) -> None:
        self.config = type("Cfg", (), {"block_out_channels": [0] * blocks})()


class _Pipe:
    """A pipeline whose divisor is real, or absent entirely."""

    def __init__(self, basis: int | None) -> None:
        if basis is None:
            self.vae = None
            # diffusers still exposes the attribute, defaulted — the trap.
            self.vae_scale_factor = 8
        else:
            self.vae = _Vae(1)
            self.vae_scale_factor = basis


# ---------------------------------------------------------------------------
# The carrier: declared once, transferred, digest-neutral
# ---------------------------------------------------------------------------


def test_the_divisor_rides_the_rows_and_lands_on_the_DECLARATION() -> None:
    """The author writes the number once, to the deriver."""
    rows = derive.cfg_image_classes(shapes=_SHAPES, latent_scale=8, text_len=77)
    assert rows.latent_basis == 8
    assert _decl(8).latent_basis == 8


def test_the_transport_is_a_drop_in_tuple() -> None:
    """Every existing call site keeps working — it IS the tuple they got."""
    rows = derive.cfg_image_classes(shapes=_SHAPES, latent_scale=8, text_len=77)
    assert isinstance(rows, tuple)
    assert tuple(rows) == tuple(iter(rows)) and len(rows) == 4


def test_classes_are_still_coerced_to_a_PLAIN_tuple() -> None:
    """The carrier is transport, not storage: nothing downstream reads it off
    `classes`, and the declaration keeps the exact type it always had."""
    assert type(_decl(8).classes) is tuple


def test_rows_from_a_NON_deriver_leave_the_basis_UNDECLARED() -> None:
    """`None` is the honest answer for rows nobody derived — it must not
    default to a number, which is the whole defect one level up."""
    assert _decl(8, derived=False).latent_basis is None


# ---------------------------------------------------------------------------
# THE FENCE: the carrier must never reach the contract digest
# ---------------------------------------------------------------------------


def test_the_carrier_is_ABSENT_from_the_contract_digest() -> None:
    """Mechanically fenced, because the failure mode is a fleet-wide re-key
    nobody notices until every pod re-mints.

    `latent_basis` is PROVENANCE — how the rows were computed — not a
    shape-contract axis; the latent extents it produced are already digested
    via `classes`. One line adding it to `contract_axes()` would silently
    re-key every compiled graph in the fleet.
    """
    assert "latent_basis" not in _decl(8).contract_axes()


def test_declaring_the_basis_changes_NO_existing_digest() -> None:
    """The same rows, with and without the carrier, digest identically — so
    every declaration that adopts the deriver keeps its compiled graphs."""
    assert _decl(8).contract_axes() == _decl(8, derived=False).contract_axes()


def test_a_DIFFERENT_basis_still_changes_the_digest_through_the_ROWS() -> None:
    """The fence must not be read as "the divisor does not matter": a
    different divisor produces different latent extents, and THOSE are
    digested. Only the provenance scalar is excluded."""
    assert _decl(8).contract_axes() != _decl(16).contract_axes()


# ---------------------------------------------------------------------------
# The reconciliation
# ---------------------------------------------------------------------------


def test_a_WRONG_divisor_refuses_before_the_export_is_paid_for() -> None:
    """RED on master: accepted, and the mint proceeded to build a whole compiled graph
    of correctly-shaped, unusable artifacts."""
    import pytest

    with pytest.raises(PreflightRefused, match="latent_basis_mismatch"):
        child._reconcile_latent_basis(_Pipe(16), _decl(8))


def test_the_refusal_blames_the_COMPOSITION_not_the_checkpoint() -> None:
    """A vae sourced from another release genuinely makes the declared latent
    extents wrong for that composition, so refusing is right — but the text
    must not send the next reader to the checkpoint's repo."""
    import pytest

    with pytest.raises(PreflightRefused) as err:
        child._reconcile_latent_basis(_Pipe(16), _decl(8))
    detail = str(err.value)
    assert "does not match this composition" in detail, detail
    assert "8" in detail and "16" in detail, detail
    assert "latent_scale=" in detail, detail


def test_a_MATCHING_divisor_reconciles() -> None:
    assert child._reconcile_latent_basis(_Pipe(8), _decl(8)) == \
        child._LATENT_RECONCILED


# ---------------------------------------------------------------------------
# UNOBSERVABLE is a state, not a pass — the `else 8` trap
# ---------------------------------------------------------------------------


def test_a_pipeline_with_NO_vae_is_UNRECONCILED_not_reconciled() -> None:
    """diffusers reports `vae_scale_factor == 8` when there is no vae — a
    default nobody chose, indistinguishable from a real observation of 8.
    Believing it would reproduce pgw#1058's silent dtype default one field
    over, so the verdict is named apart from a pass."""
    verdict = child._reconcile_latent_basis(_Pipe(None), _decl(8))
    assert verdict == child._LATENT_UNRECONCILED_NO_VAE
    assert verdict != child._LATENT_RECONCILED


def test_the_no_vae_case_does_not_accidentally_MATCH_a_declared_8() -> None:
    """The sharpest form of the trap: declared 8 against a defaulted 8 would
    look like a successful reconciliation while proving nothing."""
    assert _Pipe(None).vae_scale_factor == 8      # the default is really there
    assert child._observed_latent_basis(_Pipe(None)) is None


def test_an_UNDECLARED_basis_is_UNRECONCILED_not_reconciled() -> None:
    assert child._reconcile_latent_basis(_Pipe(8), _decl(8, derived=False)) == \
        child._LATENT_UNRECONCILED_UNDECLARED


# ---------------------------------------------------------------------------
# Delete-the-call-site: the reconciler must be WIRED, not merely present
# ---------------------------------------------------------------------------


def test_build_pipeline_actually_CALLS_the_reconciler() -> None:
    """A reconciliation that never fires is precisely the defect class this
    issue removes, so the wiring is asserted rather than assumed: delete the
    call from `build_pipeline()` and this row goes red.
    """
    import inspect

    source = inspect.getsource(child.build_pipeline)
    assert "_reconcile_latent_basis(pipeline, decl)" in source, (
        "build_pipeline() no longer calls the reconciler — the gate cannot "
        "fire")


def test_the_call_precedes_the_first_export() -> None:
    """Refusing after the export is paid for would keep the correctness and
    throw away the entire prize, which is that a wrong divisor costs seconds."""
    import inspect

    build = inspect.getsource(child.build_pipeline)
    call = build.index("_reconcile_latent_basis(pipeline, decl)")
    returned = build.index("return pipeline, spec, decl")
    run = inspect.getsource(child.run)
    setup = run.index("build_pipeline(job)")
    export = run.index("_trace_share(aot_mint, pipeline, spec, decl, job)")
    assert call < returned
    assert setup < export, "the reconciliation must run before export starts"


def test_the_gate_never_KILLS_a_mint_it_cannot_read() -> None:
    """It has exactly two outcomes: a NAMED refusal for a proven mismatch, and
    an explicit UNRECONCILED. Anything it cannot read is the latter.

    The first cut raised `AttributeError` from inside a correctness check and
    took an unrelated path down with it. A gate that can crash a compile it
    cannot judge is worse than the silence it replaced.
    """
    assert child._reconcile_latent_basis(None, None) == \
        child._LATENT_UNRECONCILED_UNDECLARED
    assert child._reconcile_latent_basis(_Pipe(8), None) == \
        child._LATENT_UNRECONCILED_UNDECLARED
