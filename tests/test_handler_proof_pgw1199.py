"""pgw#1199: the does-it-run proof runs where the weights already are.

§4.33 said a mint costs ~8 GiB because *"the model is fully resident and serving
eager"* and *"compile weight-free … VRAM cost is negligible"*. The compile is.
The MINT was not: `mint_child` ran pgw#984's does-it-run proof by materialising
REAL random values for every virtual parameter — one full checkpoint at compute
dtype, in a child process holding none, concurrently with the parent's resident
copy. On pod `729431an6ugbvq` (H100-80, wan-2.2) that is 56.2 GB against 15.5
GiB free: `CUDA out of memory`, twice, then `delegated_no_cell`, then
`boot_ended_uncompiled`. The ~8 GiB figure was measuring that same walk for an
sdxl-sized family.

§4.33 steps 4-5 already put verification on the LIVE pipeline — *"already
running eager"* — against weights that are resident and paid for once. So the
proof was in the wrong process, and this file fences the correction from both
ends:

* the CHILD may not prove it (and may not mint without one), because proving it
  there is the allocation;
* the PARENT records it from the forward it already runs, and cozy-local — whose
  mint is opened from inside the slot load, before `setup()` has bound the
  pipeline — defers the mint until a handler exists to run.

WHAT IS DELIBERATELY NOT WEAKENED
---------------------------------
pgw#984's sentence is unchanged: *a cell must not seal for a handler that cannot
serve*. A missing proof REFUSES the mint. It never degrades to "mint anyway".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gen_worker import handler_proof
from gen_worker import mint_delegate, mint_process

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"
if str(MICRO_SRC) not in sys.path:
    sys.path.insert(0, str(MICRO_SRC))


@pytest.fixture(autouse=True)
def _clean_ledger() -> Any:
    handler_proof._forget()
    yield
    handler_proof._forget()


# ---------------------------------------------------------------------------
# The ledger — a claim about THIS process and this function
# ---------------------------------------------------------------------------


def test_a_function_is_unproven_until_its_handler_has_run() -> None:
    assert handler_proof.provenance("generate") == ""
    assert not handler_proof.proven("generate")

    handler_proof.record("generate", "boot warm forward 'generate' (real weights)")

    assert handler_proof.proven("generate")
    assert "real weights" in handler_proof.provenance("generate")


def test_the_proof_is_PROVENANCE_and_not_a_boolean() -> None:
    """The child's report has to be able to say WHICH proof stood behind the
    cell it sealed. A bare True would make "proven" unauditable across the
    process boundary — the child cannot see the parent's ledger, only what the
    request carries."""
    handler_proof.record("generate", "boot warm forward 'generate' (real weights)")
    assert isinstance(handler_proof.provenance("generate"), str)
    assert handler_proof.provenance("generate").strip()


def test_the_first_proof_wins_and_a_later_one_does_not_overwrite_it() -> None:
    handler_proof.record("generate", "boot warm forward (real weights)")
    handler_proof.record("generate", "something later and weaker")
    assert handler_proof.provenance("generate") == "boot warm forward (real weights)"


def test_functions_are_proven_INDIVIDUALLY() -> None:
    """A pod serves several handlers off one pipeline and mints per function.
    Proving one must not vouch for another — that is how a broken sibling
    handler would seal a cell."""
    handler_proof.record("generate", "boot warm forward (real weights)")
    assert not handler_proof.proven("generate_turbo")


# ---------------------------------------------------------------------------
# The wire — the parent declares it, the request carries it
# ---------------------------------------------------------------------------


def _task(**kw: Any) -> mint_delegate.MintTask:
    pending = type("_Pending", (), {
        "family": "micro-diffusion", "arm_token": "arm1-x",
        "mint_root": "/tmp", "cfg": type("_Cfg", (), {
            "shapes": (), "targets": ("transformer",), "family": "micro-diffusion",
            "lora_bucket": 0, "guidance_scales": (), "text_lens": (),
        })(),
    })()
    base: Dict[str, Any] = dict(
        pending=pending, pipe=object(), function="generate", modules=("m",))
    base.update(kw)
    return mint_delegate.MintTask(**base)


def test_the_request_carries_the_parents_proof(tmp_path: Path) -> None:
    task = _task(handler_proof="boot warm forward 'generate' (real weights)")
    request = mint_delegate.build_request(task, workdir=tmp_path)
    assert request.handler_proof == "boot warm forward 'generate' (real weights)"


def test_an_unproven_parent_sends_an_EMPTY_proof_rather_than_a_guess(
    tmp_path: Path,
) -> None:
    """An absent measurement is no evidence — never a silent pass. A parent
    that cannot say its handler ran says nothing, and the child refuses."""
    request = mint_delegate.build_request(_task(), workdir=tmp_path)
    assert request.handler_proof == ""


def test_the_request_field_defaults_to_unproven() -> None:
    """A field that defaulted to "proven" would make every caller that forgot
    to set it silently satisfy pgw#984."""
    assert mint_process.MintRequest.__struct_defaults__ is not None
    request = mint_process.MintRequest(
        function="generate", modules=(), family="f", arm_token="a",
        target="t", work_root="w", report="r",
        cfg=mint_process.CompileCellSpec())
    assert request.handler_proof == ""


# ---------------------------------------------------------------------------
# The child — refuses rather than re-proving, because re-proving IS the cost
# ---------------------------------------------------------------------------


def _request(**kw: Any) -> mint_process.MintRequest:
    base: Dict[str, Any] = dict(
        function="generate", modules=("m",), family="micro-diffusion",
        arm_token="arm1-x", target="t", work_root="w", report="r",
        cfg=mint_process.CompileCellSpec(targets=("transformer",)))
    base.update(kw)
    return mint_process.MintRequest(**base)


def test_the_child_refuses_a_weight_free_mint_with_no_proof() -> None:
    """The whole of pgw#1199 in one row.

    A weight-free mint whose parent sent no proof must REFUSE. The tempting
    alternative — prove it here — is the 56.2 GB allocation that killed
    wan-2.2; the other — mint anyway — drops pgw#984.
    """
    from gen_worker import mint_child

    with pytest.raises(mint_child.MintChildRefused) as caught:
        mint_child.assert_handler_proven(_request())

    said = str(caught.value)
    assert "no handler proof" in said
    assert "handler_proof" in said, "the message must name what to set"


def test_the_child_accepts_a_mint_whose_parent_DID_prove_the_handler() -> None:
    from gen_worker import mint_child

    mint_child.assert_handler_proven(
        _request(handler_proof="boot warm forward 'generate' (real weights)"))


def test_whitespace_is_not_a_proof() -> None:
    """A caller that sets the field to something empty-but-truthy has not run
    a forward, and must be refused exactly as one that set nothing."""
    from gen_worker import mint_child

    with pytest.raises(mint_child.MintChildRefused):
        mint_child.assert_handler_proven(_request(handler_proof="   "))


def test_the_child_still_proves_on_the_REAL_WEIGHT_fallback() -> None:
    """Not every family has a structure-only path (a quantized artifact lane,
    a class with no config surface). Those mints load the checkpoint into the
    child anyway, so the proof costs nothing extra there and must NOT have been
    deleted with the weight-free one — otherwise pgw#1199 would have removed a
    guarantee instead of relocating it."""
    from gen_worker import mint_child

    source = Path(mint_child.__file__).read_text()
    assert "_drive_warm_plan(" in source
    assert "proof_only=True" in source


def test_the_structure_only_module_offers_no_route_to_real_values() -> None:
    from gen_worker.models import structure_only

    for gone in ("materialize_random", "restore_virtual", "stray_real_tensors",
                 "proof_cost_bytes"):
        assert not hasattr(structure_only, gone), gone


# ---------------------------------------------------------------------------
# cozy-local — the mint waits for a handler to exist
# ---------------------------------------------------------------------------


def test_a_local_mint_is_DEFERRED_rather_than_run_inside_the_slot_load() -> None:
    """cozy-local reaches `enable_compiled` from inside the SLOT LOAD, before
    `setup()` has bound the pipeline to the endpoint instance — so at that
    moment there is no handler to run, and a mint started there could only ever
    have proven itself the expensive way.

    Deferring is not a workaround: it is the order the fleet already boots in
    (pgw#671 eager-first — setup completes, the handler warms, THEN the
    background mint).
    """
    from gen_worker import local_serve

    deferred: List[Any] = []
    pending = object.__new__(local_serve.fleet_cells.PendingSelfMint)
    outcome = type("_Outcome", (), {"armed": False, "self_mint": pending})()
    ctx = local_serve.LocalMintContext(
        function="generate", modules=("m",), slots={})

    armed = local_serve.enable_compiled.__wrapped__ if hasattr(
        local_serve.enable_compiled, "__wrapped__") else None
    assert armed is None  # not decorated; the call below is the real one

    # Drive the branch directly: a pending nobody deferred would be minted
    # in-line, which is what this must not do.
    import gen_worker.fleet_cells as fleet_cells

    original = fleet_cells.enable_compiled
    try:
        fleet_cells.enable_compiled = lambda *a, **k: outcome  # type: ignore[assignment]
        result = local_serve.enable_compiled(
            object(), object(), None, mint=ctx, defer=deferred)
    finally:
        fleet_cells.enable_compiled = original  # type: ignore[assignment]

    assert result is False, "a deferred mint has not armed anything yet"
    assert len(deferred) == 1
    assert deferred[0].pending is pending
    assert deferred[0].mint is ctx


def test_run_deferred_refuses_the_mint_when_the_handler_does_not_run() -> None:
    """pgw#984's sentence, spoken on the parent: a handler that cannot serve
    mints nothing. Reached BEFORE a compile is paid for, which is strictly
    earlier than the child's version managed."""
    from gen_worker import local_serve

    abandoned: List[Any] = []
    minted: List[Any] = []
    pending = type("_P", (), {"family": "micro-diffusion"})()
    owed = local_serve.DeferredMint(
        pipe=object(), pending=pending,
        mint=local_serve.LocalMintContext(
            function="generate", modules=("m",), slots={}))

    def _boom(*_a: Any, **_k: Any) -> str:
        raise handler_proof.HandlerProofFailed("the handler raised TypeError")

    import gen_worker.fleet_cells as fleet_cells

    original_prove = handler_proof.prove
    original_abandon = fleet_cells.abandon_self_mint
    original_mint = local_serve._mint_here
    try:
        handler_proof.prove = _boom  # type: ignore[assignment]
        fleet_cells.abandon_self_mint = abandoned.append  # type: ignore[assignment]
        local_serve._mint_here = (  # type: ignore[assignment]
            lambda *a, **k: minted.append(a))
        local_serve.run_deferred([owed], instance=object(), specs=[])
    finally:
        handler_proof.prove = original_prove  # type: ignore[assignment]
        fleet_cells.abandon_self_mint = original_abandon  # type: ignore[assignment]
        local_serve._mint_here = original_mint  # type: ignore[assignment]

    assert minted == [], "a mint must not start behind a failed proof"
    assert abandoned == [pending], "and the obligation must end somewhere named"


def test_run_deferred_mints_once_the_handler_is_proven() -> None:
    from gen_worker import local_serve

    minted: List[Any] = []
    owed = local_serve.DeferredMint(
        pipe=object(), pending=type("_P", (), {"family": "micro-diffusion"})(),
        mint=local_serve.LocalMintContext(
            function="generate", modules=("m",), slots={}))

    original_prove = handler_proof.prove
    original_mint = local_serve._mint_here
    try:
        handler_proof.prove = (  # type: ignore[assignment]
            lambda *a, **k: "resident warm forward 'generate' (real weights)")
        local_serve._mint_here = (  # type: ignore[assignment]
            lambda *a, **k: minted.append(a))
        local_serve.run_deferred([owed], instance=object(), specs=[])
    finally:
        handler_proof.prove = original_prove  # type: ignore[assignment]
        local_serve._mint_here = original_mint  # type: ignore[assignment]

    assert len(minted) == 1


def test_prove_is_idempotent_so_the_fleet_path_pays_nothing() -> None:
    """The executor's boot warm plan has already run every declared handler by
    the time a mint is delegated, so `prove` must be a no-op there rather than
    a second forward."""
    calls: List[Any] = []
    handler_proof.record("generate", "boot warm forward (real weights)")

    original = handler_proof.run_warm_job
    try:
        handler_proof.run_warm_job = (  # type: ignore[assignment]
            lambda *a, **k: calls.append(a))
        how = handler_proof.prove(object(), [], "generate")
    finally:
        handler_proof.run_warm_job = original  # type: ignore[assignment]

    assert calls == [], "an already-proven handler must not be re-run"
    assert how == "boot warm forward (real weights)"
