"""The does-it-run proof runs where the weights already are.

§4.33 prices a mint on "the model is fully resident and serving eager" plus a
weight-free compile. A does-it-run proof inside `mint_child` breaks that: it
materialises REAL random values for every virtual parameter — one full
checkpoint at compute dtype, in a child process holding none, concurrently with
the parent's resident copy (wan-2.2 on an H100-80: 56.2 GB against 15.5 GiB
free, so `CUDA out of memory` -> `delegated_no_cell` -> `boot_ended_uncompiled`).

§4.33 steps 4-5 put verification on the LIVE pipeline, against weights that are
resident and paid for once. This file fences that from both ends:

* the CHILD may not prove it (and may not mint without one), because proving it
  there is the allocation;
* the PARENT records it from the forward it already runs, and cozy-local — whose
  mint is opened from inside the slot load, before `setup()` has bound the
  pipeline — defers the mint until a handler exists to run.

WHAT IS DELIBERATELY NOT WEAKENED
---------------------------------
A cell must not seal for a handler that cannot serve. A missing proof REFUSES
the mint. It never degrades to "mint anyway".
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gen_worker import handler_proof
from gen_worker import mint_supervisor

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
# The resident parent gate
# ---------------------------------------------------------------------------


def _task(tmp_path: Path, **kw: Any) -> mint_supervisor.MintTask:
    pending = type("_Pending", (), {
        "family": "micro-diffusion",
        "arm_token": "cg-key-v1-" + "a" * 56,
        "mint_root": tmp_path,
        "cfg": type("_Cfg", (), {
            "shapes": (),
            "targets": ("transformer",),
            "family": "micro-diffusion",
            "lora_bucket": 0,
            "guidance_scales": (),
            "text_lens": (),
        })(),
    })()
    base: Dict[str, Any] = dict(
        pending=pending,
        pipe=object(),
        function="generate",
        modules=("m",),
    )
    base.update(kw)
    return mint_supervisor.MintTask(**base)


def test_the_task_carries_the_resident_parents_proof(tmp_path: Path) -> None:
    proof = "boot warm forward 'generate' (real weights)"
    assert _task(tmp_path, handler_proof=proof).handler_proof == proof


def test_the_task_defaults_to_unproven(tmp_path: Path) -> None:
    assert _task(tmp_path).handler_proof == ""


def test_the_supervisor_refuses_before_spawning_without_a_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker import fleet_cells

    abandoned: List[Any] = []
    monkeypatch.setattr(fleet_cells, "abandon_self_mint", abandoned.append)
    monkeypatch.setattr(mint_supervisor, "_emit_abort", lambda **_kw: None)

    result = asyncio.run(mint_supervisor.supervise(_task(tmp_path), act=object()))

    assert result.status == mint_supervisor.FAILED
    assert result.reason == "handler_unproven"
    assert "no handler proof" in result.detail
    assert abandoned
    assert not (tmp_path / mint_supervisor.GRAPH_DIRNAME).exists()


def test_a_proof_reaches_the_next_parent_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker import fleet_cells

    monkeypatch.setattr(fleet_cells, "abandon_self_mint", lambda _pending: None)
    monkeypatch.setattr(mint_supervisor, "_emit_abort", lambda **_kw: None)
    monkeypatch.setattr(
        mint_supervisor,
        "assert_family_mintable",
        lambda _family: (_ for _ in ()).throw(
            mint_supervisor.DeclaredBlockerRefusal("next gate")),
    )

    result = asyncio.run(mint_supervisor.supervise(
        _task(
            tmp_path,
            handler_proof="boot warm forward 'generate' (real weights)",
        ),
        act=object(),
    ))

    assert result.reason == "declared_blocker"
    assert result.detail == "next gate"


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
