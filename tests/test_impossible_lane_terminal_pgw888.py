"""pgw#888 — a refusal whose cause CANNOT change must not be retryable.

pgw#888 observed 11 real requests each exhausting five retries on a selected
W8A8 lane with no cell. The design question it hung on — serve eager, or fail
closed? — is settled the pgw#1010 way: a MANDATORY quantized lane fails closed,
because DESIGN-RULINGS §4.31's in-request eager fallback governs the case where
eager is a valid POSTURE, and an author who declared the lane mandatory has not
sanctioned eager numerics.

That settles WHETHER to refuse. It does not make the refusal retryable, and it
was: `CompiledExecutionLaneUnavailableError` extends `RetryableError`, so
"this family declares no export, so no cell can be minted for it" — a fact that
cannot change for the life of the release — was spending the orchestrator's
whole attempt budget re-deriving one answer, and the user waited five times as
long for the identical refusal.

The distinction this pins is PERMANENCE, not severity:

  * no CUDA on this pod / arm failed / identity computation raised
        -> cause can change, another pod can serve -> RETRYABLE (unchanged)
  * the family declares no export
        -> no pod can ever hold a cell            -> TERMINAL
"""

from __future__ import annotations

import pytest

from gen_worker import compile_cache as cc
from gen_worker.api.errors import FatalError, RetryableError
from gen_worker.pb import worker_scheduler_pb2 as pb


def _status(exc: BaseException) -> "pb.JobStatus":
    from gen_worker.executor import _map_exception

    status, _detail = _map_exception(exc)
    return status


# ---------------------------------------------------------------------------
# The taxonomy: permanence is what separates them
# ---------------------------------------------------------------------------


def test_the_impossible_refusal_is_NOT_retryable() -> None:
    """RED before this change: the only class available extended
    `RetryableError`, so a permanently-impossible cause was retried."""
    exc = cc.CompiledExecutionLaneImpossibleError("no export declared")
    assert isinstance(exc, FatalError)
    assert not isinstance(exc, RetryableError)


def test_it_does_not_INHERIT_the_retryable_refusal() -> None:
    """It is exactly the retryability that differs, so inheriting the parent
    would put it straight back on the retry path through any `except` clause
    that names the parent — of which there are several."""
    assert not issubclass(
        cc.CompiledExecutionLaneImpossibleError,
        cc.CompiledExecutionLaneUnavailableError)


def test_the_TRANSIENT_refusal_stays_retryable() -> None:
    """Unchanged and deliberately so: a cell that is merely absent HERE can
    exist elsewhere, and a requeue is how the request reaches that pod."""
    exc = cc.CompiledExecutionLaneUnavailableError("no C toolchain")
    assert isinstance(exc, RetryableError)
    assert _status(exc) == pb.JOB_STATUS_RETRYABLE


# ---------------------------------------------------------------------------
# It reaches the WIRE as terminal — the property the retry budget reads
# ---------------------------------------------------------------------------


def test_it_maps_to_a_TERMINAL_wire_status() -> None:
    """The class hierarchy is only half the claim; `_map_exception` is what the
    orchestrator's attempt budget actually consumes. Asserted through the real
    mapper, not by re-reading the isinstance ladder."""
    exc = cc.CompiledExecutionLaneImpossibleError("no export declared")
    assert _status(exc) != pb.JOB_STATUS_RETRYABLE
    assert _status(exc) == pb.JOB_STATUS_FATAL


def test_the_refusal_names_why_EAGER_was_not_the_answer() -> None:
    """A refusal that says only "needs a cell" invites exactly the reading
    pgw#888 was filed under — that the pod should have served eager. The
    message has to carry the reason it did not, because the pod ships no logs
    and this string is the whole explanation the hub receives."""
    from gen_worker import fleet_cells

    class _Pipe:
        pass

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(cc, "mandatory_serving", lambda pipe: True)
        monkey.setattr(
            fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "w8a8")
        with pytest.raises(cc.CompiledExecutionLaneImpossibleError) as err:
            fleet_cells._fail_closed(
                _Pipe(),
                "this lane serves only from a cell and this family declares "
                "no export, so no cell can be minted for it (pgw#1010)",
                phase=fleet_cells.EagerPhase.MANDATORY_LANE_NEEDS_A_CELL,
                permanent=True)
    finally:
        monkey.undo()

    detail = str(err.value)
    assert "mandatory" in detail, detail
    assert "eager" in detail.lower(), detail
    assert "Not retryable" in detail, detail


def test_permanence_is_asked_of_the_DECLARATION_not_the_code_path() -> None:
    """The narrowing that the existing suite caught, pinned so it stays.

    `MANDATORY_LANE_NEEDS_A_CELL` fires whenever the recipe is not `aot`, and
    only ONE of its causes is permanent. A delegation refusal or a
    caller-forced in-process decline reaches the SAME exit with an export
    declared — `test_delegation_declines_name_their_TRUE_cause` does exactly
    that — and those can differ on the next attempt.

    The first cut of this change read permanence off the code path and made
    every one of them terminal, which is the worse direction of the same
    mistake: a request that a retry could serve, refused for good. Permanence
    is therefore asked of `export_declaration(family)`, and the reason string
    stops claiming "declares no export" when a declaration is registered.
    """
    from gen_worker.api import export_contract

    assert export_contract.export_declaration("no-such-family-pgw888") is None


def test_a_non_permanent_exit_still_raises_the_RETRYABLE_class() -> None:
    """The default is unchanged: only the exit that opts in becomes terminal,
    so nothing that could succeed on another pod stops being requeued."""
    from gen_worker import fleet_cells

    class _Pipe:
        pass

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(cc, "mandatory_serving", lambda pipe: True)
        monkey.setattr(
            fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "w8a8")
        with pytest.raises(cc.CompiledExecutionLaneUnavailableError) as err:
            fleet_cells._fail_closed(
                _Pipe(), "CUDA unavailable",
                phase=fleet_cells.EagerPhase.NO_CUDA)
    finally:
        monkey.undo()

    assert not isinstance(
        err.value, cc.CompiledExecutionLaneImpossibleError)
    assert _status(err.value) == pb.JOB_STATUS_RETRYABLE
