"""pgw#764 red-verify: a real boot sequence produces ordered phase rows.

The acceptance this file encodes:

* A boot emits an ORDERED series of typed ``BootPhase`` envelopes — the exact
  messages the stream sink sends, asserted as wired (the pgw#733/pgw#760 test
  convention), not through a double of the recorder API.
* Rows recorded BEFORE the transport exists are not lost. This is the whole
  reason the recorder buffers: ``activity.bind_sink`` is not called until
  ``Executor.ensure_setup``, i.e. after weights are on disk, so every phase of
  a cold boot precedes any sink. A recorder that only forwarded live rows would
  report its own tail and silently drop the expensive part.
* A REFUSED ARM appears with its classified reason: a real ``AdoptError``
  survives as ``outcome=refused reason=<token>`` — not as a failure, because
  the worker declined that cell deliberately and goes on serving eager. (The
  end-to-end assertion through ``aot_serve.enable`` itself lives in
  ``test_boot_phases_arm_pgw764.py``, which is held out of the commit while a
  sibling lane holds uncommitted WIP in ``aot_serve.py`` — see tracker
  pgw#764.)
* The phases RECONCILE: nested spans charge time to the child, so measured
  phases plus the residual equal the boot window (th#1111's rule, at boot
  scale). An instrument that does not close cannot say where the time went.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List

import pytest

from gen_worker import aot_serve, boot_phases, serving_mode
from gen_worker.compile_cache import AdoptError
from gen_worker.pb import worker_scheduler_pb2 as pb


@pytest.fixture(autouse=True)
def _reset() -> Any:
    boot_phases.reset_for_tests()
    yield
    boot_phases.reset_for_tests()


def _phases(rows: List[pb.BootPhase]) -> List[str]:
    return [r.phase for r in rows if r.terminal]


# ---------------------------------------------------------------------------
# ordering + the buffer-then-flush contract
# ---------------------------------------------------------------------------


def test_boot_rows_are_ordered_and_carry_one_boot_id() -> None:
    with boot_phases.span(boot_phases.PHASE_CELL_VERIFY):
        pass
    boot_phases.mark(boot_phases.PHASE_HELLO, since_process_start=True)
    with boot_phases.span(boot_phases.PHASE_WEIGHTS_FETCH, ref="repo/sdxl") as fetch:
        fetch.bytes_moved(4_720_000_000, boot_phases.SOURCE_VOLUME)

    rows = boot_phases.recorded_rows()
    assert _phases(rows) == [
        boot_phases.PHASE_CELL_VERIFY,
        boot_phases.PHASE_HELLO,
        boot_phases.PHASE_WEIGHTS_FETCH,
    ]
    # One boot id for the process; ordinals dense and strictly increasing in
    # the order the phases were opened.
    assert {r.boot_id for r in rows} == {boot_phases.BOOT_ID}
    opens = [r.ordinal for r in rows if not r.terminal]
    assert opens == sorted(opens)

    weights = [r for r in rows if r.terminal and r.phase == boot_phases.PHASE_WEIGHTS_FETCH]
    assert weights[0].bytes == 4_720_000_000
    assert weights[0].source == boot_phases.SOURCE_VOLUME
    assert weights[0].ref == "repo/sdxl"
    assert weights[0].outcome == boot_phases.OUTCOME_OK


def test_a_span_opens_a_row_and_closes_the_same_ordinal() -> None:
    """th#1269's shape: a pod that dies mid-phase leaves the OPEN row behind,
    which is how the hub can still say which phase it died in."""
    handle = boot_phases.open_span(boot_phases.PHASE_PIPELINE_LOAD)
    rows = boot_phases.recorded_rows()
    assert len(rows) == 1 and rows[0].terminal is False
    opened_ordinal = rows[0].ordinal

    handle.close()
    rows = boot_phases.recorded_rows()
    assert [r.terminal for r in rows] == [False, True]
    assert rows[1].ordinal == opened_ordinal  # the close UPSERTs onto the open


def test_rows_recorded_before_the_sink_exists_are_flushed_on_bind() -> None:
    """The boot window IS the no-sink window. Everything before bind must
    arrive, in order, once the stream comes up."""
    boot_phases.mark(boot_phases.PHASE_CELL_VERIFY)
    with boot_phases.span(boot_phases.PHASE_WEIGHTS_FETCH, ref="repo/a"):
        pass

    sent: List[pb.WorkerMessage] = []

    async def _drive() -> None:
        async def _send(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        boot_phases.bind_sink(_send, asyncio.get_running_loop())
        # Phases recorded AFTER the bind ride the same sink.
        # pgw#797: the boot CLOSE now requires `hello` first — a worker the hub
        # cannot reach is not servable, and closing before the stream existed is
        # what suppressed every span on 0.78.0. Binding the sink is exactly the
        # moment the real `on_hello_ack` records hello, so record it here too.
        boot_phases.mark(boot_phases.PHASE_HELLO, since_process_start=True)
        boot_phases.mark(boot_phases.PHASE_FIRST_REQUEST_SERVABLE,
                         since_process_start=True)
        await asyncio.sleep(0)  # let the fire-and-forget ship tasks run

    asyncio.run(_drive())

    assert sent, "no boot rows reached the transport"
    assert all(m.WhichOneof("msg") == "boot_phase" for m in sent)
    shipped = [m.boot_phase for m in sent]
    # The pre-bind rows were flushed, and the post-bind row followed them.
    assert boot_phases.PHASE_CELL_VERIFY in _phases(shipped)
    assert boot_phases.PHASE_WEIGHTS_FETCH in _phases(shipped)
    assert _phases(shipped)[-1] == boot_phases.PHASE_FIRST_REQUEST_SERVABLE


# ---------------------------------------------------------------------------
# reconciliation — the instrument must close
# ---------------------------------------------------------------------------


def test_nested_phases_charge_the_child_so_the_boot_reconciles() -> None:
    with boot_phases.span(boot_phases.PHASE_PIPELINE_LOAD):
        with boot_phases.span(boot_phases.PHASE_WARMUP):
            pass
    # pgw#797: `hello` gates the boot close (see the flush test above).
    boot_phases.mark(boot_phases.PHASE_HELLO, since_process_start=True)
    boot_phases.mark(boot_phases.PHASE_FIRST_REQUEST_SERVABLE,
                     since_process_start=True)

    rows = {r.phase: r for r in boot_phases.recorded_rows() if r.terminal}
    load = rows[boot_phases.PHASE_PIPELINE_LOAD]
    warm = rows[boot_phases.PHASE_WARMUP]
    assert warm.parent_ordinal == load.ordinal

    recon = boot_phases.reconciliation()
    # The nested load is charged once, to the child — the parent keeps only its
    # own time, so measured never exceeds the window it is measured against.
    assert recon["measured_ms"] <= recon["total_ms"]
    assert recon["residual_ms"] == recon["total_ms"] - recon["measured_ms"]
    assert "class.load" in recon


def test_a_typed_refusal_is_recorded_as_refused_not_failed() -> None:
    """A refused arm is the single most expensive boot fact to reconstruct
    after the fact: it decides whether this pod serves compiled or spends
    another 20 minutes minting. It must survive as ``refused`` with the
    CLASSIFIED reason — not as ``failed``, because the worker declined this
    cell deliberately and goes on serving eager."""
    exc = AdoptError("key_mismatch", "cell was minted for sm_90, host is sm_89")
    with boot_phases.span(
        boot_phases.PHASE_CELL_ARM,
        artifact_kind=aot_serve.ARTIFACT_KIND,
        artifact_key="ck1_deadbeef",
    ) as arm:
        # The shape aot_serve.enable uses: classified reason token + the
        # identifiers and the exception in detail (pgw#760 doctrine).
        arm.refused(str(exc.reason), f"key=ck1_deadbeef: {type(exc).__name__}: {exc}")

    row = [r for r in boot_phases.recorded_rows()
           if r.terminal and r.phase == boot_phases.PHASE_CELL_ARM][0]
    assert row.outcome == boot_phases.OUTCOME_REFUSED
    assert row.outcome != boot_phases.OUTCOME_FAILED
    assert row.reason == "key_mismatch"
    assert "sm_90" in row.detail
    assert row.artifact_kind == aot_serve.ARTIFACT_KIND
    assert row.artifact_key == "ck1_deadbeef"


def test_a_cumulative_milestone_is_not_summed_as_a_phase() -> None:
    """The bug this pins: ``hello`` and ``first_request_servable`` are measured
    from PROCESS START, so their duration covers the same wall clock the spans
    already account for. Summing both double-counts the entire boot — and any
    hub-side SUM(duration_ms) would inherit the error, which is why
    `cumulative` is a wire field and not an SDK-local detail."""
    with boot_phases.span(boot_phases.PHASE_CELL_VERIFY):
        pass
    boot_phases.mark(boot_phases.PHASE_HELLO, since_process_start=True)
    boot_phases.mark(boot_phases.PHASE_FIRST_REQUEST_SERVABLE,
                     since_process_start=True)

    rows = {r.phase: r for r in boot_phases.recorded_rows() if r.terminal}
    assert rows[boot_phases.PHASE_HELLO].cumulative is True
    assert rows[boot_phases.PHASE_FIRST_REQUEST_SERVABLE].cumulative is True
    assert rows[boot_phases.PHASE_CELL_VERIFY].cumulative is False

    recon = boot_phases.reconciliation()
    # measured_ms counts the env_seal span only; the two milestones are the
    # boot's total, not parts of it.
    assert recon["measured_ms"] < recon["total_ms"]
    assert recon["total_ms"] == boot_phases.servable_ms()


def test_phase_classes_cover_the_vocabulary() -> None:
    assert boot_phases.PHASES, "the vocabulary cannot be empty"
    for phase in boot_phases.PHASES:
        assert boot_phases.phase_class(phase) != "", phase
    # An unknown phase is reported unattributed rather than guessed.
    assert boot_phases.phase_class("something_new") == ""


def test_telemetry_never_breaks_the_boot_it_measures() -> None:
    """A span that raises still closes (as failed, with the classified reason)
    and re-raises: this is a measurement, never a behaviour change."""
    with pytest.raises(AdoptError):
        with boot_phases.span(boot_phases.PHASE_PIPELINE_LOAD):
            raise AdoptError("artifact_invalid", "truncated tar")

    row = [r for r in boot_phases.recorded_rows()
           if r.terminal and r.phase == boot_phases.PHASE_PIPELINE_LOAD][0]
    assert row.outcome == boot_phases.OUTCOME_FAILED
    assert row.reason == "artifact_invalid"
    assert "truncated tar" in row.detail


# ---------------------------------------------------------------------------
# pgw#764: the serving-mode dimensions
# ---------------------------------------------------------------------------


def test_serving_mode_distinguishes_aot_from_jit_from_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert serving_mode.classify_mode("") == serving_mode.MODE_EAGER
    # Both cell kinds set active_compile_ref, so the ref alone cannot be
    # pattern-matched — aot_serve owns the discrimination.
    monkeypatch.setattr(aot_serve, "is_aot_ref", lambda ref: ref.endswith("aot"))
    assert serving_mode.classify_mode("root/family-sdxl#ck1aot") == \
        serving_mode.MODE_AOT_CELL
    assert serving_mode.classify_mode("root/family-sdxl#ck1jit") == \
        serving_mode.MODE_JIT_CELL


def test_a_guard_miss_request_is_not_counted_as_compiled() -> None:
    """The defect this closes: a compiled lane that served ONE request eager
    still reported lane=...+compiled, contaminating every comparison."""
    served = serving_mode.resolve(
        active_compile_ref="root/family-sdxl#ck1", guard_missed=True, sm="sm_89")
    assert served.served_eager_fallback is True
    assert served.fallback_reason == serving_mode.FALLBACK_GUARD_MISS
    assert served.sm == "89"  # matches WorkerResources.gpu_sm spelling

    clean = serving_mode.resolve(
        active_compile_ref="root/family-sdxl#ck1", sm="89")
    assert clean.served_eager_fallback is False
    assert clean.fallback_reason == ""


def test_router_verdicts_are_eager_fallbacks() -> None:
    class _Router:
        volatile = {"unet"}
        healing: set = set()

    assert serving_mode.fallback_of(_Router(), "unet") == \
        serving_mode.FALLBACK_VOLATILE
    assert serving_mode.fallback_of(_Router(), "vae") == ""
    assert serving_mode.fallback_of(None, "unet") == ""


def test_shape_takes_the_executed_value_then_the_default() -> None:
    class _Payload:
        num_inference_steps = 28
        width = 1024

    class _Defaults:
        num_inference_steps = 20
        width = 512
        height = 768

    assert serving_mode.shape_of(_Payload(), _Defaults()) == (28, 1024, 768)
    # Non-spatial functions report 0 rather than inventing a shape.
    assert serving_mode.shape_of(object(), None) == (0, 0, 0)


def test_a_boot_milestone_is_recorded_once_per_process() -> None:
    """`on_hello_ack` runs again on every RECONNECT. "Process start -> hello"
    measured on a reconnect hours into a worker's life is not a boot number,
    and a second, much larger `hello` row would corrupt every boot aggregate
    that reads the series."""
    assert boot_phases.mark_once(boot_phases.PHASE_HELLO,
                                 since_process_start=True) is True
    assert boot_phases.mark_once(boot_phases.PHASE_HELLO,
                                 since_process_start=True) is False
    hellos = [r for r in boot_phases.recorded_rows()
              if r.terminal and r.phase == boot_phases.PHASE_HELLO]
    assert len(hellos) == 1
