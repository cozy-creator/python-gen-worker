"""pgw#789: the benchmark telemetry the platform could not actually answer with.

Three questions have to be answerable from durable HUB-side records — compile
duration, cold-boot phase breakdown, warm per-request latency by serving mode —
without anyone SSH-ing into a pod. pgw#764/th#1293 built the wire and the
tables for all three. This file is the red-verify for the parts that were
BUILT BUT NEVER CONNECTED, each of which was measured absent on the live chaos
stack before the fix:

* ``serving_mode.py`` was imported by nothing but its own unit test, so
  ``JobMetrics.serving_mode`` / ``served_cell_ref`` / ``sm`` / ``steps`` /
  ``width`` / ``height`` were never populated. Measured: 0 of 416
  ``request_state`` rows carried ``serving_mode``, so
  ``/v1/admin/request-latency`` could not separate AOT from JIT from eager over
  ANY traffic. Asserted here on the real wire (the terminal ``JobResult`` the
  hub-double receives), not on a return value.

* The ``weights_fetch`` boot span wrapped only the hf/civitai prefetch loop.
  The tensorhub refs that own the ~230s of a real cold boot arrive later via
  DesiredResidency/RunJob and never pass through it — so the expensive phase
  was invisible and nothing stamped bytes or source at all. Measured: six real
  boots on chaos recorded exactly two rows each (``first_request_servable`` and
  ``hello``, both cumulative milestones) — zero spans, so ``measured_ms`` was 0
  and the whole boot was residual.

* ``first_request_servable`` was marked when ``startup()`` returned, even when
  every function was still awaiting hub-supplied snapshots and the worker
  therefore advertised nothing the hub could dispatch to. Measured: those same
  six boots reported 4.2-12.3s cold boots for pods whose real boots were
  minutes.

Real codepaths throughout: the dimension test drives a real worker over a real
gRPC socket via the hub-double; the boot tests drive the real recorder and the
real lifecycle methods.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

import msgspec
import pytest

from gen_worker import boot_phases, serving_mode
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.hub_double import hub_double, is_ready, is_result_for
from harness.shape_endpoints_pgw789 import ShapedIn


@pytest.fixture(autouse=True)
def _reset() -> Any:
    boot_phases.reset_for_tests()
    serving_mode.detect_sm.cache_clear()
    yield
    boot_phases.reset_for_tests()
    serving_mode.detect_sm.cache_clear()


# ---------------------------------------------------------------------------
# Question 3: warm per-request latency, with the dimensions that make it
# comparable across serving paths.
# ---------------------------------------------------------------------------


def test_job_metrics_carry_the_serving_dimensions_on_the_wire() -> None:
    """The dimensions reach the HUB, over the real stream.

    Before this, `serving_mode.py` computed all of them correctly and nothing
    ever called it: `Executor._metrics` built a JobMetrics without one of these
    fields, so every request_state row read serving_mode='' forever.
    """
    with hub_double(modules=("harness.shape_endpoints_pgw789",)) as (scheduler, _h):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-dims", attempt=1, function_name="render",
            input_payload=msgspec.msgpack.encode(
                ShapedIn(prompt="x", num_inference_steps=28, width=1024)),
        ))
        res = conn.wait_for(is_result_for("r-dims")).job_result

    assert res.status == pb.JOB_STATUS_OK
    m = res.metrics
    # This endpoint declares no compile cell, so the honest answer is eager —
    # and it must be the WORD "eager", not an empty string. "" is
    # indistinguishable from "a worker too old to report", which is exactly the
    # ambiguity that made the aggregate unusable.
    assert m.serving_mode == serving_mode.MODE_EAGER
    assert m.served_cell_ref == ""
    assert m.served_eager_fallback is False
    assert m.fallback_reason == ""
    # The executed shape, defaults applied: steps+width came from the payload,
    # height from the struct default — because the DEFAULT is what executed.
    assert m.steps == 28
    assert m.width == 1024
    assert m.height == 768


def test_a_guard_missed_request_reports_the_fallback_not_the_tier() -> None:
    """A compiled lane that served ONE request eager must not be counted as a
    compiled latency sample. The router's own verdict outranks the generic
    class: `volatile` means permanently eager for this shape, and downgrading
    it to `guard_miss` loses that."""
    served = serving_mode.resolve(
        active_compile_ref="root/family-sdxl#ck5",
        guard_missed=True,
        verdict=serving_mode.FALLBACK_VOLATILE,
        sm="sm_89",
    )
    assert served.served_eager_fallback is True
    assert served.fallback_reason == serving_mode.FALLBACK_VOLATILE
    assert served.sm == "89"


def test_router_verdict_sets_are_keyed_by_signature_not_target() -> None:
    """`hot_swap.Router.healing`/`.volatile` hold input SIGNATURES. A caller
    that passed a TARGET name got "" for every request, so the fallback axis
    read perfectly clean while requests were in fact falling back."""
    class _Router:
        volatile = {"sig-1024x768x28"}
        healing: set = set()

    assert serving_mode.fallback_of(_Router(), "sig-1024x768x28") == \
        serving_mode.FALLBACK_VOLATILE
    assert serving_mode.fallback_of(_Router(), "unet") == ""


def test_shape_reads_a_mapping_of_defaults_too() -> None:
    """The executor's fallback defaults are `_effective_config`'s dict, not an
    attribute holder; `shape_of` must accept both or the call site has to
    fabricate an object."""
    class _Payload:
        num_inference_steps = 40

    assert serving_mode.shape_of(_Payload(), {"width": 1280, "height": 720}) == \
        (40, 1280, 720)


# ---------------------------------------------------------------------------
# Question 2: the cold-boot breakdown — and the ~230s that owns it.
# ---------------------------------------------------------------------------


def _row(rows: List[pb.BootPhase], phase: str) -> Optional[pb.BootPhase]:
    return next((r for r in rows if r.phase == phase and r.terminal), None)


def test_a_weights_fetch_span_attributes_bytes_and_their_source() -> None:
    """The ~230s question is "was it network or a warm volume". A span with a
    duration and no bytes/source cannot answer it, and every recorded boot on
    chaos had no weights_fetch span at all."""
    with boot_phases.span(boot_phases.PHASE_WEIGHTS_FETCH, ref="repo/sdxl") as f:
        f.bytes_moved(4_720_000_000, boot_phases.SOURCE_R2)
        f.note("ref=repo/sdxl net_bytes=4720000000 manifest_bytes=4720000000")
    row = _row(boot_phases.recorded_rows(), boot_phases.PHASE_WEIGHTS_FETCH)
    assert row is not None
    assert row.bytes == 4_720_000_000
    assert row.source == boot_phases.SOURCE_R2
    assert boot_phases.phase_class(row.phase) == boot_phases.CLASS_FETCH


def test_in_boot_closes_at_the_servable_milestone() -> None:
    """The gate that keeps steady-state materializations OUT of the boot
    ladder. Without it a ref delivered hours later appends a `weights_fetch`
    span to a finished boot, and `residual_ms` stops reconciling."""
    assert boot_phases.in_boot() is True
    # pgw#797: the close now has a PRECONDITION — `hello`. A worker the hub
    # cannot reach is not servable, and closing the boot before the stream
    # existed is what suppressed every span on 0.78.0.
    boot_phases.mark_once(boot_phases.PHASE_HELLO, since_process_start=True)
    boot_phases.mark(
        boot_phases.PHASE_FIRST_REQUEST_SERVABLE, since_process_start=True)
    assert boot_phases.in_boot() is False
    assert boot_phases.servable_ms() is not None


def test_the_boot_ladder_reconciles_with_the_new_phases() -> None:
    """th#1111's rule at boot scale: measured + residual == the boot window,
    and the cumulative milestone is the total, never a part of the sum."""
    boot_phases.mark_once(boot_phases.PHASE_HELLO, since_process_start=True)
    with boot_phases.span(boot_phases.PHASE_WEIGHTS_FETCH, ref="repo/sdxl") as f:
        f.bytes_moved(1_000, boot_phases.SOURCE_VOLUME)
    with boot_phases.span(boot_phases.PHASE_PIPELINE_LOAD, function="render"):
        pass
    boot_phases.mark(
        boot_phases.PHASE_FIRST_REQUEST_SERVABLE, since_process_start=True)
    rec = boot_phases.reconciliation()
    assert rec["measured_ms"] + rec["residual_ms"] == rec["total_ms"]
    # Both new phases are CLASSIFIED, so "this release's boots are
    # network-bound" is a query rather than a hunch.
    assert "class.fetch" in rec
    assert "class.load" in rec


def test_servable_is_deferred_until_awaited_functions_are_set_up() -> None:
    """The correctness bug behind every 4.8s "cold boot" recorded on chaos.

    `startup()` marked the milestone on return, but with functions still
    awaiting hub-supplied snapshots the worker advertises NOTHING the hub can
    dispatch to — so the number measured "startup() returned", not "servable",
    and the ~230s of tensorhub-ref weight fetching happens AFTER it on exactly
    that path. Drives the real `Lifecycle` methods.

    pgw#797 CORRECTED THE OWNERSHIP this test originally pinned. Deferring the
    mark into `_setup_awaiting_functions` fixed one path and left the one every
    real release takes (a `Compile`/`Slot` spec is routed to `dynamic`, never
    reaches `awaiting_hub`, and closed the boot at the end of `startup()`
    anyway). The milestone now has ONE owner — `maybe_send_state_delta`, which
    marks it on the fact itself: a StateDelta advertising a function went out.
    So what this test can still pin is the NEGATIVE half — awaiting functions
    do not close the boot — and that `_setup_awaiting_functions` is no longer an
    owner. The positive half is only assertable on a real boot, and lives in
    `test_boot_span_ladder_pgw797.py`, off the wire.
    """
    from gen_worker.lifecycle import Lifecycle

    class _Store:
        def __init__(self) -> None:
            self.present: set = set()

        def local_path(self, ref: str) -> Optional[str]:
            return "/cas/x" if ref in self.present else None

    class _Executor:
        def __init__(self, store: _Store) -> None:
            self.store = store
            self.specs: dict = {}
            self.unavailable: dict = {}

        async def ensure_setup(self, spec: Any) -> None:
            return None

    store = _Store()
    life = object.__new__(Lifecycle)
    life.executor = _Executor(store)  # type: ignore[attr-defined]
    life.draining = False  # type: ignore[attr-defined]

    async def _noop(*_a: Any, **_kw: Any) -> None:
        return None

    life.maybe_send_state_delta = _noop  # type: ignore[assignment]

    async def _drive() -> None:
        watch = asyncio.create_task(
            life._setup_awaiting_functions({"render": ["repo/sdxl"]}))
        # The ref has not landed: the worker is READY but unservable, and the
        # boot must NOT have a closing milestone yet.
        for _ in range(50):
            await asyncio.sleep(0)
            if not watch.done():
                break
        assert boot_phases.servable_ms() is None
        store.present.add("repo/sdxl")
        await watch

    asyncio.run(_drive())
    # Snapshot delivered, setup finished — and the boot is STILL open, because
    # `maybe_send_state_delta` is stubbed here and nothing has advertised a
    # function. That is the corrected contract, not a regression: the close
    # follows the advertisement, never the setup call that enabled it.
    assert boot_phases.servable_ms() is None
    assert boot_phases.in_boot() is True
    assert not [
        r for r in boot_phases.recorded_rows()
        if r.phase == boot_phases.PHASE_FIRST_REQUEST_SERVABLE
    ], "_setup_awaiting_functions is no longer an owner of the boot close"


def test_a_draining_worker_never_claims_a_boot_number() -> None:
    """It never became servable. Recording a milestone anyway would put a
    fictional cold boot into the aggregate the autoscaler prices against."""
    from gen_worker.lifecycle import Lifecycle

    life = object.__new__(Lifecycle)
    life.draining = True  # type: ignore[attr-defined]

    class _Executor:
        specs: dict = {}
        unavailable: dict = {}

        class store:  # noqa: N801
            @staticmethod
            def local_path(_ref: str) -> None:
                return None

    life.executor = _Executor()  # type: ignore[attr-defined]
    asyncio.run(life._setup_awaiting_functions({"render": ["repo/sdxl"]}))
    assert boot_phases.servable_ms() is None
