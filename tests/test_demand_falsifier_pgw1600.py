"""pgw#1600 acceptance (c): `demand_miss`, counted and hub-visible per lane x regime.

The falsifier is the whole point of the issue's ordering — it ships BEFORE any
consumer, so the first thing that ever happens to a demand number is that
something tries to prove it wrong. These tests exercise the real banking path
and the real event emitter; nothing here is a mock of the instrument being
tested.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from gen_worker import activity as activity_mod
from gen_worker import demand_falsifier as falsifier
from gen_worker.demand import Basis, GiB, MiB, RequestShape, const, per_mp_batch

SHAPE = RequestShape(width=1024, height=1024, batch=2)
FORMULA = const(GiB(1.2)) + per_mp_batch(MiB(220))
PREDICTED = FORMULA.evaluate(SHAPE)


@pytest.fixture(autouse=True)
def _clean() -> Any:
    falsifier.reset_banked()
    yield
    falsifier.reset_banked()


@pytest.fixture()
def wire(monkeypatch: pytest.MonkeyPatch) -> List[tuple]:
    seen: List[tuple] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, phase="", **kw: seen.append((kind, phase, detail, kw)),
    )
    return seen


def _arena(allocated: int, outside: int = 0) -> falsifier.MeasuredArena:
    return falsifier.MeasuredArena(
        allocated_bytes=allocated, out_of_allocator_bytes=outside, measured=True,
    )


def test_a_request_INSIDE_the_prediction_banks_a_serve_and_no_miss(
    wire: List[tuple],
) -> None:
    falsifier.observe(
        lane="sdxl.diffusers@1+plain.bf16@1", regime="eager",
        demand=FORMULA, shape=SHAPE, measured=_arena(PREDICTED - MiB(1)),
    )
    (row,) = falsifier.banked()
    assert (row.served, row.misses) == (1, 0)
    assert not wire


def test_a_request_OVER_the_prediction_is_a_counted_hub_visible_event(
    wire: List[tuple],
) -> None:
    falsifier.observe(
        lane="sdxl.diffusers@1+plain.bf16@1", regime="eager",
        demand=FORMULA, shape=SHAPE, measured=_arena(PREDICTED + MiB(64)),
    )
    (row,) = falsifier.banked()
    assert (row.served, row.misses) == (1, 1)
    assert row.worst_miss_bytes == MiB(64)
    assert row.basis is Basis.UNCALIBRATED

    (kind, phase, detail, kw) = wire[-1]
    assert kind == falsifier.KIND_DEMAND_MISS == "demand_miss"
    # The KEY rides on the event, so a hub reader groups by (lane x regime)
    # without parsing the sentence.
    assert phase.startswith("sdxl.diffusers@1+plain.bf16@1|eager|")
    # (misses, served) ride as step/total_steps — the leak-RATE shape.
    assert (kw["step"], kw["total_steps"]) == (1, 1)
    assert "coefficient_basis=uncalibrated" in detail
    assert "shape=" in detail


def test_eager_and_compiled_NEVER_pool() -> None:
    """pgw#1586's rule, restated where it can be broken again.

    An eager sample landing in the compiled regime's count is the pgw#1548
    daemon death, delivered by a number that looks measured.
    """

    for regime in ("eager", "compiled"):
        falsifier.observe(
            lane="L", regime=regime, demand=FORMULA, shape=SHAPE,
            measured=_arena(PREDICTED + MiB(8)),
        )
    keys = {(row.lane, row.regime) for row in falsifier.banked()}
    assert keys == {("L", "eager"), ("L", "compiled")}
    assert all(row.served == 1 for row in falsifier.banked())


def test_an_unrecognised_regime_pools_with_NOTHING() -> None:
    falsifier.observe(
        lane="L", regime="", demand=FORMULA, shape=SHAPE,
        measured=_arena(PREDICTED + 1),
    )
    (row,) = falsifier.banked()
    assert row.regime.startswith("unknown("), (
        "a regime the worker could not determine must not be filed as eager"
    )


def test_the_COMPILED_regime_is_judged_on_the_DRIVER_total(
    wire: List[tuple],
) -> None:
    """AOTI's first-call pool lands outside the torch allocator (tcg#80).

    Judging a compiled serve on `max_memory_allocated` alone hides exactly the
    bytes that kill the process, so the compiled basis is
    allocated + out-of-allocator.
    """

    measured = _arena(PREDICTED - MiB(100), outside=MiB(200))
    falsifier.observe(
        lane="L", regime="compiled", demand=FORMULA, shape=SHAPE,
        measured=measured,
    )
    (row,) = falsifier.banked()
    assert row.misses == 1, (
        "the allocator alone was under the prediction; the driver total was "
        "not, and the driver total is the budget a compiled admit has"
    )
    assert measured.for_regime("eager") < PREDICTED
    assert measured.for_regime("compiled") > PREDICTED


def test_a_compiled_miss_is_a_P0_STAMP_DEFECT_not_a_statistic(
    wire: List[tuple],
) -> None:
    falsifier.observe(
        lane="L", regime="compiled", demand=FORMULA, shape=SHAPE,
        measured=_arena(PREDICTED + MiB(1)),
    )
    (_kind, phase, detail, _kw) = wire[-1]
    assert phase.endswith("|p0_stamp_defect")
    assert "severity=p0_stamp_defect" in detail


def test_an_UNMEASURED_request_banks_nothing(wire: List[tuple]) -> None:
    """A zero measurement would read as "predicted generously" and quietly
    prove every formula right — the exact failure mode a falsifier exists to
    avoid."""

    assert falsifier.observe(
        lane="L", regime="eager", demand=FORMULA, shape=SHAPE,
        measured=falsifier.MeasuredArena(),
    ) is None
    assert falsifier.banked() == ()
    assert not wire


def test_a_lane_with_no_declared_formula_banks_nothing() -> None:
    assert falsifier.observe(
        lane="L", regime="eager", demand=None, shape=SHAPE,
        measured=_arena(GiB(9)),
    ) is None
    assert falsifier.banked() == ()


def test_the_reconstructed_peak_alone_UNDERSTATES_the_driver_reading() -> None:
    """tcg#80's sm_89 decomposition, and the gap it exposes.

    The run read 4907 MiB allocated and 1155 MiB out-of-allocator at denoise
    against a DRIVER reading of 6649 MiB. Those do not add up, and the
    difference (~587 MiB) is allocator CACHE — reserved-but-not-allocated
    blocks that driver-free counts as gone and that a compiled call cannot
    spend at all. So the compiled basis takes the LARGER of the reconstructed
    peak and the driver growth; taking only the first would under-report by
    exactly the bytes pgw#1627 says are unavailable.
    """

    reconstructed_only = falsifier.MeasuredArena(
        allocated_bytes=MiB(4907), out_of_allocator_bytes=MiB(1155), measured=True,
    )
    assert round(reconstructed_only.driver_bytes / MiB(1)) == 6062
    assert reconstructed_only.driver_bytes < MiB(6649)

    with_driver = falsifier.MeasuredArena(
        allocated_bytes=MiB(4907), out_of_allocator_bytes=MiB(1155),
        driver_growth_bytes=MiB(6649), measured=True,
    )
    assert with_driver.driver_bytes == MiB(6649)
    assert with_driver.for_regime("eager") == MiB(4907), (
        "the eager basis is the allocator, and eager CAN spend the cache"
    )


def test_the_worst_miss_is_kept_not_the_last_one() -> None:
    for over in (MiB(10), MiB(300), MiB(5)):
        falsifier.observe(
            lane="L", regime="eager", demand=FORMULA, shape=SHAPE,
            measured=_arena(PREDICTED + over),
        )
    (row,) = falsifier.banked()
    assert (row.served, row.misses) == (3, 3)
    assert row.worst_miss_bytes == MiB(300)
    assert row.worst_measured_bytes == PREDICTED + MiB(300)
    assert row.as_document()["demand_miss"] == 3


def test_measure_request_arena_is_inert_and_silent_with_no_card() -> None:
    """CPU-only is the box this repo is developed on; the instrument must be
    a no-op there rather than a crash or a fabricated zero-sample."""

    with falsifier.measure_request_arena() as arena:
        pass
    assert arena[0].measured is False
    assert arena[0].driver_bytes == 0
