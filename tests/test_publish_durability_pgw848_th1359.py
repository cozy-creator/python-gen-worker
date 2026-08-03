"""pgw#848 item 1 — the mint's activity must stay RUNNING until the cell is
DURABLE, and a FAILING RETRY LOOP must not read as progress.

The second half is the test that matters. Extending the window wrongly turns a
reaped-too-early bug into a never-reap bug — a publish retrying forever on a
paid card, for a pod with no attendant (every production forge pod). That is
strictly worse than the bug being fixed, and it is the same reasoning that
refused `self_mint_publish` as a podguard progress kind.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

from gen_worker import fleet_cells


class _Act:
    def __init__(self) -> None:
        self.phases: List[str] = []

    def phase(self, name: str, **_: Any) -> None:
        self.phases.append(name)


def _reset() -> None:
    with fleet_cells._IN_FLIGHT_LOCK:
        fleet_cells._IN_FLIGHT.clear()
        fleet_cells._DURABLE_SEEN.clear()
        fleet_cells._DURABLE_PROGRESS = 0


def test_a_new_key_starting_its_upload_is_durable_progress() -> None:
    _reset()
    before = fleet_cells.publish_durable_progress()
    fleet_cells._note_durable("ck5-a", "started")
    assert fleet_cells.publish_durable_progress() == before + 1
    fleet_cells._note_durable("ck5-a", "published")
    assert fleet_cells.publish_durable_progress() == before + 2


def test_a_retry_of_the_SAME_key_is_not_progress() -> None:
    """THE LOAD-BEARING NEGATIVE. A publish that fails and retries forever must
    never advance the counter, or the activity never goes stale and the pod is
    never reaped."""
    _reset()
    fleet_cells._note_durable("ck5-a", "started")
    baseline = fleet_cells.publish_durable_progress()
    for _ in range(50):  # fifty failed attempts at the same key
        fleet_cells._note_durable("ck5-a", "started")
    assert fleet_cells.publish_durable_progress() == baseline, (
        "a failing publish retry loop advanced durable progress — the activity "
        "would read as progressing forever on a paid card"
    )


def test_failure_is_not_durable_progress() -> None:
    _reset()
    fleet_cells._note_durable("ck5-a", "started")
    baseline = fleet_cells.publish_durable_progress()
    with fleet_cells._IN_FLIGHT_LOCK:
        fleet_cells._REFUSED["ck5-a"] = "cell_publish_untrusted_compute"
    assert fleet_cells.publish_durable_progress() == baseline


def _drive(executor: Any, act: _Act) -> None:
    asyncio.run(executor._await_publish_durable(act))


def test_the_wait_beats_the_activity_ONLY_on_durable_movement(monkeypatch) -> None:
    from gen_worker.executor import Executor

    _reset()
    monkeypatch.setattr("gen_worker.executor._PUBLISH_SETTLE_POLL_S", 0.001)
    ticks = {"n": 0}

    def _in_flight():
        ticks["n"] += 1
        if ticks["n"] <= 6:
            return {"ck5-a": ("sdxl", 0.0)}
        return {}

    monkeypatch.setattr(fleet_cells, "publishes_in_flight", _in_flight)
    # Durable movement exactly twice across six polls.
    seq = iter([0, 0, 1, 1, 1, 2, 2, 2])
    monkeypatch.setattr(fleet_cells, "publish_durable_progress",
                        lambda: next(seq, 2))

    act = _Act()
    _drive(Executor.__new__(Executor), act)
    assert act.phases == ["publishing", "publishing"], (
        f"activity beat {len(act.phases)} times for 2 durable transitions — "
        "UpdatedAt must mean last PROGRESS, not last poll"
    )


def test_a_stuck_publish_never_beats_the_activity(monkeypatch) -> None:
    """A publish that is in flight forever with no durable movement must leave
    the activity's window ageing, so the stall verdict can condemn it."""
    from gen_worker.executor import Executor

    _reset()
    monkeypatch.setattr("gen_worker.executor._PUBLISH_SETTLE_POLL_S", 0.001)
    polls = {"n": 0}

    def _in_flight():
        polls["n"] += 1
        return {} if polls["n"] > 20 else {"ck5-a": ("sdxl", 0.0)}

    monkeypatch.setattr(fleet_cells, "publishes_in_flight", _in_flight)
    monkeypatch.setattr(fleet_cells, "publish_durable_progress", lambda: 7)

    act = _Act()
    _drive(Executor.__new__(Executor), act)
    assert act.phases == [], (
        "a stuck publish beat the activity — a liveness signal a failing retry "
        "loop can satisfy is not a liveness signal"
    )
