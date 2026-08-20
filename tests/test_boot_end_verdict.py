"""pgw#1480: `boot_ended_uncompiled` must be EMITTABLE, and only when earned.

se#780's deliverable-2a names row 6 *"the one that makes this a proof rather
than a story"*:

    | 6 | and the pod did NOT silently serve eager |
        `self_mint_skipped/boot_ended_uncompiled` must be ABSENT |

`EagerPhase.BOOT_ENDED_UNCOMPILED` was defined and emitted by NOTHING, so that
row was ABSENT on every pod that has ever run — including every pod it exists to
condemn. An 80 GB rental was about to be judged on an unfalsifiable pass.

This file is the falsifiability proof, and it is a THREE-ARMED one, because the
useful half of the fix is the silence:

  * a boot that DECLARED compile and armed nothing -> the phase FIRES;
  * a boot that armed a graph specialization       -> SILENT (one armed
    specialization is dispatchable; pgw#844 says this token means "nothing is",
    never "partial");
  * an eager-by-design release                     -> SILENT (not every eager
    boot is a defect — this phase means DECLARED-and-didn't).

A test that only proved the first arm would prove the emitter fires, not that it
MEANS anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

from gen_worker import activity as activity_mod
from gen_worker.compiled_graph_adopt import EagerPhase
from gen_worker.serving.self_mint import KIND_SKIPPED
from gen_worker.serving.serve_adoption import (
    EAGER_BY_DESIGN_REFUSALS,
    ServeAdoption,
)

PHASE = EagerPhase.BOOT_ENDED_UNCOMPILED.value


class FakeSession:
    """What `AdoptSession` is to this verdict: five counted collections.

    pgw#1534 added `ambiguous` and `unclaimed_marks` to what the verdict
    reads, and this double went stale the moment it did — which is the
    argument for keeping it a MODEL of the contract rather than the two
    fields one caller happened to use. `silently_eager` is a method on the
    real session, so it is one here too.
    """

    def __init__(
        self,
        adopted: int = 0,
        holes: int = 0,
        unclaimed: int = 0,
        ambiguous: int = 0,
        unclaimed_marks: int = 0,
    ):
        self.adopted = ["graph"] * adopted
        self.holes = ["hole"] * holes
        self.unclaimed = ["unclaimed"] * unclaimed
        self.ambiguous = ["ambiguous"] * ambiguous
        self.unclaimed_marks = [_Mark()] * unclaimed_marks

    def silently_eager(self) -> bool:
        return bool(self.unclaimed_marks) and not self.adopted and not self.holes

    def adopt(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - unused
        raise AssertionError("the verdict never runs a graph")


class _Mark:
    """One `UnclaimedMark`: the verdict only ever asks it to describe itself."""

    def describe(self) -> str:
        return "FakeModule: marked with ctx.compile, matched NO graph in this lane"


class FakeMintStatus:
    def __init__(self, running: bool) -> None:
        self.running = running


@pytest.fixture()
def seen(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Every ActivityUpdate this pod put on the wire, in order."""
    captured: List[Any] = []
    monkeypatch.setattr(activity_mod, "_sink", captured.append)
    return captured


@pytest.fixture()
def adoption(tmp_path: Path) -> ServeAdoption:
    return ServeAdoption("rel-1", sm="sm_90", artifacts_dir=tmp_path / "artifacts")


def phases(seen: List[Any], kind: str = KIND_SKIPPED) -> List[str]:
    return [u.phase for u in seen if u.kind == kind]


def verdicts(seen: List[Any]) -> List[Any]:
    return [u for u in seen if u.kind == KIND_SKIPPED and u.phase == PHASE]


# -- RED: the condition the instrument exists to catch -----------------------


def test_declared_compile_and_ZERO_armed_specializations_FIRES(
    adoption: ServeAdoption, seen: List[Any]
) -> None:
    """The whole defect, in one boot: the release stamped a lane document, the
    document claimed graph specializations, not one of them armed, and nothing
    is going to fix it. Before this fix the pod said NOTHING."""

    adoption.adoption = FakeSession(adopted=0, holes=0, unclaimed=3)
    adoption.contract = "h3.diffusers-bf16@1"
    adoption.loaded()

    rows = verdicts(seen)
    assert len(rows) == 1, f"expected the boot-end verdict, saw {phases(seen)}"
    assert (rows[0].step, rows[0].total_steps) == (0, 0)
    assert "serves EAGER" in rows[0].detail


def test_the_counts_ride_the_NUMERIC_columns_not_the_prose(
    adoption: ServeAdoption, seen: List[Any]
) -> None:
    """`0 of 6 armed` interpolated into `detail` is a metric nobody can group
    by. armed -> `step`, claimed -> `total_steps`, as numbers."""

    adoption.adoption = FakeSession(adopted=0, holes=6)
    adoption.contract = "h3.diffusers-bf16@1"
    adoption._on_adopted = lambda _self: FakeMintStatus(running=False)
    adoption.loaded()

    row = verdicts(seen)[0]
    assert (row.step, row.total_steps) == (0, 6)


def test_a_mint_that_NEVER_STARTED_is_terminal_and_FIRES(
    adoption: ServeAdoption, seen: List[Any]
) -> None:
    """`SelfMint.arm` answers `unavailable` on a pod with no compiler. Nothing
    will fill the holes, so the eager window has no end — which is exactly what
    this token means. Dropping the mint's status made this indistinguishable
    from a healthy background mint."""

    adoption.adoption = FakeSession(adopted=0, holes=2)
    adoption.contract = "sdxl.diffusers-bf16@1"
    adoption._on_adopted = lambda _self: FakeMintStatus(running=False)
    adoption.loaded()

    assert len(verdicts(seen)) == 1


def test_a_STAMPED_release_with_no_lane_document_for_this_sm_FIRES(
    adoption: ServeAdoption, seen: List[Any]
) -> None:
    """`no_document` is a release that DID declare compiled serving and simply
    has no graphs for this (lane x sm) — a pod serving eager under it is the
    headline shape, not an exemption."""

    adoption._refuse("no_document", "no lane document for (rel-1 x lane x sm_90)")
    adoption.loaded()

    assert len(verdicts(seen)) == 1
    assert "no_document" in verdicts(seen)[0].detail


def test_an_environment_mismatch_FIRES_because_the_release_still_declared(
    adoption: ServeAdoption, seen: List[Any]
) -> None:
    """pgw#1472's silent-exact-env refusal: the pod is not the release's env,
    so it adopts nothing forever. The release declared; the pod did not."""

    adoption._refuse("environment_mismatch", "closure abc != stamped def")
    adoption.loaded()

    assert len(verdicts(seen)) == 1


# -- GREEN: the silences, without which the red proves nothing ---------------


def test_a_boot_that_ARMS_a_specialization_stays_SILENT(
    adoption: ServeAdoption, seen: List[Any]
) -> None:
    """One armed graph specialization is dispatchable. pgw#844: this token must
    mean `nothing is dispatchable`, NEVER `partial` — a token that fires on a
    partially-armed pod is a token operators learn to ignore."""

    adoption.adoption = FakeSession(adopted=1, holes=4)
    adoption.contract = "sdxl.diffusers-bf16@1"
    adoption._on_adopted = lambda _self: FakeMintStatus(running=True)
    adoption.loaded()

    assert verdicts(seen) == []
    # ...and the boot DID report, so the silence is a decision, not a dead path.
    assert [u.phase for u in seen if u.kind == "boot_adopt_summary"] == ["minting"]


def test_a_MINT_IN_FLIGHT_stays_SILENT_because_the_eager_window_has_an_END(
    adoption: ServeAdoption, seen: List[Any]
) -> None:
    """Boot 1 of any reuse proof: zero armed, holes registered, the background
    mint running. That is `mint_in_progress`, a NON-terminal state, and
    `boot_adopt_summary phase='minting' step=0` already carries it. Firing the
    terminal token here would convert an unfalsifiable check into a FALSE
    ALARM, which burns the rental just as thoroughly."""

    adoption.adoption = FakeSession(adopted=0, holes=3)
    adoption.contract = "h3.diffusers-bf16@1"
    adoption._on_adopted = lambda _self: FakeMintStatus(running=True)
    adoption.loaded()

    assert verdicts(seen) == []


def test_an_UNANSWERING_hook_with_holes_stays_SILENT(
    adoption: ServeAdoption, seen: List[Any]
) -> None:
    """A wiring that returns nothing is UNKNOWN, and unknown resolves to
    silence when there are holes: a false alarm is worse than a missed one
    because it teaches the reader to stop reading."""

    adoption.adoption = FakeSession(adopted=0, holes=3)
    adoption.contract = "h3.diffusers-bf16@1"
    adoption._on_adopted = lambda _self: None
    adoption.loaded()

    assert verdicts(seen) == []


@pytest.mark.parametrize("phase", sorted(EAGER_BY_DESIGN_REFUSALS))
def test_an_EAGER_BY_DESIGN_release_stays_SILENT(
    adoption: ServeAdoption, seen: List[Any], phase: str
) -> None:
    """Not every eager boot is a defect. A release that declared no compile is
    the contract working; an instrument that fires on it is noise."""

    adoption._refuse(phase, "eager is this release's contract")
    adoption.loaded()

    assert verdicts(seen) == []


def test_a_pod_that_never_ATTEMPTED_an_adopt_stays_SILENT(
    adoption: ServeAdoption, seen: List[Any]
) -> None:
    """No session and no refusal: `sink_for` was never reached (a CPU-only
    pod, a release with no id). Nothing was declared, so nothing is condemned."""

    adoption.loaded()

    assert verdicts(seen) == []


# -- the verdict is a BOOT verdict, so it fires once -------------------------


def test_the_verdict_fires_ONCE_per_pod_not_once_per_instance(
    adoption: ServeAdoption, seen: List[Any]
) -> None:
    """`ServeLoop` makes instances lazily under residency leases, so `loaded`
    is called again for every new (class x checkpoint x lane). A per-instance
    verdict would multiply one pod's eager boot into N rows and turn a count
    into a lease-churn measurement."""

    adoption.adoption = FakeSession(adopted=0, holes=0, unclaimed=2)
    adoption.contract = "h3.diffusers-bf16@1"
    for _ in range(4):
        adoption.loaded()

    assert len(verdicts(seen)) == 1


def test_every_EAGER_BY_DESIGN_phase_is_one_this_module_actually_emits() -> None:
    """A silence-list nothing can match silences nothing. Same rule pgw#1472's
    `PERMANENT_REFUSALS` is held to: a set that cannot fire is not a
    classification, it is folklore with a frozenset around it."""

    import gen_worker.serving.serve_adoption as module

    source = Path(module.__file__).read_text()
    for phase in EAGER_BY_DESIGN_REFUSALS:
        assert f'_refuse("{phase}"' in source, (
            f"{phase!r} is declared eager-by-design but this module never "
            f"refuses with it")


def test_the_kind_is_the_one_the_HUB_ALREADY_STORES() -> None:
    """se#780's row 6 names `self_mint_skipped/boot_ended_uncompiled`, and
    tensorhub's `ActivityKindSelfMintSkipped` is that kind. Inventing a new
    kind here would have made the fix un-queryable until a hub change landed."""

    assert KIND_SKIPPED == "self_mint_skipped"
    assert PHASE == "boot_ended_uncompiled"
