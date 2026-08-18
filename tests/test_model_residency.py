"""Model residency: admission before allocation, LRU two-tier moves, gates.

Integration-shaped: real ResidencyManager, real threads, recording backends —
the byte arithmetic, ordering and blocking behavior under test are the real
production objects; only the CUDA moves are recorders (this engine never
touches tensors — the pgw#1382 Model wrapper does, behind ModelBackend).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List

import pytest

from gen_worker.serving.residency import (
    NeverFits,
    ResidencyManager,
    Tier,
)

GB = 1024**3


class Sizer:
    def __init__(self, sizes: Dict[str, int], headroom: int = 1 * GB) -> None:
        self.sizes = sizes
        self.headroom = headroom

    def resident_bytes(self, checkpoint_ref: str, lane: str) -> int:
        return self.sizes[checkpoint_ref]

    def activation_headroom_bytes(self, checkpoint_ref: str, lane: str) -> int:
        return self.headroom


class Backend:
    """Records every move; asserts loads never overlap (the load gate)."""

    active_loads = 0
    load_overlap = False

    def __init__(self, name: str, journal: List[str]) -> None:
        self.name = name
        self.journal = journal

    def load(self) -> None:
        Backend.active_loads += 1
        if Backend.active_loads > 1:
            Backend.load_overlap = True
        time.sleep(0.02)  # widen the window a racing load would need
        self.journal.append(f"load:{self.name}")
        Backend.active_loads -= 1

    def demote_to_host(self) -> None:
        self.journal.append(f"demote:{self.name}")

    def promote_to_device(self) -> None:
        self.journal.append(f"promote:{self.name}")

    def drop(self) -> None:
        self.journal.append(f"drop:{self.name}")


@pytest.fixture(autouse=True)
def _reset_backend_class():
    Backend.active_loads = 0
    Backend.load_overlap = False
    yield


def manager(vram_gb: int, sizes: Dict[str, int], host_gb: int = 0) -> ResidencyManager:
    return ResidencyManager(
        vram_gb * GB, Sizer(sizes), host_budget_bytes=host_gb * GB
    )


def factory(name: str, journal: List[str]) -> "Callable[[], Backend]":
    def make() -> Backend:
        journal.append(f"construct:{name}")
        return Backend(name, journal)

    return make


LANE = "sdxl.diffusers-bf16@1"


def test_admission_refuses_a_never_fitting_model_before_any_allocation() -> None:
    journal: List[str] = []
    m = manager(10, {"huge": 12 * GB, "ok": 4 * GB})
    with pytest.raises(NeverFits) as excinfo:
        m.lease("huge", LANE, factory("huge", journal))
    # The refusal is at ADMISSION: nothing was constructed, no byte moved.
    assert journal == []
    message = str(excinfo.value)
    assert "13958643712" in message and "10737418240" in message
    assert "refuse at admission" in message
    # The card is untouched and still serves what fits.
    with m.lease("ok", LANE, factory("ok", journal)):
        pass
    assert journal == ["construct:ok", "load:ok"]


def test_lru_eviction_makes_room_and_demotes_to_host_in_order() -> None:
    journal: List[str] = []
    m = manager(10, {"a": 3 * GB, "b": 3 * GB, "c": 3 * GB}, host_gb=8)
    with m.lease("a", LANE, factory("a", journal)):
        pass
    with m.lease("b", LANE, factory("b", journal)):
        pass
    # a(4) + b(4) reserved incl. headroom; c needs 4 -> evict LRU = a.
    with m.lease("c", LANE, factory("c", journal)):
        pass
    assert journal == [
        "construct:a", "load:a",
        "construct:b", "load:b",
        "demote:a",  # eviction happened BEFORE c's construction/load
        "construct:c", "load:c",
    ]
    assert m.tier_of("a", LANE) is Tier.HOST
    assert m.tier_of("b", LANE) is Tier.VRAM
    assert m.tier_of("c", LANE) is Tier.VRAM


def test_repromotion_is_an_h2d_copy_never_a_reload() -> None:
    journal: List[str] = []
    m = manager(10, {"a": 3 * GB, "b": 3 * GB, "c": 3 * GB}, host_gb=8)
    for name in ("a", "b", "c"):  # c evicts a to host
        with m.lease(name, LANE, factory(name, journal)):
            pass
    journal.clear()
    with m.lease("a", LANE, factory("a", journal)):  # b is LRU now
        pass
    assert "promote:a" in journal
    assert "construct:a" not in journal and "load:a" not in journal
    assert journal.index("demote:b") < journal.index("promote:a")


def test_host_tier_overflow_drops_the_oldest_host_resident() -> None:
    journal: List[str] = []
    # Host tier holds ONE 3 GB body only: the second demotion drops the first.
    m = manager(10, {"a": 3 * GB, "b": 3 * GB, "c": 3 * GB, "d": 3 * GB}, host_gb=5)
    for name in ("a", "b", "c", "d"):
        with m.lease(name, LANE, factory(name, journal)):
            pass
    assert m.tier_of("a", LANE) is Tier.ABSENT  # dropped when b demoted
    assert m.tier_of("b", LANE) is Tier.HOST
    assert journal.index("demote:a") < journal.index("drop:a") < journal.index("demote:b")


def test_a_zero_host_budget_drops_straight_to_the_chunk_store() -> None:
    journal: List[str] = []
    m = manager(10, {"a": 4 * GB, "b": 5 * GB})
    with m.lease("a", LANE, factory("a", journal)):
        pass
    with m.lease("b", LANE, factory("b", journal)):
        pass
    assert "drop:a" in journal and "demote:a" not in journal
    assert m.tier_of("a", LANE) is Tier.ABSENT


def test_concurrent_loads_are_serialized_by_the_load_gate() -> None:
    journal: List[str] = []
    m = manager(64, {f"m{i}": 2 * GB for i in range(6)})
    threads = [
        threading.Thread(
            target=lambda n=n: m.lease(n, LANE, factory(n, journal)).release()
        )
        for n in (f"m{i}" for i in range(6))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not Backend.load_overlap, "two instance loads overlapped on one GPU"
    assert sorted(j for j in journal if j.startswith("load:")) == sorted(
        f"load:m{i}" for i in range(6)
    )


def test_single_flight_per_instance_and_no_tier_moves_in_flight() -> None:
    journal: List[str] = []
    m = manager(6, {"a": 4 * GB, "b": 4 * GB}, host_gb=8)
    order: List[str] = []
    first_in = threading.Event()
    release_first = threading.Event()

    def first() -> None:
        with m.lease("a", LANE, factory("a", journal)):
            order.append("first-in")
            first_in.set()
            release_first.wait(30)
        order.append("first-out")

    def second() -> None:
        first_in.wait(30)
        with m.lease("a", LANE, factory("a", journal)):
            order.append("second-in")

    def evictor() -> None:
        first_in.wait(30)
        # b needs the room a holds; a is IN FLIGHT, so this must BLOCK until
        # a's requests drain — never move a mid-request.
        with m.lease("b", LANE, factory("b", journal)):
            order.append("evictor-in")

    threads = [threading.Thread(target=f) for f in (first, second, evictor)]
    for t in threads:
        t.start()
    time.sleep(0.15)
    # Nobody got in while the first request held the instance, and a was not
    # demoted underneath it.
    assert order == ["first-in"]
    assert "demote:a" not in journal and "drop:a" not in journal
    release_first.set()
    for t in threads:
        t.join(timeout=30)
    assert order[0] == "first-in" and set(order) == {
        "first-in", "first-out", "second-in", "evictor-in",
    }
    # The eviction happened only once a was idle.
    assert "demote:a" in journal


def test_a_fits_after_drain_admission_blocks_instead_of_ooming() -> None:
    journal: List[str] = []
    m = manager(6, {"a": 4 * GB, "b": 4 * GB})
    entered = threading.Event()
    release = threading.Event()
    landed: List[str] = []

    def holder() -> None:
        with m.lease("a", LANE, factory("a", journal)):
            entered.set()
            release.wait(30)

    def contender() -> None:
        entered.wait(30)
        with m.lease("b", LANE, factory("b", journal)):
            landed.append("b")

    threads = [threading.Thread(target=holder), threading.Thread(target=contender)]
    for t in threads:
        t.start()
    time.sleep(0.1)
    assert landed == []  # blocked on the drain, not refused, not loaded
    assert "construct:b" not in journal  # admission-before-allocation held
    release.set()
    for t in threads:
        t.join(timeout=30)
    assert landed == ["b"]


def test_reservation_accounting_includes_headroom_and_reads_back() -> None:
    journal: List[str] = []
    m = manager(10, {"a": 3 * GB})
    with m.lease("a", LANE, factory("a", journal)):
        vram, host = m.reserved_bytes()
        assert vram == 4 * GB  # 3 weights + 1 headroom
        assert host == 0


# -- the BUDGET the manager admits against (pgw#1411) ------------------------
#
# `Worker.__init__` chose it inline, so which arm ran was a property of the
# machine running the suite — and the arms disagree about whether a host can
# serve at all. Measured on this workspace's dev box and on CI, they cover the
# decision between them like this:
#
#     dev box   cuda_ready() True,  headroom 7.7 GiB  -> the headroom arm
#     CI        cuda_ready() False, headroom None     -> the host-RAM arm
#     NOWHERE   cuda_ready() True,  headroom None     -> THE REFUSAL
#
# The refusal shipped having never executed anywhere, because no environment
# could reach it. So the arms are chosen here by naming the two readings rather
# than by hoping for a machine.

import pytest as _pytest_budget

from gen_worker import hostfacts as _hostfacts
from gen_worker.worker import WorkerBootError as _WorkerBootError
from gen_worker.worker import residency_budget as _residency_budget


def _readings(
    monkeypatch: _pytest_budget.MonkeyPatch, *, headroom: int | None, cuda: bool
) -> None:
    monkeypatch.setattr(_hostfacts, "headroom_bytes", lambda *a, **k: headroom)
    monkeypatch.setattr(_hostfacts, "cuda_ready", lambda *a, **k: cuda)


def test_a_stated_budget_wins_over_every_reading(
    monkeypatch: _pytest_budget.MonkeyPatch,
) -> None:
    """The caller owns it: a stated budget is a CONFIG, so it is not
    second-guessed by a card that happens to be readable."""
    _readings(monkeypatch, headroom=99, cuda=True)
    assert _residency_budget(4096) == 4096


def test_the_cards_headroom_is_the_budget_when_it_reads(
    monkeypatch: _pytest_budget.MonkeyPatch,
) -> None:
    _readings(monkeypatch, headroom=8 << 30, cuda=True)
    assert _residency_budget() == 8 << 30


def test_no_cuda_at_all_sizes_from_host_ram(
    monkeypatch: _pytest_budget.MonkeyPatch,
) -> None:
    """cozy-local, CI and every fake-weights drive are this arm. Refusing here
    made the worker unbootable on all three while `python -m gen_worker.serving`
    served happily beside it — two paths in one repo disagreeing about whether
    a host can serve."""
    _readings(monkeypatch, headroom=None, cuda=False)
    monkeypatch.setattr(
        "gen_worker.models.memory.get_available_ram_gb", lambda: 32.0
    )
    assert _residency_budget() == int(32.0 * (1024 ** 3))


def test_a_card_we_can_SEE_but_not_MEASURE_refuses(
    monkeypatch: _pytest_budget.MonkeyPatch,
) -> None:
    """THE ARM NO ENVIRONMENT REACHES, which is why it is written down here.

    A visible card whose memory cannot be read is the one genuinely unknown
    case: sizing from host RAM would hand the residency manager a budget in
    the wrong memory entirely, and admission happens BEFORE allocation, so the
    error would surface as an OOM on a rented pod rather than as a refusal."""
    _readings(monkeypatch, headroom=None, cuda=True)
    with _pytest_budget.raises(_WorkerBootError) as caught:
        _residency_budget()
    message = str(caught.value)
    assert "card it can see" in message, message
    assert "mem_get_info" in message, message


def test_an_unreadable_host_refuses_rather_than_admitting_against_zero(
    monkeypatch: _pytest_budget.MonkeyPatch,
) -> None:
    """A zero budget admits nothing and would read as "this model never fits",
    which is a lie about the model rather than about the reading."""
    _readings(monkeypatch, headroom=None, cuda=False)
    monkeypatch.setattr(
        "gen_worker.models.memory.get_available_ram_gb", lambda: 0.0
    )
    with _pytest_budget.raises(_WorkerBootError) as caught:
        _residency_budget()
    assert "no readable host RAM" in str(caught.value)
