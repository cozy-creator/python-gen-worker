"""Residency state machine (#366) — CPU-only, deterministic budget + fakes.

Covers: tier transitions with events, LRU eviction ORDER under make_room,
pin / pin-while-executing protection, unload/evict semantics, and the
free-VRAM-only decision input (an explicit budget replaces the CUDA probe).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.models import residency as residency_mod
from gen_worker.models.residency import Residency, Tier

_GiB = 1024 ** 3


class _Pipe:
    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device: str) -> "_Pipe":
        self.moves.append(device)
        return self


@pytest.fixture(autouse=True)
def _plenty_of_ram(monkeypatch):
    # Keep the warm tier deterministic regardless of host RAM.
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 64.0)


def _res(events: list, budget_gb: int = 10) -> Residency:
    return Residency(
        on_event=lambda ref, state, vb, dur=0: events.append((ref, state, vb)),
        vram_budget_bytes=budget_gb * _GiB,
    )


# --------------------------------------------------------------------------- #
# State machine: DISK -> VRAM -> RAM -> VRAM -> DISK -> EVICTED, with events
# --------------------------------------------------------------------------- #


def test_state_machine_and_events(tmp_path: Path) -> None:
    events: list = []
    res = _res(events)
    pipe = _Pipe()

    res.track_disk("a/model", tmp_path)
    assert res.tier("a/model") is Tier.DISK
    assert res.local_path("a/model") == tmp_path

    res.track_vram("a/model", pipe, vram_bytes=3 * _GiB)
    assert res.tier("a/model") is Tier.VRAM
    assert res.vram_bytes("a/model") == 3 * _GiB
    assert res.free_vram_bytes() == 7 * _GiB

    assert res.demote("a/model") is True
    assert res.tier("a/model") is Tier.RAM
    assert pipe.moves[-1] == "cpu"
    assert res.free_vram_bytes() == 10 * _GiB  # accounting released

    assert res.promote("a/model") is True
    assert res.tier("a/model") is Tier.VRAM
    assert pipe.moves[-1] == "cuda"
    assert res.vram_bytes("a/model") == 3 * _GiB  # hint restored the footprint

    assert res.release_to_disk("a/model") is True
    assert res.tier("a/model") is Tier.DISK
    assert res.obj("a/model") is None

    assert res.evict("a/model") is True
    assert res.tier("a/model") is None

    states = [(r, s, v) for r, s, v in events]
    assert states == [
        ("a/model", "on_disk", 0),
        ("a/model", "in_vram", 3 * _GiB),
        ("a/model", "in_ram", 0),
        ("a/model", "in_vram", 3 * _GiB),
        ("a/model", "on_disk", 0),
        ("a/model", "evicted", 0),
    ]


def test_track_disk_is_idempotent_no_event_spam(tmp_path: Path) -> None:
    events: list = []
    res = _res(events)
    res.track_disk("a/m", tmp_path)
    res.track_disk("a/m", tmp_path)
    assert [s for _, s, _ in events] == ["on_disk"]


def test_demote_refuses_unmovable_entries(monkeypatch) -> None:
    """demote() never books a transition it can't perform: entries without a
    movable object (tenant-loaded weights) and RAM-tight hosts are refused —
    freeing them is the owner's (executor teardown) job."""
    events: list = []
    res = _res(events)
    res.track_vram("no-obj", None, vram_bytes=1 * _GiB)
    assert res.demote("no-obj") is False
    assert res.tier("no-obj") is Tier.VRAM  # still the truth

    res.track_vram("movable", _Pipe(), vram_bytes=1 * _GiB)
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 1.0)
    assert res.demote("movable") is False  # RAM floor: no warm tier


# --------------------------------------------------------------------------- #
# LRU eviction order under make_room
# --------------------------------------------------------------------------- #


def test_make_room_demotes_in_lru_order() -> None:
    events: list = []
    res = _res(events, budget_gb=10)
    pipes = {name: _Pipe() for name in ("one", "two", "three")}
    res.track_vram("one", pipes["one"], vram_bytes=3 * _GiB)
    res.track_vram("two", pipes["two"], vram_bytes=3 * _GiB)
    res.track_vram("three", pipes["three"], vram_bytes=3 * _GiB)
    res.touch("one")  # "two" is now LRU

    # free = 10 - 9 = 1GiB; need 3 + 2 margin => demote LRU until free >= 5.
    assert res.make_room(3 * _GiB) is True
    assert res.tier("two") is None or res.tier("two") is Tier.RAM
    demoted = [r for r, s, _ in events if s == "in_ram"]
    assert demoted[0] == "two"          # LRU victim first
    assert "one" not in demoted[:1]
    assert res.free_vram_bytes() >= 5 * _GiB


def test_make_room_skips_pinned_and_executing() -> None:
    res = _res([], budget_gb=6)
    res.track_vram("pinned", _Pipe(), vram_bytes=3 * _GiB, pinned=True)
    res.track_vram("busy", _Pipe(), vram_bytes=3 * _GiB)

    with res.executing("busy"):
        # Nothing evictable: pinned + executing both protected.
        assert res.make_room(1 * _GiB) is False
        assert res.tier("pinned") is Tier.VRAM
        assert res.tier("busy") is Tier.VRAM

    # After execution finishes, "busy" is a candidate again.
    assert res.make_room(1 * _GiB) is True
    assert res.tier("busy") is Tier.RAM
    assert res.tier("pinned") is Tier.VRAM  # pinned never moves


def test_unload_refused_while_executing() -> None:
    res = _res([])
    res.track_vram("m", _Pipe(), vram_bytes=1 * _GiB)
    with res.executing("m"):
        assert res.in_use("m") is True
        assert res.release_to_disk("m") is False
        assert res.evict("m") is False
    assert res.in_use("m") is False
    assert res.release_to_disk("m") is True  # no disk path -> entry gone
    assert res.tier("m") is None


# --------------------------------------------------------------------------- #
# Snapshot shape (feeds Hello.models)
# --------------------------------------------------------------------------- #


def test_snapshot_reports_ref_tier_vram(tmp_path: Path) -> None:
    res = _res([])
    res.track_disk("d", tmp_path)
    res.track_vram("v", _Pipe(), vram_bytes=2 * _GiB)
    snap = dict((ref, (tier, vb)) for ref, tier, vb in res.snapshot())
    assert snap["d"] == (Tier.DISK, 0)
    assert snap["v"] == (Tier.VRAM, 2 * _GiB)
