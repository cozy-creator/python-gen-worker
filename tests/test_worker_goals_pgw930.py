"""pgw#930 (§1.17) — serve and mint are COMPOSABLE goals, not an exclusive mode.

Paul, 2026-08-03: *"you can spawn a GPU and tell it to mint some AOT-cells,
while also using it to serve jobs (inference) as needed."*

Two load-bearing tests here, and neither is "the goals exist":

* :func:`test_serve_and_mint_compose` is the RED TEST for the ruling. Under the
  deleted ``worker_mode``, ``_KNOWN = (MODE_SERVE, MODE_FORGE)`` was a closed
  two-tuple with no way to spell "both", ``is_forge()`` was the only predicate,
  and three reserve terms were ``0 if forge else X``. This asserts the fourth
  combination is real everywhere it has to be: dispatch admitted, mint driven,
  tenant reserve KEPT, and the pod NOT retired when the mint finishes.
* :func:`test_goals_do_not_move_the_env_seal` is pgw#846's gate, carried over
  unchanged in force: a cell minted under a mint goal must be **the same
  artifact** a serve-only pod would have produced, and the cheapest way for a
  posture knob to break that silently is to leak into an identity axis.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import config as gw_config
from gen_worker import env_seal, fleet_cells, mint_goal, worker_goals


def _declare(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Seed the goal set the way a pod launch does — through Settings."""
    monkeypatch.setenv("WORKER_MODE", value)
    settings = gw_config.reload_for_test()
    worker_goals.install(worker_goals.from_settings(settings))


def _goals(**kw: Any) -> worker_goals.WorkerGoals:
    """Install an arbitrary goal set, as a Directive will once th#1488 lands."""
    goals = worker_goals.WorkerGoals(**kw)
    worker_goals.install(goals)
    return goals


# ---------------------------------------------------------------------------
# The goal set
# ---------------------------------------------------------------------------


def test_default_is_serve_only() -> None:
    assert gw_config.current().worker_mode == "serve"
    goals = worker_goals.from_settings(gw_config.current())
    assert (goals.serve, goals.mint) == (True, False)
    assert goals.serve_admitted() is True
    assert goals.drives_mint() is False


def test_mint_only_is_seeded_by_the_hub_injected_declaration(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(monkeypatch, "forge")
    goals = worker_goals.current()
    assert (goals.serve, goals.mint) == (False, True)
    assert goals.wire_declaration() == "forge"


def test_serve_and_mint_compose(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE RULING, asserted. The deleted `worker_mode` could not express this
    combination at all: `_KNOWN` was a two-tuple and every consumer asked
    `is_forge()`, so "both" collapsed onto whichever branch the boolean fell
    into. Each assertion below is a site that was strictly two-valued."""
    goals = _goals(serve=True, mint=True, declared="serve")

    # executor: dispatch is admitted, because a serve goal is held.
    assert goals.serve_admitted() is True
    # lifecycle: the mint driver runs, because a mint goal is held.
    assert goals.drives_mint() is True
    # mint_budget / aot_compile_pool: the tenant reserve is KEPT. Under
    # `0 if forge else X` a pod minting at all lost every reserve.
    assert goals.tenant_reserve_applies() is True
    # mint_goal driver: finishing the mint does NOT retire a pod that is also
    # serving. `forge.run` used to call start_drain unconditionally.
    assert goals.retires_when_mint_completes() is False


def test_all_four_combinations_are_expressible() -> None:
    """A blocked mint goal must never turn a still-valid serve goal into a
    refusal — which requires the two to be independent, not a bitmask."""
    matrix = {
        (True, False): (True, False, True, False),
        (False, True): (False, True, False, True),
        (True, True): (True, True, True, False),
        (False, False): (False, False, False, False),
    }
    for (serve, mint), expected in matrix.items():
        g = worker_goals.WorkerGoals(serve=serve, mint=mint)
        assert (g.serve_admitted(), g.drives_mint(),
                g.tenant_reserve_applies(),
                g.retires_when_mint_completes()) == expected, (serve, mint)


def test_unknown_declaration_seeds_serve_but_is_reported_verbatim(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A declaration from a newer hub vocabulary must not brick the pod — but
    the hub has to be able to SEE that this image did not understand it.

    pgw#846 measured the cost of dropping it: the process-split parent echoed
    the protobuf default and every mint-only pod bought was idle-reaped as
    `cold_idle_never_dispatched`."""
    _declare(monkeypatch, "quarry")
    goals = worker_goals.current()
    assert (goals.serve, goals.mint) == (True, False)
    assert goals.declaration_understood is False
    assert goals.wire_declaration() == "quarry"


# ---------------------------------------------------------------------------
# THE GATE (pgw#846): process may change, product may not
# ---------------------------------------------------------------------------


def test_the_goal_declaration_is_not_a_sealed_knob() -> None:
    """`WORKER_MODE` must never join the canonical (sealed) config table.

    A sealed knob re-digests the ``env_seal`` key axis, which would give a
    forge-minted cell a key no serving pod can ever compute — i.e. it would
    destroy the single property the forge exists to have. Same reasoning that
    keeps ``GEN_WORKER_PREFER_AOT`` out of the table.
    """
    assert "WORKER_MODE" not in env_seal.CANONICAL_CONFIG
    assert not any(
        "WORKER_MODE" in str(k) for k in env_seal.CANONICAL_CONFIG)


def test_goals_do_not_move_the_env_seal(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The `env_seal` digest is IDENTICAL whatever goals this pod holds.

    This is the mechanical half of "identity comes free": same image, same
    toolchain, same seal, therefore the same key axis — measured here, not
    asserted in a design doc.
    """
    monkeypatch.delenv("WORKER_MODE", raising=False)
    gw_config.reload_for_test()
    serve_digest = env_seal.seal_digest(env_seal.effective_seal())

    _declare(monkeypatch, "forge")
    mint_digest = env_seal.seal_digest(env_seal.effective_seal())

    assert mint_digest == serve_digest, (
        "the goal declaration moved the env_seal digest — a mint-only pod "
        "would mint cells "
        "under a key no serving pod can compute")


def test_the_goal_declaration_survives_scrub_env(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """`scrub_env` erases the behaviour namespaces; the mode is plumbing and
    must survive it, or the compute child boots as a serving worker."""
    _declare(monkeypatch, "forge")
    env_seal.scrub_env()
    import os

    assert os.environ.get("WORKER_MODE") == "forge"


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


class _FakeExecutor:
    def __init__(self, *, declares: bool = True) -> None:
        self._declares = declares

    def background_mint_tasks(self) -> List[asyncio.Task]:
        return []

    def declares_compile(self) -> bool:
        return self._declares


class _FakeLifecycle:
    def __init__(self, *, declares: bool = True) -> None:
        self.executor = _FakeExecutor(declares=declares)
        self.drains: List[int] = []

    def start_drain(self, deadline_ms: int) -> None:
        self.drains.append(deadline_ms)


@pytest.fixture()
def events(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str, str]]:
    seen: List[Tuple[str, str, str]] = []

    def _emit(kind: str, detail: str, phase: str = "", **_: Any) -> None:
        seen.append((kind, detail, phase))

    monkeypatch.setattr(mint_goal.activity_mod, "emit_event", _emit)
    return seen


@pytest.fixture(autouse=True)
def _reset_mint_goal_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mint_goal, "_disposition", None)
    monkeypatch.setattr(mint_goal, "_PUBLISH_POLL_S", 0.01)


def _ledger(monkeypatch: pytest.MonkeyPatch, published: Dict[str, str],
            refused: Dict[str, str]) -> None:
    monkeypatch.setattr(fleet_cells, "publishes_in_flight", lambda: {})
    monkeypatch.setattr(fleet_cells, "published_cells", lambda: dict(published))
    monkeypatch.setattr(fleet_cells, "refused_publishes", lambda: dict(refused))


def _run(lc: _FakeLifecycle) -> None:
    async def go() -> None:
        task = asyncio.create_task(mint_goal.run(lc))  # type: ignore[arg-type]
        await asyncio.sleep(0)
        mint_goal.note_disposition_final()
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_serving_worker_never_runs_the_driver(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    monkeypatch.delenv("WORKER_MODE", raising=False)
    gw_config.reload_for_test()
    lc = _FakeLifecycle()
    asyncio.run(mint_goal.run(lc))  # type: ignore[arg-type]
    assert lc.drains == []
    assert events == []


def test_published_cell_is_a_typed_terminal_then_retire(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    _declare(monkeypatch, "forge")
    _ledger(monkeypatch, {"ck5-abc": "sha256:deadbeef"}, {})
    lc = _FakeLifecycle()
    _run(lc)
    kinds = [(k, p) for k, _, p in events]
    assert (mint_goal.KIND_MINT_GOAL, mint_goal.TERMINAL_PUBLISHED) in kinds
    assert lc.drains == [mint_goal.MINT_GOAL_DRAIN_DEADLINE_MS]


def test_refused_publish_is_distinguished_from_no_cell(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    """"minted but the hub said no" and "never produced a cell" are different
    operational facts and must not share a terminal."""
    _declare(monkeypatch, "forge")
    _ledger(monkeypatch, {}, {"ck5-abc": "cell_publish_untrusted_compute"})
    lc = _FakeLifecycle()
    _run(lc)
    assert (mint_goal.KIND_MINT_GOAL, mint_goal.TERMINAL_REFUSED) in [
        (k, p) for k, _, p in events]
    assert lc.drains == [mint_goal.MINT_GOAL_DRAIN_DEADLINE_MS]


def test_no_cell_still_retires(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    _declare(monkeypatch, "forge")
    _ledger(monkeypatch, {}, {})
    lc = _FakeLifecycle()
    _run(lc)
    assert (mint_goal.KIND_MINT_GOAL, mint_goal.TERMINAL_FAILED) in [
        (k, p) for k, _, p in events]
    assert lc.drains == [mint_goal.MINT_GOAL_DRAIN_DEADLINE_MS]


def test_release_with_no_compile_declaration_says_nothing_owed(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    """A mint-only pod bought for a release that declares no compile family
    must say so and retire — not idle on a paid card."""
    _declare(monkeypatch, "forge")
    _ledger(monkeypatch, {}, {})
    lc = _FakeLifecycle(declares=False)
    _run(lc)
    assert (mint_goal.KIND_MINT_GOAL, mint_goal.TERMINAL_NOTHING_OWED) in [
        (k, p) for k, _, p in events]


def test_driver_waits_for_the_publish_to_land(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    """"this pod must survive the upload or the cell is lost" — a pod with no
    serve goal is the one pod that can simply wait, so it must."""
    _declare(monkeypatch, "forge")
    state = {"n": 3}

    def _in_flight() -> Dict[str, Any]:
        if state["n"] > 0:
            state["n"] -= 1
            return {"ck5-abc": ("sdxl", 0.0)}
        return {}

    monkeypatch.setattr(fleet_cells, "publishes_in_flight", _in_flight)
    monkeypatch.setattr(fleet_cells, "published_cells", lambda: {"ck5-abc": "cp"})
    monkeypatch.setattr(fleet_cells, "refused_publishes", lambda: {})
    lc = _FakeLifecycle()
    _run(lc)
    assert state["n"] == 0
    assert lc.drains == [mint_goal.MINT_GOAL_DRAIN_DEADLINE_MS]


def test_driver_failure_still_retires_the_pod(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    """A driver that dies must not leave a pod that mints nothing and never
    retires — that is strictly worse than any outcome it could report."""
    _declare(monkeypatch, "forge")

    def _boom() -> Dict[str, Any]:
        raise RuntimeError("ledger exploded")

    monkeypatch.setattr(fleet_cells, "publishes_in_flight", _boom)
    lc = _FakeLifecycle()
    _run(lc)
    assert lc.drains == [mint_goal.MINT_GOAL_DRAIN_DEADLINE_MS]
    assert any(p == mint_goal.TERMINAL_FAILED for _, _, p in events)


# ---------------------------------------------------------------------------
# The dual-goal driver behaviour — the half a boolean could not have
# ---------------------------------------------------------------------------


def test_dual_goal_pod_states_its_terminal_but_does_NOT_retire(
        monkeypatch: pytest.MonkeyPatch,
        events: List[Tuple[str, str, str]]) -> None:
    """Paul's case, end to end through the driver.

    The pod holds a serve goal AND a mint goal. It must publish its cell, state
    the same typed terminal a mint-only pod would state — and then keep
    serving. `forge.run` called `start_drain` unconditionally, so before
    pgw#930 this pod would have torn down its own live serving instance the
    moment its mint finished.
    """
    _goals(serve=True, mint=True, declared="serve")
    _ledger(monkeypatch, {"ck5-abc": "sha256:deadbeef"}, {})
    lc = _FakeLifecycle()
    _run(lc)
    assert (mint_goal.KIND_MINT_GOAL, mint_goal.TERMINAL_PUBLISHED) in [
        (k, p) for k, _, p in events], "the terminal must still be stated"
    assert lc.drains == [], (
        "a pod holding a serve goal was drained when its mint finished — "
        "that is the exclusivity assumption pgw#930 deleted")


def test_a_blocked_mint_does_not_refuse_a_valid_serve_goal(
        monkeypatch: pytest.MonkeyPatch,
        events: List[Tuple[str, str, str]]) -> None:
    """§1.17, verbatim: "A blocked mint goal never turns a still-valid serve
    goal into a refusal." The mint fails hard; serve admission is untouched."""
    _goals(serve=True, mint=True, declared="serve")

    def _boom() -> Dict[str, Any]:
        raise RuntimeError("ledger exploded")

    monkeypatch.setattr(fleet_cells, "publishes_in_flight", _boom)
    lc = _FakeLifecycle()
    _run(lc)
    assert any(p == mint_goal.TERMINAL_FAILED for _, _, p in events)
    assert lc.drains == [], "a failed mint goal retired a serving pod"
    assert worker_goals.current().serve_admitted() is True, (
        "a failed mint goal revoked the serve goal")
