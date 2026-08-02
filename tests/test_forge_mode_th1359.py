"""th#1359 Part 2 — the forge worker mode.

The load-bearing test in this file is not "the mode exists"; it is
:func:`test_forge_mode_does_not_move_the_env_seal`. pgw#846's gate is that a
forge-minted cell must be **the same artifact** a serving pod would have
produced, and the cheapest way for a mode knob to break that silently is to
leak into an identity axis. ``env_seal`` is the axis it would leak into, so
that is the one this file proves by construction rather than by assertion.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import env_seal, fleet_cells, forge, worker_mode
from gen_worker.config.loader import get_settings


def _mode(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("WORKER_MODE", value)
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# The mode itself
# ---------------------------------------------------------------------------


def test_default_is_serve() -> None:
    assert get_settings().worker_mode == "serve"
    assert worker_mode.mode() == worker_mode.MODE_SERVE
    assert worker_mode.is_forge() is False


def test_forge_is_selected_by_the_hub_injected_env(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _mode(monkeypatch, "forge")
    assert worker_mode.is_forge() is True
    assert worker_mode.declared() == "forge"


def test_unknown_mode_reads_as_serve_but_is_reported_verbatim(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A mode from a newer hub vocabulary must not brick the pod — but the hub
    has to be able to SEE that this image did not understand it."""
    _mode(monkeypatch, "quarry")
    assert worker_mode.mode() == worker_mode.MODE_SERVE
    assert worker_mode.is_forge() is False
    assert worker_mode.declared() == "quarry"


# ---------------------------------------------------------------------------
# THE GATE (pgw#846): process may change, product may not
# ---------------------------------------------------------------------------


def test_worker_mode_is_not_a_sealed_knob() -> None:
    """`WORKER_MODE` must never join the canonical (sealed) config table.

    A sealed knob re-digests the ``env_seal`` key axis, which would give a
    forge-minted cell a key no serving pod can ever compute — i.e. it would
    destroy the single property the forge exists to have. Same reasoning that
    keeps ``GEN_WORKER_PREFER_AOT`` out of the table.
    """
    assert worker_mode.ENV_WORKER_MODE not in env_seal.CANONICAL_CONFIG
    assert not any(
        "WORKER_MODE" in str(k) for k in env_seal.CANONICAL_CONFIG)


def test_forge_mode_does_not_move_the_env_seal(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The `env_seal` digest is IDENTICAL with and without forge mode.

    This is the mechanical half of "identity comes free": same image, same
    toolchain, same seal, therefore the same key axis — measured here, not
    asserted in a design doc.
    """
    monkeypatch.delenv("WORKER_MODE", raising=False)
    get_settings.cache_clear()
    serve_digest = env_seal.seal_digest(env_seal.effective_seal())

    _mode(monkeypatch, "forge")
    forge_digest = env_seal.seal_digest(env_seal.effective_seal())

    assert forge_digest == serve_digest, (
        "WORKER_MODE moved the env_seal digest — a forge pod would mint cells "
        "under a key no serving pod can compute")


def test_worker_mode_is_scrubbed_by_nothing(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """`scrub_env` erases the behaviour namespaces; the mode is plumbing and
    must survive it, or the compute child boots as a serving worker."""
    _mode(monkeypatch, "forge")
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

    monkeypatch.setattr(forge.activity_mod, "emit_event", _emit)
    return seen


@pytest.fixture(autouse=True)
def _reset_forge_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forge, "_disposition", None)
    monkeypatch.setattr(forge, "_PUBLISH_POLL_S", 0.01)


def _ledger(monkeypatch: pytest.MonkeyPatch, published: Dict[str, str],
            refused: Dict[str, str]) -> None:
    monkeypatch.setattr(fleet_cells, "publishes_in_flight", lambda: {})
    monkeypatch.setattr(fleet_cells, "published_cells", lambda: dict(published))
    monkeypatch.setattr(fleet_cells, "refused_publishes", lambda: dict(refused))


def _run(lc: _FakeLifecycle) -> None:
    async def go() -> None:
        task = asyncio.create_task(forge.run(lc))  # type: ignore[arg-type]
        await asyncio.sleep(0)
        forge.note_disposition_final()
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_serving_worker_never_runs_the_driver(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    monkeypatch.delenv("WORKER_MODE", raising=False)
    get_settings.cache_clear()
    lc = _FakeLifecycle()
    asyncio.run(forge.run(lc))  # type: ignore[arg-type]
    assert lc.drains == []
    assert events == []


def test_published_cell_is_a_typed_terminal_then_retire(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    _mode(monkeypatch, "forge")
    _ledger(monkeypatch, {"ck5-abc": "sha256:deadbeef"}, {})
    lc = _FakeLifecycle()
    _run(lc)
    kinds = [(k, p) for k, _, p in events]
    assert (forge.KIND_FORGE, forge.TERMINAL_PUBLISHED) in kinds
    assert lc.drains == [forge.FORGE_DRAIN_DEADLINE_MS]


def test_refused_publish_is_distinguished_from_no_cell(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    """"minted but the hub said no" and "never produced a cell" are different
    operational facts and must not share a terminal."""
    _mode(monkeypatch, "forge")
    _ledger(monkeypatch, {}, {"ck5-abc": "cell_publish_untrusted_compute"})
    lc = _FakeLifecycle()
    _run(lc)
    assert (forge.KIND_FORGE, forge.TERMINAL_REFUSED) in [
        (k, p) for k, _, p in events]
    assert lc.drains == [forge.FORGE_DRAIN_DEADLINE_MS]


def test_no_cell_still_retires(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    _mode(monkeypatch, "forge")
    _ledger(monkeypatch, {}, {})
    lc = _FakeLifecycle()
    _run(lc)
    assert (forge.KIND_FORGE, forge.TERMINAL_FAILED) in [
        (k, p) for k, _, p in events]
    assert lc.drains == [forge.FORGE_DRAIN_DEADLINE_MS]


def test_release_with_no_compile_declaration_says_nothing_owed(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    """A forge pod bought for a release that declares no compile family must
    say so and retire — not idle on a paid card."""
    _mode(monkeypatch, "forge")
    _ledger(monkeypatch, {}, {})
    lc = _FakeLifecycle(declares=False)
    _run(lc)
    assert (forge.KIND_FORGE, forge.TERMINAL_NOTHING_OWED) in [
        (k, p) for k, _, p in events]


def test_driver_waits_for_the_publish_to_land(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    """"this pod must survive the upload or the cell is lost" — a forge pod is
    the one pod that can simply wait, so it must."""
    _mode(monkeypatch, "forge")
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
    assert lc.drains == [forge.FORGE_DRAIN_DEADLINE_MS]


def test_driver_failure_still_retires_the_pod(
        monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]]) -> None:
    """A driver that dies must not leave a pod that mints nothing and never
    retires — that is strictly worse than any outcome it could report."""
    _mode(monkeypatch, "forge")

    def _boom() -> Dict[str, Any]:
        raise RuntimeError("ledger exploded")

    monkeypatch.setattr(fleet_cells, "publishes_in_flight", _boom)
    lc = _FakeLifecycle()
    _run(lc)
    assert lc.drains == [forge.FORGE_DRAIN_DEADLINE_MS]
    assert any(p == forge.TERMINAL_FAILED for _, _, p in events)
