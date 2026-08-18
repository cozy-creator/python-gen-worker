"""A mint must TERMINALISE on every exit — including "I am not going to publish".

Measured on `j56tate13oav13` (A40, $0.44/hr, 2026-08-18): a self-mint whose
compile children ALL finished (`status=compiled exit=0` x2, pool ledger
complete) but whose publish the hub declined (`release_unserved`) never reached
a terminal phase. `self_mint_compile inductor_compile` stayed RUNNING, the pod
read as busy on background work, and it billed for 30 minutes until a human
deleted it. The hub's stall detector fired four times and was RIGHT not to
condemn a serving pod for its background work — the defect is that the worker
never said it was finished, so a correct policy had nothing to act on.

The two shapes that matter, and nothing between them:

1. the mint runs to term and the publish is DECLINED -> one terminal
   `self_mint_skipped` carrying the hub's own refusal code as `phase`;
2. the mint runs to term and the publish LANDS -> untouched: `self_mint_publish
   phase=published` is that end's terminus and no skip is invented beside it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Optional

import pytest

from gen_worker import fleet_compiled_graphs
from gen_worker.pb import worker_scheduler_pb2 as pb

from test_eager_first_boot_pgw671 import _Harness

REFUSAL_CODE = "release_unserved"


@pytest.fixture(autouse=True)
def _clean_publish_ledger() -> Any:
    """Both rows mint the SAME stub key, so the process-wide publish ledger
    would carry one test's verdict into the other."""
    fleet_compiled_graphs._PUBLISH_OUTCOME.clear()
    yield
    fleet_compiled_graphs._PUBLISH_OUTCOME.clear()


class _Publisher:
    """The fleet sink, minus the hub. ``refuse`` is the hub's classified code."""

    def __init__(self, refuse: str = "") -> None:
        self.refuse = refuse
        self.calls = 0

    def enabled(self) -> bool:
        return True

    def publish(
        self, family: str, artifact: Path, meta: Any, provenance: Any,
        mint_duration_ms: int = 0,
    ) -> str:
        self.calls += 1
        if self.refuse:
            raise fleet_compiled_graphs.CompiledGraphPublishRefused(
                f"the release this compiled graph was built for is unserved "
                f"({self.refuse})", status=409, code=self.refuse)
        return "checkpoint-1"


def _boot_with_publisher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publisher: _Publisher,
) -> _Harness:
    """The pgw#671 rig, with a real publish sink hung off every obligation."""
    h = _Harness(tmp_path, monkeypatch, compile_delay_s=0.05)
    inner = h._fake_enable_compiled

    def _armed(*args: Any, **kwargs: Any) -> Any:
        outcome = inner(*args, **kwargs)
        # PendingSelfMint is frozen; the sink is the one fact this rig injects.
        object.__setattr__(outcome.self_mint, "publisher", publisher)
        return outcome

    monkeypatch.setattr(fleet_compiled_graphs, "enable_compiled", _armed)
    return h


def _events(h: _Harness, kind: str) -> List[Any]:
    return [
        m.activity_update for m in h.sent
        if m.WhichOneof("msg") == "activity_update"
        and m.activity_update.kind == kind
    ]


def _mint_activity_terminal(h: _Harness) -> Optional[int]:
    states = h.activity_states("self_mint_compile")
    return states[-1] if states else None


def _run_mint(h: _Harness) -> None:
    async def _go() -> None:
        await h.boot()
        await h.wait_mint(timeout=60.0)

    asyncio.run(_go())


def test_declined_publish_still_terminalises_with_the_hub_s_own_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured shape: children finish, the hub refuses, the mint ENDS."""
    publisher = _Publisher(refuse=REFUSAL_CODE)
    h = _boot_with_publisher(tmp_path, monkeypatch, publisher)

    _run_mint(h)

    assert publisher.calls == 1, "the rig never reached the publish leg"
    skipped = _events(h, "self_mint_skipped")
    assert len(skipped) == 1, (
        "a mint that will not publish must terminalise exactly once — "
        f"got {[(e.phase, e.detail) for e in skipped]}. Without it the pod "
        "holds a paid card on an activity that never ends."
    )
    assert skipped[0].phase == REFUSAL_CODE, (
        f"phase={skipped[0].phase!r} — the terminus must carry the hub's own "
        "classified code, not prose and not a shared 'ended' token: a reason "
        "that cannot be grouped cannot be acted on"
    )
    assert _mint_activity_terminal(h) == pb.ActivityState.ACTIVITY_STATE_COMPLETED


def test_a_mint_that_publishes_invents_no_terminus_beside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The publishing end is untouched — one terminus, and it is the publish."""
    publisher = _Publisher()
    h = _boot_with_publisher(tmp_path, monkeypatch, publisher)

    _run_mint(h)

    assert publisher.calls == 1
    published = [
        e for e in _events(h, "self_mint_publish") if e.phase == "published"]
    assert len(published) == 1, "the publish leg lost its own terminus"
    assert _events(h, "self_mint_skipped") == [], (
        "a mint that shipped is not skipped — a second terminal row here "
        "would double-count the only end that leaves bytes with the fleet"
    )
    assert _mint_activity_terminal(h) == pb.ActivityState.ACTIVITY_STATE_COMPLETED
