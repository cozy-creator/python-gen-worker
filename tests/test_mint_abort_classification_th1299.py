"""th#1299 — an abandoned mint must say WHICH cause abandoned it.

The abort event reported ``phase="abandoned"`` with the detail
``"(adopt-on-arm / vacate / shutdown)"`` — three unrelated causes in one
string, on the only wire record the hub keeps. 41 such rows on the master
stack could not be triaged from worker evidence alone; the real cause (the
hub's idle-retire destroying the pod mid-mint) was only found by joining
``worker_activity_events`` to ``worker_pods.retire_reason`` by hand.

A cause that only exists in a disjunction is not classified, and Paul's rule
is that a failure reaches the hub carrying its classified reason.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

import pytest

from test_eager_first_boot_pgw671 import _Harness


def _abort_events(h: _Harness) -> List[object]:
    return [
        m.activity_update for m in h.sent
        if m.WhichOneof("msg") == "activity_update"
        and m.activity_update.kind == "self_mint_abort"
    ]


@pytest.mark.parametrize(
    "code,reason",
    [
        ("adopt_on_arm", "adopting peer compiled graph repo/compiled graph"),
        ("vacate", "instance vacate"),
        ("shutdown", "worker shutdown"),
        ("tenant_oom", "tenant OOM — the mint loses, the request wins"),
    ],
)
def test_abandon_reports_its_own_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str, reason: str,
) -> None:
    h = _Harness(tmp_path, monkeypatch, compile_delay_s=2.0)

    async def _run() -> None:
        await h.boot()
        rec = h.rec
        assert rec.background_mint is not None
        await h.ex.abandon_background_mint(rec, reason=reason, code=code)

    asyncio.run(_run())

    aborts = _abort_events(h)
    assert len(aborts) == 1, f"expected one abort event, got {aborts}"
    ev = aborts[0]
    assert ev.phase == f"abandoned_{code}", (
        f"phase={ev.phase!r} — the cause must be the queryable field, not "
        "prose; 'abandoned' alone cannot distinguish a superseded mint from "
        "a killed one"
    )
    assert reason in ev.detail
    assert "adopt-on-arm / vacate / shutdown" not in ev.detail, (
        "the disjunction is back — every cause reads as every other cause"
    )


def test_unclassified_abandon_is_still_honest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller that forgets the code gets ``unspecified`` — a legible gap,
    never a plausible-looking wrong cause."""
    h = _Harness(tmp_path, monkeypatch, compile_delay_s=2.0)

    async def _run() -> None:
        await h.boot()
        assert h.rec.background_mint is not None
        await h.ex.abandon_background_mint(h.rec, reason="something happened")

    asyncio.run(_run())

    (ev,) = _abort_events(h)
    assert ev.phase == "abandoned_unspecified"
