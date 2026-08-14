"""pgw#1206 C3: residency's two mutators live in `models.records` and reach
the mint supervisor ONLY through th#1834's injected seam.

The ruling (th#1834, 2026-08-13): `abandon_background_mint` stays with the
mint supervisor and is exposed to residency as an injected callable
`(rec, *, reason, code="unspecified", free_targets=False)`. `vacate_record`
and `shutdown_instances` are residency operations that must first tell the
supervisor to stop — residency CALLS supervision.

These rows exercise the seam through the REAL supervisor body (a real
`_BackgroundMint` on a real `Executor`), so they go red if the call is
dropped, if the direction is inverted, or if the ruled reason/code stop being
what a stopped mint reports.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import msgspec

from gen_worker.api.binding import Hub
from gen_worker.api.decorators import Resources
from gen_worker.executor import Executor, _BackgroundMint
from gen_worker.models.records import (
    RecordTeardown,
    shutdown_instances,
    vacate_record,
)
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _In(msgspec.Struct):
    prompt: str = ""


class _Fake:
    def setup(self, pipeline: Any) -> None:  # pragma: no cover - never run here
        self.pipeline = pipeline

    def generate(self, ctx: Any, payload: _In) -> Dict[str, Any]:  # pragma: no cover
        return {}


def _executor() -> Executor:
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    spec = EndpointSpec(
        name="generate", method=_Fake.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=_Fake,
        models={"pipeline": Hub("acme/z-image")},
        resources=Resources(gpu=True),
    )
    return Executor([spec], _send)


def _ready_record_with_mint(ex: Executor) -> Tuple[Any, Any]:
    spec = ex.specs["generate"]
    rec = ex._classes[spec.instance_key]
    rec.instance = _Fake()
    rec.ready = True
    bg = _BackgroundMint(
        spec=spec, instance=rec.instance, snapshots=None, pendings={}, pipes={})
    rec.background_mint = bg
    return rec, bg


def test_vacate_record_stops_the_mint_through_the_seam() -> None:
    """pgw#671 through th#1834's seam: a departing instance takes its
    background mint with it, and the supervisor is told WHY."""

    async def _run() -> None:
        ex = _executor()
        rec, bg = _ready_record_with_mint(ex)

        released = await vacate_record(rec, ex.teardown_seam)

        assert released == []
        assert bg.abandon.is_set(), "the supervisor was never signalled"
        assert (bg.abandon_reason, bg.abandon_code) == ("instance vacate", "vacate")
        assert rec.background_mint is None
        assert rec.instance is None and rec.ready is False

    asyncio.run(_run())


def test_shutdown_instances_stops_every_mint_through_the_seam() -> None:
    async def _run() -> None:
        ex = _executor()
        rec, bg = _ready_record_with_mint(ex)

        await shutdown_instances(ex.teardown_seam)

        assert bg.abandon.is_set()
        assert (bg.abandon_reason, bg.abandon_code) == ("worker shutdown", "shutdown")
        assert rec.background_mint is None
        assert rec.instance is None and rec.ready is False

    asyncio.run(_run())


def test_the_seam_is_the_supervisor_and_nothing_else() -> None:
    """The record book owns no abandonment of its own: what it calls IS the
    executor's supervisor method, injected, with the ruled keyword contract."""

    ex = _executor()
    seam = ex.teardown_seam

    assert isinstance(seam, RecordTeardown)
    assert seam.abandon_background_mint == ex.abandon_background_mint

    import inspect

    sig = inspect.signature(ex.abandon_background_mint)
    assert list(sig.parameters) == ["rec", "reason", "code", "free_targets"]
    assert sig.parameters["code"].default == "unspecified"
    assert sig.parameters["free_targets"].default is False
    for kw in ("reason", "code", "free_targets"):
        assert sig.parameters[kw].kind is inspect.Parameter.KEYWORD_ONLY

    # Residency calls supervision, never the reverse: the record book must not
    # import the executor, at module scope or inside a function.
    import ast

    import gen_worker.models.records as records_mod

    tree = ast.parse(inspect.getsource(records_mod))
    imported: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(f"{'.' * node.level}{node.module or ''}")
    assert not [m for m in imported if "executor" in m], imported


def test_state_change_is_read_at_call_time_not_construction() -> None:
    """worker.py assigns `_on_state_change` after `Executor.__init__`, so a
    seam that captured it at construction would call the placeholder."""

    async def _run() -> None:
        ex = _executor()
        rec, _bg = _ready_record_with_mint(ex)
        ticks: List[int] = []
        ex._on_state_change = lambda: ticks.append(1)

        await vacate_record(rec, ex.teardown_seam)

        assert ticks, "the vacate never reported a state change"

    asyncio.run(_run())
