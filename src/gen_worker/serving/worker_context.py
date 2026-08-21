"""The ONE place the worker builds a production :class:`LoadContext`.

pgw#1549. There are two objects in this repo that serve an endpoint's requests
— :class:`~gen_worker.serving.host.EndpointHost` (the local CLI and the daemon)
and :class:`~gen_worker.serving.serve_loop.ServeLoop` (the pod) — and each of
them assembled its own load context from its own copy of the worker's
decisions. That is not a style problem. It is the shape of the ~21 h fleet
outage, three times over:

* **pgw#1452** handed the placement device down through ``EndpointHost`` and
  not through ``ServeLoop``. On a pod ``LoadContext._device`` was ``""``, so
  ``_placed`` returned every eagerly-bridged pipeline UNPLACED — the CPU — for
  the entire life of the v2 worker. Nothing failed; it was simply the wrong
  processor, which is the exact defect pgw#1452 was filed to delete and which
  survived on the half of the fleet that matters.
* **pgw#1380/#1544** bound the streaming engine in ``EndpointHost`` and not in
  ``ServeLoop``. Every pod fell to the eager bridge and met a pointer stub.
* **pgw#1543** put the pin repair on ``announce_resident`` and
  ``_materialize_local``, neither of which is on the path that refuses.

Each fix landed on the caller somebody was testing. The tests passed because
the tests were the caller. **A capability that two callers must both have does
not get added to both callers — it gets constructed once, where neither can
omit it.**

So: nothing outside this module builds a production ``LoadContext``.
A BARE ``LoadContext(...)`` remains inert on purpose — it names no device and
binds no engine — because the derive/trace path (``release/derive.py``) must
not acquire either, and pgw#1452's arm 2 asserts exactly that.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .context import DeployBinding, LoadContext, LoaderEngine
from .placement import serving_device

__all__ = ["worker_load_context"]


def worker_load_context(
    *,
    binding: DeployBinding,
    model_type: Optional[type] = None,
    lane: Any = None,
    resolved: Any = None,
    compile_sink: Optional[Callable[[Any], Any]] = None,
    engine: Optional[LoaderEngine] = None,
    device: str = "",
    io: str = "buffered",
    weight_budget_bytes: int = 0,
) -> LoadContext[Any]:
    """A ``LoadContext`` carrying every decision the WORKER owns.

    ``device`` is measured when the caller states none — ``serving_device()``,
    the same probe ``EndpointHost`` uses. The literal ``"cuda"`` would be an
    enumeration of what a pod usually is rather than a measurement of this
    machine (pgw#1452), and ``""`` would be the pod bug this module exists to
    close.

    ``engine`` is an explicit OVERRIDE and production passes nothing: since
    pgw#1544 ``ctx.load`` asks ``engine_for`` itself, at the one place that
    always has the tree. Pre-binding it here would restore the two-constructors
    state — two spellings of one decision, free to drift.
    """

    return LoadContext(
        binding=binding,
        model_type=model_type,
        lane=lane,
        resolved=resolved,
        engine=engine,
        compile_sink=compile_sink,
        device=str(device or "") or serving_device(),
        io=str(io or "buffered"),
        weight_budget_bytes=weight_budget_bytes,
    )
