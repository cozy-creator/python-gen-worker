"""The trace context answers the serving context's surface — SAME KINDS.

pgw#1510. `TraceRequestContext.log` was a bound `logging.Logger` while the
serving `RequestContext.log` is a METHOD, so the documented author line

    ctx.log("resolved scheduler: %s" % name)

died `TypeError: 'Logger' object is not callable` in the middle of the drive —
on a line that is correct at serve. h3's `generate` hit it and its 0.11.3 went
out lockless.

This is pgw#1461's shape one turn later. That issue grew the trace surface from
five members to the serving contract's ~48; what nothing checked is that each
member is the same KIND. A name that exists but is not callable is worse than
a name that is missing: `hasattr` says yes, review says yes, and it fails at
drive time.

So the fence compares KINDS mechanically rather than pinning `log` alone —
pinning the instance would have let the next one through.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from gen_worker.release.trace_context import TraceLoadContext, TraceRequestContext
from gen_worker.request_context import RequestContext
from gen_worker.serving.context import LoadContext


def _serving_kind(owner: type, name: str) -> str | None:
    """"callable" / "value" for a member DEFINED on the serving class."""

    for klass in owner.__mro__:
        if name in klass.__dict__:
            member = klass.__dict__[name]
            if isinstance(member, property):
                return "value"
            if inspect.isfunction(member) or inspect.ismethod(member):
                return "callable"
            return "value"
    return None


def _trace_kind(instance: Any, name: str) -> str:
    return "callable" if callable(getattr(instance, name)) else "value"


def _pairs(serving: type, instance: Any) -> list[tuple[str, str, str]]:
    rows = []
    for name in sorted(n for n in dir(serving) if not n.startswith("_")):
        serving_kind = _serving_kind(serving, name)
        if serving_kind is None or not hasattr(instance, name):
            continue
        rows.append((name, serving_kind, _trace_kind(instance, name)))
    return rows


@pytest.fixture
def request_ctx() -> TraceRequestContext:
    return TraceRequestContext(lane=None, checkpoint_ref="trace:x", step_budget=1)


def test_every_shared_request_member_has_the_SAME_KIND(
    request_ctx: TraceRequestContext,
) -> None:
    """A member the trace answers must be callable iff serving's is.

    The whole defect in one assertion, and it is a SWEEP: whatever the trace
    surface grows next is covered the day it is added.
    """

    wrong = [
        f"{name}: serving={serving}, trace={trace}"
        for name, serving, trace in _pairs(RequestContext, request_ctx)
        if serving != trace
    ]
    assert wrong == []


def test_the_sweep_actually_covers_something(
    request_ctx: TraceRequestContext,
) -> None:
    """A green sweep over an empty set proves nothing (fixture set-then-assert).

    `log` in particular must be IN the compared set — it is the member that
    caused this issue, and a comparison that silently skipped it would read
    exactly as green.
    """

    compared = {name for name, _s, _t in _pairs(RequestContext, request_ctx)}
    assert len(compared) > 20, f"only {len(compared)} members compared"
    assert "log" in compared


def test_ctx_log_is_callable_with_the_documented_author_spelling(
    request_ctx: TraceRequestContext,
) -> None:
    """The verbatim failing line, and the kwargs form the serving one takes."""

    request_ctx.log("resolved scheduler: euler")
    request_ctx.log("degraded", level="warn", reason="no fp8 engine")


def test_the_load_context_shares_no_member_of_a_different_kind() -> None:
    """The same sweep for the load half.

    `TraceLoadContext` keeps a private logger, which is fine BECAUSE the
    serving `LoadContext` exposes no `log` at all — asserted rather than
    assumed, since that is the only reason the private name is safe.
    """

    load_ctx = TraceLoadContext(lane=None, checkpoint_dir=Path("."))
    wrong = [
        f"{name}: serving={serving}, trace={trace}"
        for name, serving, trace in _pairs(LoadContext, load_ctx)
        if serving != trace
    ]
    assert wrong == []
    assert _serving_kind(LoadContext, "log") is None


def test_the_derive_logger_is_private_so_the_public_name_stays_the_method(
    request_ctx: TraceRequestContext,
) -> None:
    import logging

    assert isinstance(request_ctx._log, logging.Logger)
    assert not isinstance(request_ctx.log, logging.Logger)
