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
import logging
from pathlib import Path
from typing import Any

import pytest

from gen_worker.release.trace_context import TraceLoadContext, TraceRequestContext
from gen_worker.request_context import JobContext, RequestContext
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
    """The same sweep for the load half."""

    load_ctx = TraceLoadContext(lane=None, checkpoint_dir=Path("."))
    wrong = [
        f"{name}: serving={serving}, trace={trace}"
        for name, serving, trace in _pairs(LoadContext, load_ctx)
        if serving != trace
    ]
    assert wrong == []


#: The serving classes each trace class is the trace-time counterpart of.
#: The request side is a FAMILY: `JobContext` carries members `RequestContext`
#: does not (`checkpoint_dir`), and an entrypoint can receive either.
_COUNTERPARTS: dict[str, tuple[type, ...]] = {
    "load": (LoadContext,),
    "request": (RequestContext, JobContext),
}


def _family_kind(family: tuple[type, ...], name: str) -> str | None:
    for owner in family:
        kind = _serving_kind(owner, name)
        if kind is not None:
            return kind
    return None


def _sibling(which: str) -> tuple[type, ...]:
    return _COUNTERPARTS["request" if which == "load" else "load"]


@pytest.mark.parametrize("which", ["load", "request"])
def test_a_trace_context_never_borrows_a_name_the_contract_gave_ITS_SIBLING(
    which: str,
) -> None:
    """The fence the first pass missed, and the reason it missed it.

    The shared-member sweep compares only names present on BOTH sides, so a
    trace class carrying a name its OWN counterpart does not define is
    invisible to it. `TraceLoadContext.log` was exactly that: serving
    `LoadContext` has no `log`, so nothing paired with it, while
    `RequestContext.log` IS a method — an author who logged inside `load()`
    got "'Logger' object is not callable" where serve gives a plain
    AttributeError.

    So: a name its own counterpart does not define, but the SIBLING context
    does, must not be public here at all. The authoring contract already gave
    that word a meaning; a trace context answering something else under it is
    the trap.

    Deliberately NOT a flat union of both contexts — that produced a false
    positive on `checkpoint_dir`, which genuinely means two different things
    (a bound-tree property on `LoadContext`, a keyed scratch method on
    `RequestContext`). Each trace class is judged against ITS OWN counterpart
    first; the sibling only decides whether a name is spoken for.
    """

    instance: Any = (
        TraceLoadContext(lane=None, checkpoint_dir=Path("."))
        if which == "load"
        else TraceRequestContext(lane=None, checkpoint_ref="x", step_budget=1)
    )
    own, sibling = _COUNTERPARTS[which], _sibling(which)
    borrowed = [
        f"{name}: the contract calls it {_family_kind(sibling, name)}, "
        f"this context answers {_trace_kind(instance, name)}"
        for name in sorted(n for n in dir(instance) if not n.startswith("_"))
        if _family_kind(own, name) is None
        and _family_kind(sibling, name) is not None
        and _family_kind(sibling, name) != _trace_kind(instance, name)
    ]
    assert borrowed == []


def test_that_fence_would_have_CAUGHT_the_defect_it_was_written_for() -> None:
    """Red-proof the fence itself, or it is decoration.

    A sweep that cannot go red is worth nothing, and this one is a sweep over
    a `dir()` — the easiest kind to make vacuously green.
    """

    # The precondition the rule rests on: `log` is spoken for on the request
    # context and absent from the load one. If that ever stops being true the
    # fence stops meaning anything, so it is asserted rather than assumed.
    assert _family_kind(_COUNTERPARTS["request"], "log") == "callable"
    assert _family_kind(_COUNTERPARTS["load"], "log") is None

    class Regressed:
        log = logging.getLogger("would-be-regression")

    borrowed = [
        name
        for name in dir(Regressed())
        if not name.startswith("_")
        and _family_kind(_COUNTERPARTS["load"], name) is None
        and _family_kind(_COUNTERPARTS["request"], name) == "callable"
        and _trace_kind(Regressed(), name) != "callable"
    ]
    assert borrowed == ["log"]
