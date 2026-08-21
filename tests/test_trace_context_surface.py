"""The trace context answers the serving context's surface — SAME KINDS."""

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
    """A member the trace answers must be callable iff serving's is."""

    wrong = [
        f"{name}: serving={serving}, trace={trace}"
        for name, serving, trace in _pairs(RequestContext, request_ctx)
        if serving != trace
    ]
    assert wrong == []


def test_the_sweep_actually_covers_something(
    request_ctx: TraceRequestContext,
) -> None:
    """A green sweep over an empty set proves nothing (fixture set-then-assert)."""

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
    """The fence the first pass missed, and the reason it missed it."""

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
    """Red-proof the fence itself, or it is decoration."""

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
