"""pgw#1461: the derive answers the WHOLE author surface, or it answers none.

The trace context presented FIVE members while the serving ``RequestContext``
presents 48. The authoring guide's own canonical example --
``view = ctx.for_request(self.pipeline, seed=42)``, docs/endpoint-authoring.md
line 51 -- raised ``AttributeError`` straight into a hard ``DeriveError``, so
most real endpoints could not derive at all. H3's derive died at the
component-attachment seam for exactly this reason.

Every release fixture passed, and that is the part worth remembering: all eight
of them dodged the hole by calling none of the missing members. A surface gap
is invisible to any test that does not exercise the surface, which is why the
first test here is a FENCE over the member set rather than a sample of calls.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from gen_worker.release.trace_context import (
    TraceRequestContext,
    TraceSurfaceUnavailable,
)


class _Lane:
    contract = "trace-lane@1"
    dtype = "float32"


def _ctx(**kwargs: Any) -> TraceRequestContext:
    return TraceRequestContext(lane=_Lane(), **kwargs)


def test_the_trace_context_answers_every_member_the_serving_one_does() -> None:
    """THE fence. A new serving member without a trace answer fails here.

    Not a sample of calls: the defect was a hole nobody's test happened to
    step in, so the assertion has to be over the SET. Fixing this once
    without the fence buys a few months at most -- the surface grows.
    """

    from gen_worker.serving.context import RequestContext as Serving

    serving = {name for name in dir(Serving) if not name.startswith("_")}
    trace = {name for name in dir(TraceRequestContext) if not name.startswith("_")}
    # Instance attributes are set in __init__, so `dir()` on the class misses
    # them; name them rather than instantiate, so the fence needs no fixture.
    trace |= {"lane", "step_budget", "checkpoint_ref", "log"}

    missing = sorted(serving - trace)
    assert not missing, (
        f"the derive runs author code AS-IS, so a member the serving context "
        f"answers and the trace context does not is an endpoint that cannot "
        f"be derived at all -- it raises AttributeError into a hard "
        f"DeriveError. {len(missing)} unanswered: {missing}. Answer each one "
        f"really, as a recorder, or as a stated empty; refuse by name "
        f"(TraceSurfaceUnavailable) only when the member's whole content is "
        f"bytes or a peer that a trace genuinely does not have."
    )


def test_the_guides_canonical_example_derives_clean() -> None:
    """docs/endpoint-authoring.md line 51, verbatim -- the reported defect.

    ``ctx.for_request`` is answered for REAL, not stubbed: the view it builds
    clones schedulers, and a different scheduler is a different denoise call
    is a different observed graph. A stub returning the pipeline unchanged
    would derive graphs for a pipeline no request ever runs.
    """

    class Pipeline:
        def __init__(self) -> None:
            self.scheduler = _Scheduler()

    class _Scheduler:
        config: dict[str, Any] = {}

        @classmethod
        def from_config(cls, config: Any, **_: Any) -> "_Scheduler":
            del config
            return cls()

    pipeline = Pipeline()
    view = _ctx().for_request(pipeline, seed=42)
    assert view is not None
    # The view is a VIEW: the module objects are shared, so a bound compiled
    # graph stays bound.
    assert getattr(view, "scheduler", None) is not None


def test_the_recorders_record_and_a_clamp_that_moved_says_so() -> None:
    """"trace records nothing" was the old comment, and it was the bug.

    A clamp changes which arm executes and therefore which graph is observed.
    Leaving no trace of one having happened means the derive cannot report
    that the graph it observed is not the graph the payload asked for.
    """

    ctx = _ctx()
    assert ctx.clamp("steps", 50, hi=28) == 28.0
    assert ctx.clamp("steps", 20, hi=28) == 20.0, "an untouched value is not an adjustment"
    assert [row["field"] for row in ctx.adjustments] == ["steps"]

    ctx.warn("scheduler fell back")
    assert ctx.warnings == ("scheduler fell back",)

    ctx.progress(0.5, "denoise", step=14)
    assert ctx.progress_events == [(0.5, "denoise", {"step": 14})]


def test_stated_empties_are_stated_and_egress_returns_stub_refs() -> None:
    ctx = _ctx()
    assert ctx.models == {} and ctx.loras == {} and ctx.config == {}
    assert ctx.cancelled is False and ctx.publishes is False
    # TRUE on purpose: the media egress path is code whose graphs the derive
    # must observe, so an endpoint must not branch away from it at trace.
    assert ctx.emits_media is True
    assert ctx.execution_lane == "trace-lane@1"

    assert ctx.save_image(object()).ref.startswith("trace://")
    assert ctx.save_video(b"").ref.startswith("trace://")
    assert ctx.save_audio(b"").ref.startswith("trace://")
    assert ctx.save_bytes("out.bin", b"").ref == "trace://out.bin"


def test_the_device_is_the_TRACE_device_not_the_hosts_availability() -> None:
    """pgw#1458: a cuda trace on a GPU-less box must still report cuda.

    Reading the host's real availability here would place an author's tensor
    on cpu inside a cuda trace, producing the mixed placement AOTI rejects --
    the same defect as the derive's, arriving through the author's own code.
    """

    import torch

    assert _ctx(device="cuda").device == torch.device("cuda")
    assert _ctx().device == torch.device("cpu")


def test_workflow_checkpoint_always_RUNS_the_work_at_trace() -> None:
    """Answering from a cache would skip the code the derive exists to see."""

    ran = []

    def work() -> str:
        # A named function, not `lambda: ran.append(1) or "value"`: `append`
        # returns None, so the `or` was load-bearing punctuation rather than a
        # choice, and mypy reads it as the mistake it usually is.
        ran.append(1)
        return "value"

    result = _ctx().workflow_checkpoint("k", work)
    assert result == "value" and ran == [1]


def test_a_peer_call_is_answered_empty_and_STATED_never_silently() -> None:
    """Refusing would make every composing endpoint underivable -- the same
    defect in a new place. Answering silently would claim coverage the trace
    does not have. So: empty, and recorded, exactly like an unobserved target.
    """

    ctx = _ctx()
    assert ctx.call_endpoint("upscaler", "generate", {"image": "x"}) == {}
    assert ctx.unanswered_calls == [("upscaler", "generate")]


@pytest.mark.parametrize(
    "member,args",
    [
        ("resolve_dataset", ("dataset://x",)),
        ("materialize_blob", ("sha256:abc", "/tmp/x")),
        ("open_checkpoint_stream", ("ckpt://x",)),
    ],
)
def test_the_bytes_only_members_refuse_BY_NAME(member: str, args: tuple) -> None:
    """The three whose whole content is bytes a trace does not have.

    A no-op is WORSE than a refusal here: a fabricated path or an empty file
    makes the endpoint read something that is not there and fail two frames
    later, naming the author's line instead of the derive's gap.
    """

    with pytest.raises(TraceSurfaceUnavailable, match=member):
        getattr(_ctx(), member)(*args)


def test_every_answered_member_is_callable_with_the_serving_signature() -> None:
    """Answering a member with an incompatible signature is still a hole.

    The guide's line passes `seed=`; a `for_request(self, pipeline)` would
    satisfy the set fence and still raise at the author's line.
    """

    from gen_worker.serving.context import RequestContext as Serving

    mismatched = []
    for name in sorted(n for n in dir(Serving) if not n.startswith("_")):
        serving_member = inspect.getattr_static(Serving, name)
        trace_member = inspect.getattr_static(TraceRequestContext, name, None)
        if trace_member is None or not callable(serving_member):
            continue
        if isinstance(serving_member, property) or not callable(trace_member):
            continue
        try:
            serving_names = set(inspect.signature(serving_member).parameters)
            trace_names = set(inspect.signature(trace_member).parameters)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        # A `**kwargs` catch-all answers anything the serving member takes.
        if any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(trace_member).parameters.values()
        ):
            continue
        absent = serving_names - trace_names
        if absent:
            mismatched.append(f"{name}: does not accept {sorted(absent)}")
    assert not mismatched, (
        "these trace members exist but would raise TypeError on a call the "
        "serving context accepts:\n  " + "\n  ".join(mismatched)
    )
