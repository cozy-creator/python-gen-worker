"""pgw#1447: the eager `from_pretrained` bridge and `torch_dtype`.

`ctx.load`'s docstring states the author contract as *"No ``torch_dtype=`` (the
lane contract IS the dtype)"* — and twelve lines below it, the eager bridge
passed exactly that to the author's own class. For a `ModularPipeline` the
keyword is not consumed by `from_pretrained`; it funnels through `**kwargs` into
the constructor, where a strict pipeline refuses an argument that is neither a
pipeline argument nor a component.

Found by se#780 running the LOCAL config-only derive on minimax-h3 — the derive
has no streaming engine bound, so it takes this bridge:

    lane 'minimax.h3-dit-diffusers@1': load() failed under the trace session:
    TypeError: MiniMaxH3StreamingPipeline: ['torch_dtype'] are neither pipeline
    arguments nor components of this pipeline

Both arms are asserted here because a fix that only satisfies the broken family
would silently drop the dtype for the family that works.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.serving.context import LoadContext, _rejected_torch_dtype


class _Recorder:
    """Base for the doubles: records what the loader was handed."""

    seen_dtype: object = "<never called>"
    to_calls: list = []


def _ctx(lane) -> LoadContext:
    """A LoadContext with NO loader engine, which is what forces the bridge."""
    return LoadContext(
        binding=_Binding(),
        lane=lane,
        engine=None,
    )


class _Binding:
    checkpoint_ref = "tensorhub/fixture@prod"
    checkpoint_dir = Path("/nonexistent/fixture")


class _Lane:
    """A tensorfs-contract stand-in whose `.dtype` is the lane's load dtype."""

    def __init__(self, dtype):
        self.dtype = dtype


# --------------------------------------------------------------------------
# the refusal predicate


def test_a_torch_dtype_refusal_is_recognised_by_the_keyword_it_names():
    exc = TypeError(
        "MiniMaxH3StreamingPipeline: ['torch_dtype'] are neither pipeline "
        "arguments nor components of this pipeline"
    )
    assert _rejected_torch_dtype(exc)


def test_a_typeerror_from_inside_the_model_is_NOT_swallowed():
    """The retry must be narrow. A TypeError raised by the author's own
    __init__ for an unrelated reason has to keep propagating — otherwise this
    fix converts a real bug into a silent second load."""
    assert not _rejected_torch_dtype(TypeError("unsupported operand type(s) for +"))
    assert not _rejected_torch_dtype(TypeError("expected Tensor, got NoneType"))


# --------------------------------------------------------------------------
# ARM 1 — the loader that ACCEPTS torch_dtype keeps getting it (no regression)


def test_a_loader_that_accepts_torch_dtype_still_receives_it():
    calls = {}

    class Accepts:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["path"] = path
            calls["torch_dtype"] = kwargs.get("torch_dtype", "<absent>")
            return cls()

        def to(self, dtype):  # pragma: no cover - must NOT be reached
            calls["to"] = dtype
            return self

    ctx = _ctx(_Lane("bfloat16"))
    ctx.load(Accepts)

    assert calls["torch_dtype"] == "bfloat16", (
        "the accepting family must keep receiving torch_dtype at load time; "
        "applying it post-load instead would change when the cast happens"
    )
    assert "to" not in calls, "no post-load cast when the loader took the dtype"


# --------------------------------------------------------------------------
# ARM 2 — the ModularPipeline family: H3's refusal, verbatim


def test_a_modular_pipeline_that_refuses_torch_dtype_loads_and_is_cast_after():
    """se#780's exact production failure, reproduced as a unit."""
    calls = {"attempts": []}

    class RefusesLikeModularPipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["attempts"].append(sorted(kwargs))
            if "torch_dtype" in kwargs:
                raise TypeError(
                    "MiniMaxH3StreamingPipeline: ['torch_dtype'] are neither "
                    "pipeline arguments nor components of this pipeline"
                )
            return cls()

        def to(self, dtype):
            calls["to"] = dtype
            return self

    ctx = _ctx(_Lane("bfloat16"))
    got = ctx.load(RefusesLikeModularPipeline)

    assert isinstance(got, RefusesLikeModularPipeline)
    assert calls["attempts"] == [["torch_dtype"], []], (
        "expected one attempt WITH torch_dtype, then a retry without it"
    )
    assert calls["to"] == "bfloat16", (
        "THE LANE MUST STILL BE HONOURED. Dropping the dtype here would load a "
        "bf16 lane in the checkpoint's own dtype and serve different numerics "
        "with nothing raised — the silent-degradation shape this repo refuses."
    )


def test_an_unrelated_typeerror_from_the_loader_propagates():
    class Broken:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            raise TypeError("expected Tensor, got NoneType")

    with pytest.raises(TypeError, match="expected Tensor"):
        _ctx(_Lane("bfloat16")).load(Broken)


# --------------------------------------------------------------------------
# the no-lane path is untouched


def test_a_model_with_no_lane_never_mentions_dtype_at_all():
    calls = {}

    class NoLane:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["kwargs"] = sorted(kwargs)
            return cls()

        def to(self, dtype):  # pragma: no cover - must NOT be reached
            calls["to"] = dtype
            return self

    ctx = _ctx(None)
    ctx.load(NoLane)
    assert calls["kwargs"] == [], "an eager-permanent model has no dtype to pass"
    assert "to" not in calls


# --------------------------------------------------------------------------
# SECOND SITE — the DERIVE's own bridge (release/trace_context.py).
#
# Fixing only the serve-path copy left the derive broken, and the derive is the
# one path that ALWAYS takes an eager bridge (a trace binds no streaming
# engine). Two implementations of one contract; both are asserted here so they
# cannot drift apart again.


def test_the_derive_bridge_also_survives_a_loader_that_refuses_torch_dtype():
    from gen_worker.release.trace_context import TraceLoadContext

    calls = {"attempts": []}

    class RefusesLikeModularPipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["attempts"].append(sorted(kwargs))
            if "torch_dtype" in kwargs:
                raise TypeError(
                    "MiniMaxH3StreamingPipeline: ['torch_dtype'] are neither "
                    "pipeline arguments nor components of this pipeline"
                )
            return cls()

        def to(self, dtype):
            calls["to"] = dtype
            return self

    ctx = TraceLoadContext(
        checkpoint_dir=Path("/nonexistent/fixture"),
        lane=_Lane("bfloat16"),
    )
    got = ctx.load(RefusesLikeModularPipeline)

    assert isinstance(got, RefusesLikeModularPipeline)
    assert calls["attempts"] == [["torch_dtype"], []]
    assert calls["to"] == "bfloat16", "the derive must honour the lane dtype too"
