"""The eager `from_pretrained` bridge and `torch_dtype`.

# pgw#1447: a kwargs-STRICT pipeline refuses the keyword.
# pgw#1448: the lane handed it a STRING, so kwargs-ACCEPTING pipelines ran fp32.


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
from typing import Any, Dict

import pytest

from gen_worker.serving.context import (
    DeployBinding, LoadContext, _rejected_torch_dtype,
)


def _ctx(lane: Any) -> LoadContext[Any]:
    """A LoadContext with NO loader engine, which is what forces the bridge."""
    return LoadContext(
        binding=DeployBinding(
            checkpoint_ref="tensorhub/fixture@prod",
            checkpoint_dir=Path("/nonexistent/fixture"),
        ),
        lane=lane,
        engine=None,
    )


class _FakeDtype:
    """A stand-in for a real `torch.dtype` — deliberately NOT a str."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"torch.{self.name}"


class _Lane:
    """A tensorfs-contract stand-in.

    It carries BOTH spellings because the real `Contract` does, and pgw#1448
    is precisely about which one the bridge reads: `.dtype` is the string
    `'bfloat16'`, `.torch_dtype` is the object.
    """

    def __init__(self, dtype):
        self.dtype = dtype
        self.torch_dtype = _FakeDtype(dtype)


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
    calls: Dict[str, Any] = {}

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

    assert repr(calls["torch_dtype"]) == "torch.bfloat16", (
        "the accepting family must keep receiving torch_dtype at load time; "
        "applying it post-load instead would change when the cast happens"
    )
    assert "to" not in calls, "no post-load cast when the loader took the dtype"


# --------------------------------------------------------------------------
# ARM 2 — the ModularPipeline family: H3's refusal, verbatim


def test_a_modular_pipeline_that_refuses_torch_dtype_loads_and_is_cast_after():
    """se#780's exact production failure, reproduced as a unit."""
    calls: Dict[str, Any] = {"attempts": []}

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
    assert repr(calls["to"]) == "torch.bfloat16", (
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
    calls: Dict[str, Any] = {}

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


def test_the_derive_bridge_NEVER_OFFERS_torch_dtype_so_it_cannot_be_refused():
    """pgw#1512 SUPERSEDES the retry this used to assert, and strengthens it.

    This test pinned the pgw#1447 rescue: the derive bridge offered
    `torch_dtype=` first, a strict `ModularPipeline` refused it by TypeError,
    and the bridge retried without and cast afterwards. Paul's per-component
    passthrough ruling DELETED the offer — the lane's dtype governs only the
    component its contract describes, resolved per component inside the hollow
    session — so there is no longer a global cast to be refused.

    The hazard is therefore DESIGNED OUT rather than handled, which is why the
    assertion inverts instead of relaxing: exactly one attempt, carrying no
    `torch_dtype` at all. A loader that refuses it is never given the chance.
    """

    from gen_worker.release.trace_context import TraceLoadContext

    calls: Dict[str, Any] = {"attempts": []}

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
    # ONE attempt, and it carried no dtype — so the refusal above never fires.
    assert calls["attempts"] == [[]]
    # And no blanket `.to(dtype)` afterwards either: casting the whole object
    # is the same fabrication by another route (pgw#1512).
    assert "to" not in calls


# ==========================================================================
# pgw#1448 — THE SECOND DEFECT IN THE SAME FUNCTION: the lane's dtype was a
# STRING, and diffusers answers a string by WARNING and loading fp32.
#
# Found by the local lane's first real sd1.5 run on a 4070: 13 s/it, ~20-40x
# off, because fp32 doubles the weights on a 7.63 GiB card. The precision bug
# IS the performance bug, and it is invisible — a scroll-past warning, then a
# model that works and is merely slow.
#
# THE ASSERTION HAS TO BE THE LOADED DTYPE. Asserting "we passed something"
# would pass on the broken code too: the string WAS passed, and was ignored.

from gen_worker.serving.context import _lane_torch_dtype, _torch_dtype_from_name


class _ContractWithBoth:
    """A tensorfs Contract as it really is: BOTH spellings, and they differ.

        contracts.MINIMAX_H3_DIT_DIFFUSERS.dtype       -> 'bfloat16'  (str)
        contracts.MINIMAX_H3_DIT_DIFFUSERS.torch_dtype -> torch.bfloat16
    """

    dtype = "bfloat16"
    torch_dtype = _FakeDtype("bfloat16")


class _DiffusersLike:
    """Honours a real dtype object; a STRING falls back to fp32 with a warning
    — which is exactly what diffusers does, and why this was silent."""

    warned = False

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        got = kwargs.get("torch_dtype")
        obj = cls()
        if isinstance(got, str) or got is None:
            cls.warned = True
            obj.dtype = "float32"          # the silent fallback
        else:
            obj.dtype = got
        return obj

    def to(self, dtype):
        self.dtype = dtype
        return self


def test_the_lane_dtype_read_prefers_the_OBJECT_over_the_spelling():
    got = _lane_torch_dtype(_ContractWithBoth())
    assert not isinstance(got, str), (
        "returning Contract.dtype hands diffusers a string, which it refuses "
        "with a warning before silently loading fp32"
    )
    assert repr(got) == "torch.bfloat16"


def test_a_bf16_lane_ACTUALLY_LOADS_BF16_through_the_eager_bridge():
    """The red arm the fp32 defect requires.

    Reverting `_lane_torch_dtype` to `self._lane.dtype` leaves this failing
    with `dtype == 'float32'` — the exact production symptom — while a test
    that only asserted "torch_dtype was passed" would pass on the broken code,
    because the string was passed and ignored.
    """
    _DiffusersLike.warned = False
    ctx = _ctx(_ContractWithBoth())
    loaded = ctx.load(_DiffusersLike)

    assert repr(loaded.dtype) == "torch.bfloat16", (
        f"a bf16 lane loaded as {loaded.dtype!r} — this is the fp32 silent "
        f"fallback that made sd1.5 run at 13 s/it on a 4070"
    )
    assert not _DiffusersLike.warned, "the loader should never see a string dtype"


def test_a_contract_carrying_only_the_spelling_is_still_resolved():
    """Belt and braces: if some contract exposes only `.dtype`, turn the name
    into the object rather than passing the string on."""

    class SpellingOnly:
        dtype = "bfloat16"

    got = _lane_torch_dtype(SpellingOnly())
    if got is not None:            # torch present in this env
        assert not isinstance(got, str)


def test_a_name_that_is_not_a_dtype_never_becomes_one():
    """`getattr(torch, 'load')` is a real attribute and not a dtype; handing a
    FUNCTION to torch_dtype= would be worse than handing it a string."""
    assert _torch_dtype_from_name("load") is None
    assert _torch_dtype_from_name("definitely_not_a_dtype") is None
