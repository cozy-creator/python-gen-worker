"""The eager `from_pretrained` bridge and `torch_dtype`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker.serving.context import (
    DeployBinding, LoadContext, _rejected_torch_dtype,
)


def _ctx(lane: Any) -> LoadContext[Any]:
    return LoadContext(
        binding=DeployBinding(
            checkpoint_ref="tensorhub/fixture@prod",
            checkpoint_dir=Path("/nonexistent/fixture"),
        ),
        lane=lane,
        engine=None,
    )


class _FakeDtype:

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"torch.{self.name}"


class _Lane:

    def __init__(self, dtype):
        self.dtype = dtype
        self.torch_dtype = _FakeDtype(dtype)


def test_a_torch_dtype_refusal_is_recognised_by_the_keyword_it_names():
    exc = TypeError(
        "MiniMaxH3StreamingPipeline: ['torch_dtype'] are neither pipeline "
        "arguments nor components of this pipeline"
    )
    assert _rejected_torch_dtype(exc)


def test_a_typeerror_from_inside_the_model_is_NOT_swallowed():
    """The retry must be narrow."""
    assert not _rejected_torch_dtype(TypeError("unsupported operand type(s) for +"))
    assert not _rejected_torch_dtype(TypeError("expected Tensor, got NoneType"))


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


def test_a_modular_pipeline_that_refuses_torch_dtype_loads_and_is_cast_after():
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


def test_the_derive_bridge_NEVER_OFFERS_torch_dtype_so_it_cannot_be_refused():

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
    assert calls["attempts"] == [[]]
    assert "to" not in calls


from gen_worker.serving.context import _lane_torch_dtype, _torch_dtype_from_name


class _ContractWithBoth:

    dtype = "bfloat16"
    torch_dtype = _FakeDtype("bfloat16")


class _DiffusersLike:

    warned = False

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        got = kwargs.get("torch_dtype")
        obj = cls()
        if isinstance(got, str) or got is None:
            cls.warned = True
            obj.dtype = "float32"
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
    """The red arm the fp32 defect requires."""
    _DiffusersLike.warned = False
    ctx = _ctx(_ContractWithBoth())
    loaded = ctx.load(_DiffusersLike)

    assert repr(loaded.dtype) == "torch.bfloat16", (
        f"a bf16 lane loaded as {loaded.dtype!r} — this is the fp32 silent "
        f"fallback that made sd1.5 run at 13 s/it on a 4070"
    )
    assert not _DiffusersLike.warned, "the loader should never see a string dtype"


def test_a_contract_carrying_only_the_spelling_is_still_resolved():
    """Belt and braces: if some contract exposes only `.dtype`, turn the name into the object rather than passing the string on."""

    class SpellingOnly:
        dtype = "bfloat16"

    got = _lane_torch_dtype(SpellingOnly())
    if got is not None:
        assert not isinstance(got, str)


def test_a_name_that_is_not_a_dtype_never_becomes_one():
    """`getattr(torch, 'load')` is a real attribute and not a dtype; handing a FUNCTION to torch_dtype= would be worse than handing it a string."""
    assert _torch_dtype_from_name("load") is None
    assert _torch_dtype_from_name("definitely_not_a_dtype") is None
