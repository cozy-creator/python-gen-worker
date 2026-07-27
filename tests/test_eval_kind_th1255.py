"""th#1255: kind="eval" is declarable, survives discovery/lock validation, and
maps to a context that materializes reserved refs without a publish surface."""
import msgspec
import pytest

from gen_worker.api.decorators import KINDS, endpoint
from gen_worker.discovery.validation import _KNOWN_KINDS
from gen_worker.executor import _CONTEXT_BY_KIND
from gen_worker.request_context import ConversionContext, RequestContext


class In(msgspec.Struct, forbid_unknown_fields=True):
    source: dict


class Out(msgspec.Struct):
    lpips_mean: float


def test_eval_kind_is_declarable():
    assert "eval" in KINDS and "eval" in _KNOWN_KINDS

    @endpoint(kind="eval")
    class ExternalQualityEval:
        def external_quality_eval(self, ctx, payload: In) -> Out:
            return Out(lpips_mean=0.0)

    assert ExternalQualityEval is not None


def test_unknown_kind_still_rejected():
    with pytest.raises(ValueError, match="kind must be one of"):
        @endpoint(kind="evaluation")
        class Nope:
            def go(self, ctx, payload: In) -> Out:
                return Out(lpips_mean=0.0)


def test_eval_context_materializes_reserved_refs_and_is_not_the_base():
    # The executor sets producer=True for every non-inference kind and passes
    # source_info/destination_info kwargs; a base RequestContext would raise
    # TypeError on them. Registering the kind is mandatory, not cosmetic.
    cls = _CONTEXT_BY_KIND["eval"]
    assert cls is ConversionContext
    assert cls is not RequestContext
    assert hasattr(cls, "source") and hasattr(cls, "source_path")
