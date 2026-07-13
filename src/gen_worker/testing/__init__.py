"""Test helpers for authoring gen-worker endpoints (pgw#524 item 3).

Every ``Slot``-declared endpoint needs a ``ctx.slots["<name>"]`` stub to
unit-test its handler without a live hub — before this module, every
endpoint hand-rolled its own ``FakeCtx``. :func:`fake_context` builds a real
:class:`~gen_worker.request_context.RequestContext` (or a producer-kind
subclass) with ``ctx.slots`` pre-resolved from plain ``(ref, defaults)``
pairs::

    from gen_worker.testing import fake_context
    from gen_worker import HF
    from gen_worker.families import SdxlDefaults

    ctx = fake_context(slots={
        "pipeline": (HF("stabilityai/stable-diffusion-xl-base-1.0"), SdxlDefaults(steps=28)),
    })
    out = Generate().generate(ctx, TextToImage(prompt="a cat"))

Not imported by ``gen_worker`` itself — production code has no reason to
import test helpers; import ``gen_worker.testing`` explicitly from test
modules.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple, Type, TypeVar

from ..api.binding import ModelRef
from ..api.slot import ResolvedSlot
from ..families.base import FamilyDefaults
from ..request_context import RequestContext

C = TypeVar("C", bound=RequestContext)


def stub_slots(
    slots: Mapping[str, Tuple[ModelRef, FamilyDefaults]],
) -> Dict[str, "ResolvedSlot[Any]"]:
    """``{slot_name: (ref, defaults)}`` -> ``{slot_name: ResolvedSlot}`` —
    the same shape ``ctx.slots`` hands a handler in production, built
    directly instead of via the repo-metadata resolution chain."""
    return {name: ResolvedSlot(ref=ref, defaults=defaults) for name, (ref, defaults) in slots.items()}


def fake_context(
    *,
    request_id: str = "test-request",
    slots: Mapping[str, Tuple[ModelRef, FamilyDefaults]] = {},
    cls: Type[C] = RequestContext,  # type: ignore[assignment]
    **kwargs: Any,
) -> C:
    """Build a :class:`RequestContext` (or ``cls=``, a producer-kind
    subclass: ``ConversionContext``/``DatasetContext``/``TrainingContext``)
    for a handler unit test, with ``ctx.slots`` pre-populated.

    ``slots`` maps slot name -> ``(ref, defaults)`` — exactly what a
    ``Slot``-declared endpoint's handler reads via
    ``ctx.slots["<name>"].ref`` / ``.defaults``. Every other
    :class:`RequestContext` constructor kwarg (``owner``, ``timeout_ms``,
    ``compute``, ...) passes through via ``**kwargs``.
    """
    return cls(
        request_id=request_id,
        resolved_slots=stub_slots(slots),
        **kwargs,
    )


__all__ = ["fake_context", "stub_slots"]
