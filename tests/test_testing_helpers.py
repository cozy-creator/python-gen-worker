"""``gen_worker.testing`` (pgw#524 item 3): the ``ctx.slots`` stub helper
every Slot-declared endpoint's unit tests need, instead of a hand-rolled
FakeCtx."""

from __future__ import annotations

from gen_worker import HF
from gen_worker.families import SdxlDefaults
from gen_worker.request_context import ConversionContext, RequestContext
from gen_worker.testing import fake_context, stub_slots

_REF = HF("stabilityai/stable-diffusion-xl-base-1.0")


def test_fake_context_returns_a_request_context() -> None:
    ctx = fake_context()
    assert isinstance(ctx, RequestContext)
    assert ctx.request_id == "test-request"


def test_fake_context_stubs_resolved_slots() -> None:
    ctx = fake_context(slots={"pipeline": (_REF, SdxlDefaults(steps=28))})
    resolved = ctx.slots["pipeline"]
    assert resolved.ref is _REF
    assert resolved.defaults.steps == 28


def test_fake_context_passes_through_kwargs() -> None:
    ctx = fake_context(request_id="r42", owner="acme")
    assert ctx.request_id == "r42"


def test_fake_context_accepts_a_producer_context_subclass() -> None:
    ctx = fake_context(cls=ConversionContext, slots={"pipeline": (_REF, SdxlDefaults())})
    assert isinstance(ctx, ConversionContext)
    assert ctx.slots["pipeline"].ref is _REF


def test_stub_slots_builds_the_ctx_slots_shape() -> None:
    resolved = stub_slots({"vae": (_REF, SdxlDefaults(steps=20))})
    assert resolved["vae"].ref is _REF
    assert resolved["vae"].defaults.steps == 20
