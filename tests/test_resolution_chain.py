"""pgw#520 resolution chain: ``resolve_slot``/``resolve_slots`` merge repo-
metadata inference defaults over an endpoint's code ``Slot(default_config=...)``
preset, and ``ctx.slots[name]`` surfaces the result (or a lazy error) to the
handler."""

from __future__ import annotations

import msgspec
import pytest

from gen_worker import HF
from gen_worker.api.slot import ResolvedSlot, Slot, resolve_slot, resolve_slots
from gen_worker.families import SdxlDefaults
from gen_worker.request_context import RequestContext

_REF = HF("stabilityai/stable-diffusion-xl-base-1.0")


def test_no_metadata_uses_fallback_preset() -> None:
    slot = Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28, guidance=6.0))
    resolved = resolve_slot("pipeline", slot, ref=_REF)
    assert resolved.ref is _REF
    assert resolved.defaults.steps == 28
    assert resolved.defaults.guidance == 6.0


def test_repo_metadata_wins_over_fallback_wholesale() -> None:
    """Precedence is whole-object, not field-by-field: the repo metadata
    instance replaces the fallback entirely (tensorhub validates the WHOLE
    object at PUT time, so a partial merge would hide invalid metadata
    behind the code default)."""
    slot = Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28, guidance=6.0))
    raw = msgspec.json.encode(SdxlDefaults(steps=40, scheduler="dpmpp_2m_karras")).decode()
    resolved = resolve_slot("pipeline", slot, ref=_REF, raw_metadata_json=raw)
    assert resolved.defaults.steps == 40
    assert resolved.defaults.scheduler == "dpmpp_2m_karras"
    # Fields NOT in the fallback's non-defaults still come through as the
    # family's OWN defaults (whole-object decode), not the code fallback's.
    assert resolved.defaults.guidance == 6.0  # SdxlDefaults' own field default


def test_repo_metadata_with_no_fallback_resolves_via_explicit_family() -> None:
    """A hub-only slot (no code fallback) can still decode repo metadata
    when the endpoint's Compile(family=...) supplies the family name."""
    slot = Slot(object, default_checkpoint=_REF)  # no default_config
    raw = msgspec.json.encode(SdxlDefaults(steps=22)).decode()
    resolved = resolve_slot("pipeline", slot, ref=_REF, family="sdxl", raw_metadata_json=raw)
    assert isinstance(resolved.defaults, SdxlDefaults)
    assert resolved.defaults.steps == 22


def test_repo_metadata_with_no_resolvable_family_raises() -> None:
    slot = Slot(object, default_checkpoint=_REF)  # no default_config, no family
    raw = msgspec.json.encode(SdxlDefaults(steps=22)).decode()
    with pytest.raises(ValueError, match="no family is resolvable"):
        resolve_slot("pipeline", slot, ref=_REF, raw_metadata_json=raw)


def test_invalid_repo_metadata_raises_clear_validation_error() -> None:
    slot = Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28))
    with pytest.raises(ValueError, match="validation"):
        resolve_slot("pipeline", slot, ref=_REF, raw_metadata_json='{"steps": "not-an-int"}')


def test_no_metadata_and_no_fallback_raises_clear_error() -> None:
    slot = Slot(object, default_checkpoint=_REF)  # no default_config, no metadata
    with pytest.raises(ValueError, match="nothing to resolve"):
        resolve_slot("pipeline", slot, ref=_REF)


def test_no_ref_raises_clear_error() -> None:
    slot = Slot(object, default_config=SdxlDefaults(steps=28))
    with pytest.raises(ValueError, match="no resolved model ref"):
        resolve_slot("pipeline", slot, ref=None)


def test_resolve_slots_collects_per_slot_failures_without_raising() -> None:
    slots = {
        "pipeline": Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28)),
        "vae": Slot(object),  # no default_config, no metadata -> will fail
    }
    out = resolve_slots(slots, refs={"pipeline": _REF, "vae": _REF})
    assert isinstance(out["pipeline"], ResolvedSlot)
    assert isinstance(out["vae"], ValueError)


# --------------------------------------------------------------------------- #
# ctx.slots — lazy per-key errors, not a blanket dispatch-time failure         #
# --------------------------------------------------------------------------- #


def test_ctx_slots_returns_resolved_slot() -> None:
    resolved = resolve_slot(
        "pipeline", Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28)), ref=_REF,
    )
    ctx = RequestContext(request_id="r1", resolved_slots={"pipeline": resolved})
    got = ctx.slots["pipeline"]
    assert got.ref is _REF
    assert got.defaults.steps == 28


def test_ctx_slots_error_raises_only_on_access() -> None:
    ctx = RequestContext(
        request_id="r1",
        resolved_slots={},
        slot_errors={"vae": "no repo metadata and no fallback"},
    )
    # Declared but unresolved: iterable, but raises only when READ.
    assert "vae" in list(ctx.slots)
    with pytest.raises(ValueError, match="no repo metadata"):
        ctx.slots["vae"]


def test_ctx_slots_missing_key_is_a_keyerror() -> None:
    ctx = RequestContext(request_id="r1")
    with pytest.raises(KeyError):
        ctx.slots["never-declared"]


def test_ctx_set_resolved_slots_mutator_used_by_cli() -> None:
    """The CLI's hub-less path builds ctx before resolving models (unlike
    the executor, which has everything up front); it mutates ctx.slots
    after the fact via this private hook."""
    ctx = RequestContext(request_id="r1")
    assert list(ctx.slots) == []
    resolved = resolve_slot(
        "pipeline", Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28)), ref=_REF,
    )
    ctx._set_resolved_slots({"pipeline": resolved})
    assert ctx.slots["pipeline"].defaults.steps == 28
