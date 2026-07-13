"""pgw#520 resolution chain: ``resolve_slot``/``resolve_slots`` merge repo-
metadata inference defaults over an endpoint's code ``Slot(default_config=...)``
preset, and ``ctx.slots[name]`` surfaces the result (or a lazy error) to the
handler."""

from __future__ import annotations

import msgspec
import pytest

from gen_worker import HF
from gen_worker.api.slot import ResolvedSlot, Slot, resolve_slot, resolve_slots
from gen_worker.families import SdxlDefaults, SdxlLoraDefaults
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


# --------------------------------------------------------------------------- #
# pgw#516 composition rule: lora inference_defaults override the resolved     #
# checkpoint recipe FIELD BY FIELD (not whole-object), in lora order.         #
# --------------------------------------------------------------------------- #


def test_lora_overrides_apply_field_level_on_fallback_recipe() -> None:
    """The worked example from CONTRACT.md: a distillation lora's
    steps=4/guidance=0 beats the base checkpoint's 28/6; fields the lora
    left null (scheduler/max_guidance) stay untouched."""
    slot = Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28, guidance=6.0))
    lora_raw = msgspec.json.encode(SdxlLoraDefaults(steps=4, guidance=0.0)).decode()
    resolved = resolve_slot(
        "pipeline", slot, ref=_REF, family="sdxl", lora_metadata_json=[lora_raw],
    )
    assert resolved.defaults.steps == 4
    assert resolved.defaults.guidance == 0.0
    assert resolved.defaults.scheduler == "euler_a"  # SdxlDefaults' own default, untouched


def test_lora_overrides_apply_on_top_of_repo_metadata_whole_object_result() -> None:
    """Repo metadata (whole-object) resolves first; the lora's field-level
    override applies on top of THAT result, not the code fallback."""
    slot = Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28, guidance=6.0))
    repo_raw = msgspec.json.encode(SdxlDefaults(steps=40, guidance=5.0)).decode()
    lora_raw = msgspec.json.encode(SdxlLoraDefaults(guidance=0.0)).decode()
    resolved = resolve_slot(
        "pipeline", slot, ref=_REF, raw_metadata_json=repo_raw, lora_metadata_json=[lora_raw],
    )
    assert resolved.defaults.steps == 40  # from repo metadata, lora had no opinion
    assert resolved.defaults.guidance == 0.0  # lora override wins


def test_multiple_loras_apply_in_order_later_wins_on_shared_field() -> None:
    slot = Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28, guidance=6.0))
    lora_a = msgspec.json.encode(SdxlLoraDefaults(steps=8, guidance=2.0)).decode()
    lora_b = msgspec.json.encode(SdxlLoraDefaults(steps=4)).decode()  # guidance untouched
    resolved = resolve_slot(
        "pipeline", slot, ref=_REF, family="sdxl", lora_metadata_json=[lora_a, lora_b],
    )
    assert resolved.defaults.steps == 4  # lora_b (later) wins over lora_a
    assert resolved.defaults.guidance == 2.0  # lora_b left it null; lora_a's value stands


def test_lora_only_fields_never_ride_ctx_slots_defaults() -> None:
    """trigger_words/recommended_weight have no checkpoint-recipe analog —
    they are NOT merged into ctx.slots[slot].defaults (out of this issue's
    settled endpoint-authoring scope)."""
    slot = Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28))
    lora_raw = msgspec.json.encode(
        SdxlLoraDefaults(trigger_words=("mystyle",), recommended_weight=0.7)
    ).decode()
    resolved = resolve_slot(
        "pipeline", slot, ref=_REF, family="sdxl", lora_metadata_json=[lora_raw],
    )
    assert not hasattr(resolved.defaults, "trigger_words")
    assert not hasattr(resolved.defaults, "recommended_weight")
    assert resolved.defaults.steps == 28  # unaffected


def test_lora_with_no_opinions_leaves_recipe_untouched() -> None:
    slot = Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28, guidance=6.0))
    lora_raw = msgspec.json.encode(SdxlLoraDefaults(trigger_words=("x",))).decode()
    resolved = resolve_slot(
        "pipeline", slot, ref=_REF, family="sdxl", lora_metadata_json=[lora_raw],
    )
    assert resolved.defaults.steps == 28
    assert resolved.defaults.guidance == 6.0


def test_empty_lora_metadata_entries_are_skipped() -> None:
    slot = Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28))
    resolved = resolve_slot(
        "pipeline", slot, ref=_REF, lora_metadata_json=["", "  "],
    )
    assert resolved.defaults.steps == 28


def test_lora_metadata_for_unregistered_lora_kind_family_is_skipped() -> None:
    """No kind="lora" vocabulary registered for this family -> best-effort
    skip, never blocks the checkpoint's own resolved recipe."""
    slot = Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28))
    resolved = resolve_slot(
        "pipeline", slot, ref=_REF, family="does-not-exist",
        lora_metadata_json=['{"steps": 4}'],
    )
    assert resolved.defaults.steps == 28


def test_malformed_lora_metadata_raises_like_repo_metadata_does() -> None:
    slot = Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28))
    with pytest.raises(ValueError, match="validation"):
        resolve_slot(
            "pipeline", slot, ref=_REF, family="sdxl",
            lora_metadata_json=['{"steps": "not-an-int"}'],
        )


def test_resolve_slots_threads_lora_metadata_per_slot() -> None:
    slots = {
        "pipeline": Slot(object, default_checkpoint=_REF, default_config=SdxlDefaults(steps=28, guidance=6.0)),
    }
    lora_raw = msgspec.json.encode(SdxlLoraDefaults(steps=4)).decode()
    out = resolve_slots(
        slots, refs={"pipeline": _REF}, families={"pipeline": "sdxl"},
        lora_metadata={"pipeline": [lora_raw]},
    )
    resolved = out["pipeline"]
    assert isinstance(resolved, ResolvedSlot)
    assert resolved.defaults.steps == 4
    assert resolved.defaults.guidance == 6.0


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
