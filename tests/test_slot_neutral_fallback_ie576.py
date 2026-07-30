"""ie#576 (0.58.3): the SDK v2 neutral-fallback tail, backported.

Boot warmup builds its RequestContext via ``_resolve_slots_kwargs(spec, None)``
— no RunJob, so no hub-stamped metadata can ever reach it. On <=0.58.2 a
defaults-bearing slot without ``Slot(default_config=...)`` therefore FATALed
every boot warmup (chaos sdxl 0.1.7: 4 requests, 8 pods, 0 completions).
These tests go red if the old ``raise ValueError`` tail is restored.
"""

import pytest

from gen_worker.api.binding import HF
from gen_worker.api.slot import Slot, resolve_slot
from gen_worker.families import SdxlDefaults


def test_bare_slot_with_registered_family_resolves_neutral() -> None:
    # The chaos sdxl shape: Slot(selected_by="model"), no default_config,
    # family "sdxl" from Compile(family=...). Must resolve to the NEUTRAL
    # schema defaults, not raise.
    slot = Slot(object, selected_by="model")
    resolved = resolve_slot(
        "pipeline", slot, ref=HF("acme/wai-illustrious"), family="sdxl",
    )
    assert resolved.defaults == SdxlDefaults()


def test_bare_slot_with_no_family_resolves_ref_only() -> None:
    # The chaos anima/z-image shape: no default_config, no registered family
    # vocabulary. The ref resolves; defaults is None (v2 semantics).
    slot = Slot(object)
    resolved = resolve_slot("pipeline", slot, ref=HF("acme/anima"), family="z-image")
    assert resolved.defaults is None
    assert resolved.ref.path == "acme/anima"


def test_metadata_still_wins_over_neutral() -> None:
    raw = '{"steps": 12, "guidance": 1.5}'
    slot = Slot(object, selected_by="model")
    resolved = resolve_slot(
        "pipeline", slot, ref=HF("acme/wai-illustrious"), family="sdxl",
        raw_metadata_json=raw,
    )
    assert resolved.defaults.steps == 12
    assert resolved.defaults != SdxlDefaults()


def test_missing_ref_is_still_an_error() -> None:
    slot = Slot(object)
    with pytest.raises(ValueError, match="no resolved model ref"):
        resolve_slot("pipeline", slot, ref=None)
