"""pgw#692 (th#1174, migration 0046): every registered family vocabulary's
field names ARE the hub's recipe wire names.

The hub stamps one resolved recipe per slot from
``internal/modelfamily/inferencedefaults/families/*.schema.json`` and the
worker decodes it with ``forbid_unknown_fields``, so a single renamed field
makes every request of that family FATAL at slot resolution — which is
exactly what migration 0046's ``steps`` -> ``num_inference_steps`` rename did
to wan-2.2 on chaos before this fix.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import gen_worker.families
from gen_worker import Hub
from gen_worker.api.slot import Slot, resolve_slot
from gen_worker.families.base import family_for

# Snapshot of each hub family schema's ``properties`` keys, read from
# tensorhub after migration 0046. A struct that drifts from its schema is a
# production outage, not a test failure — update BOTH sides together.
HUB_SCHEMA_FIELDS = {
    ("sdxl", "checkpoint"): {
        "schema_version", "scheduler", "steps", "guidance",
        "quality_preamble", "negative", "max_guidance",
    },
    ("sdxl", "lora"): {
        "schema_version", "trigger_words", "recommended_weight", "steps",
        "guidance", "max_guidance", "scheduler",
    },
    ("wan22", "checkpoint"): {
        "schema_version", "num_inference_steps", "guidance", "guidance_2",
        "max_guidance", "shift",
    },
}

# The exact shape the chaos hub stamps for a wan-2.2 slot post-0046.
WAN22_HUB_RECIPE = json.dumps({
    "schema_version": 1,
    "num_inference_steps": 40,
    "guidance": 4.0,
    "guidance_2": 3.0,
    "max_guidance": None,
    "shift": None,
})

# The pre-0046 spelling — no hub anywhere stamps this any more, and the
# ``PUT .../metadata/inference-defaults`` route refuses it outright.
WAN22_PRE_0046_RECIPE = json.dumps({
    "schema_version": 1,
    "steps": 40,
    "guidance": 4.0,
    "guidance_2": 3.0,
})


def _resolve(raw: str) -> Any:
    return resolve_slot(
        "pipeline",
        Slot(str),
        ref=Hub("cozy/wan22-t2v-a14b", tag="prod"),
        family="wan22",
        raw_metadata_json=raw,
    )


def test_wan22_hub_recipe_decodes() -> None:
    resolved = _resolve(WAN22_HUB_RECIPE)
    assert resolved.defaults.num_inference_steps == 40
    assert resolved.defaults.guidance_2 == 3.0


def test_wan22_pre_0046_steps_spelling_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown field `steps`"):
        _resolve(WAN22_PRE_0046_RECIPE)


def _shipped_vocabularies() -> set[tuple[str, str]]:
    """Every vocabulary this package SHIPS. Not ``family_registry()`` — that
    is process-global and test modules register their own throwaway families
    into it."""
    out: set[tuple[str, str]] = set()
    for obj in vars(gen_worker.families).values():
        if not isinstance(obj, type):
            continue
        fam = str(getattr(obj, "__gen_worker_family__", "") or "")
        if fam:
            out.add((fam, str(getattr(obj, "__gen_worker_kind__", "") or "")))
    return out


@pytest.mark.parametrize("key", sorted(HUB_SCHEMA_FIELDS))
def test_shipped_vocabularies_match_the_hub_schemas(key: tuple[str, str]) -> None:
    name, kind = key
    cls = family_for(name, kind=kind)
    assert cls is not None, f"family {name!r}/{kind} is no longer registered"
    assert set(cls.__struct_fields__) == HUB_SCHEMA_FIELDS[key], (
        f"{cls.__name__} has drifted from the hub's {name} recipe wire names; "
        "the hub stamps recipes the worker cannot decode"
    )


def test_every_shipped_vocabulary_has_a_recorded_hub_schema() -> None:
    assert _shipped_vocabularies() == set(HUB_SCHEMA_FIELDS), (
        "a shipped family vocabulary has no recorded hub schema — add it to "
        "HUB_SCHEMA_FIELDS from tensorhub's families/*.schema.json"
    )
