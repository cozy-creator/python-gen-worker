"""pgw#692 (th#1174, migration 0046): every registered family vocabulary's
field names ARE the hub's recipe wire names.

The hub stamps one resolved recipe per slot from
``internal/modelfamily/inferencedefaults/families/*.schema.json`` and the
worker decodes it with ``forbid_unknown_fields``, so a single renamed field
makes every request of that family FATAL at slot resolution — which is
exactly what migration 0046's ``steps`` -> ``num_inference_steps`` rename did
to wan-2.2 on chaos before this fix.

this package no longer SHIPS any family vocabulary, so the
per-family hub snapshots moved to the endpoints that own them
(inference-endpoints ie#567 — ``sdxl/tests/test_family_vocabulary_ie567.py``
and ``wan-2.2/tests/test_family_vocabulary_ie567.py``, which pin the exact
field sets and the pre-0046 rejection). What stays here is the MECHANISM those
tests depend on: a hub recipe decodes into a declared vocabulary field for
field, an unknown field is refused by name, and the SDK ships no vocabulary
that could silently drift against the hub.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import gen_worker.families
from gen_worker import Hub
from gen_worker.api.slot import Slot, resolve_slot
from gen_worker.families.base import family_for

from _example_family import ExampleDefaults  # noqa: F401  (registers "example")

# The shape a hub stamps for a slot of the declared family: every field of the
# vocabulary, spelled exactly as the struct spells it.
HUB_RECIPE = json.dumps({
    "schema_version": 1,
    "scheduler": "lcm",
    "steps": 8,
    "guidance": 1.0,
    "negative": "",
    "max_guidance": 1.5,
})

# A recipe carrying a field the vocabulary does not declare — the shape a
# post-rename hub stamps at a worker that has not caught up.
HUB_RECIPE_STALE_FIELD = json.dumps({
    "schema_version": 1,
    "steps": 8,
    "num_inference_steps": 8,
})


def _resolve(raw: str) -> Any:
    return resolve_slot(
        "pipeline",
        Slot(str),
        ref=Hub("cozy/example-model", tag="prod"),
        family="example",
        raw_metadata_json=raw,
    )


def test_a_hub_recipe_decodes_into_the_declared_vocabulary() -> None:
    resolved = _resolve(HUB_RECIPE)
    assert resolved.defaults.steps == 8
    assert resolved.defaults.guidance == 1.0
    assert resolved.defaults.max_guidance == 1.5
    assert resolved.defaults.scheduler == "lcm"


def test_a_field_the_vocabulary_does_not_declare_is_refused_by_name() -> None:
    """The 0046 failure mode, generically: a renamed field must be FATAL and
    named, never silently dropped into a default."""
    with pytest.raises(ValueError, match="unknown field `num_inference_steps`"):
        _resolve(HUB_RECIPE_STALE_FIELD)


def test_the_declared_field_set_is_the_wire_contract() -> None:
    cls = family_for("example")
    assert cls is not None
    assert set(cls.__struct_fields__) == {
        "schema_version", "scheduler", "steps", "guidance", "negative", "max_guidance",
    }


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


def test_the_sdk_ships_no_family_vocabulary() -> None:
    """pgw#740 B1, as a standing guard: a vocabulary shipped here is one that
    needs a WHEEL RELEASE to change and can drift against the hub between
    releases. It belongs in the endpoint that owns the family."""
    assert _shipped_vocabularies() == set(), (
        "a family vocabulary has been added back to gen_worker.families; it "
        "belongs in the endpoint that owns the family (see ie#567)"
    )
