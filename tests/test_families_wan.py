"""WanDefaults family (animegen lane, pgw#520/th#767 second family after
SDXL): registration + schema export + the real th#767 resolution chain
(repo metadata over Slot(default_config=...))."""

from __future__ import annotations

import msgspec

from gen_worker.api.binding import Hub
from gen_worker.api.slot import Slot, resolve_slot
from gen_worker.families import WanDefaults, export_json_schema, family_for


def test_wan_defaults_registered_under_wan22_root() -> None:
    # Registered under tensorhub's canonical architecture root "wan22"
    # (internal/modelfamily/modelfamily.go), NOT a gen-worker Compile(family=)
    # string — repo-metadata validation keys on the repo's own classified
    # model_family column, shared by every wan22-envelope checkpoint.
    assert family_for("wan22") is WanDefaults
    d = WanDefaults()
    assert (d.steps, d.guidance, d.guidance_2, d.max_guidance, d.shift) == (40, 4.0, 3.0, None, None)
    assert d.family == "wan22"


def test_wan_defaults_schema_exports_closed_object() -> None:
    schema = export_json_schema("wan22")
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) >= {"steps", "guidance", "guidance_2", "max_guidance", "shift"}


def test_resolve_slot_prefers_repo_metadata_over_default_config() -> None:
    base = WanDefaults(steps=40, guidance=4.0, guidance_2=3.0)
    slot: Slot[WanDefaults] = Slot(
        object, selected_by="model", default_checkpoint=Hub("tensorhub/wan22-t2v-a14b"),
        default_config=base,
    )
    animegen = WanDefaults(steps=8, guidance=1.0, guidance_2=1.0, max_guidance=1.5)
    raw = msgspec.json.encode(animegen).decode()

    # No repo metadata (base checkpoint): falls back to the endpoint's code preset.
    resolved_base = resolve_slot(
        "pipeline", slot, ref=Hub("tensorhub/wan22-t2v-a14b"), family="wan22",
    )
    assert resolved_base.defaults == base

    # Repo metadata present (animegen checkpoint): wins outright over default_config.
    resolved_animegen = resolve_slot(
        "pipeline", slot, ref=Hub("tensorhub/wan22-animegen-t2v"), family="wan22",
        raw_metadata_json=raw,
    )
    assert resolved_animegen.defaults == animegen
