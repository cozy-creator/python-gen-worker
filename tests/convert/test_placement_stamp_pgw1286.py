"""pgw#1286 (HARDCUT M3): the ONE precision fact both repos still restate.

`models/ladder.py` used to vendor three tables of tensorhub policy — the fp8
compute floor, the family-root overrides and the conv-UNet w8a8 exclusion set.
All three lost their last reader when the local AUTO fold died (pgw#1148 /
pgw#1180), all three had already drifted from their hub twins, and one named a
twin that no longer exists (`convUNetW8A8ExcludedRoots` is `convUNetUpcastRoots`
today). They are deleted, not fenced: the walk, the family root and the lane
policy are hub decisions that reach the worker resolved.

What SURVIVES the deletion is the placement STAMP — the worker is the producer
of `checkpoints.metadata["placement"]`, and tensorhub keeps a mirror of these
same defaults as its fallback for unstamped rows
(`internal/orchestrator/precision/placement.go`: `defaultPlacement`,
`svdqFP4SMs`, `svdqINT4SMs`, `PlacementFromMetadata`). Those two tables must
agree byte for byte or an unstamped row is admitted onto silicon the producer
would have excluded. THIS is the drift fence, and it is written at the wire:
every row publishes through the real `publish_flavors` against the fake hub and
reads the placement block off the declare request the hub actually receives.

Revert-turns-red: change any SM window, floor, engine list or class token in
`gen_worker.models.ladder.default_placement` (or drop the stamp from
`convert/publish.py`) and the matching row fails on the exact declared block.

    pytest tests/convert/test_placement_stamp_pgw1286.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fake_hub import _FakeHub

from gen_worker.convert.produced import ProducedFlavor
from gen_worker.convert.publish import publish_flavors
from gen_worker.models import ladder

#: Cut by `tests/convert/conftest.py`'s fake repo.
RELEASE = "2026.08"

#: The hub's `precision.defaultPlacement` twin, transcribed. Written as
#: literals on purpose: a fence that derived them from `default_placement`
#: would only ever agree with itself.
HUB_DEFAULTS: dict[str, dict[str, Any] | None] = {
    # ClassBase — zero constraints, so the producer sends NO block at all and
    # the hub's `Placement{PrecisionClass: ClassBase}` fallback is exact.
    "bf16": None,
    "fp16": None,
    # ClassFP8 — fp8 storage serves on any silicon (bf16-upcast path).
    "fp8": {"precision_class": "fp8"},
    "fp8-w8a8": {"precision_class": "fp8"},
    # ClassSvdqINT4 / ClassSvdqFP4 — nunchaku kernel wheels are per-arch, so
    # the allow-list is fail-closed. `svdqINT4SMs` / `svdqFP4SMs` hub-side.
    "svdq-int4": {
        "precision_class": "svdq-int4",
        "sm_allowed": [75, 80, 86, 89],
        "engines": ["nunchaku"],
    },
    "svdq-fp4": {
        "precision_class": "svdq-fp4",
        "sm_allowed": [120, 121],
        "engines": ["nunchaku"],
    },
    # ClassNVFP4W4A4 — Blackwell only, open-ended floor.
    "nvfp4-w4a4": {"precision_class": "nvfp4-w4a4", "sm_min": 100},
    # Unrecognized tokens stay opaque and are never ladder rungs.
    "q4_k_m": None,
}


class _Ctx:
    def __init__(self, base_url: str) -> None:
        self._file_api_base_url = base_url
        self._worker_capability_token = "cap-token"
        self.owner = "acme"

    def log(self, message: str, **fields: Any) -> None:
        pass


def _tree(tmp_path: Path, name: str) -> Path:
    out = tmp_path / name
    out.mkdir()
    (out / "diffusion.safetensors").write_bytes(b"\x11" * 2048)
    return out


def _publish(fake_hub: Any, tmp_path: Path, flavor: str, **attrs: str) -> dict:
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    publish_flavors(
        ctx,
        [ProducedFlavor(
            path=str(_tree(tmp_path, "out")), flavor=flavor, attributes=dict(attrs),
        )],
        destination_repo="acme/qwen-image",
        release=RELEASE,
    )
    return dict(_FakeHub.state["publish_request"].get("metadata") or {})


@pytest.mark.parametrize("token", sorted(HUB_DEFAULTS))
def test_declared_placement_matches_the_hub_fallback_table(
    fake_hub: Any, tmp_path: Path, token: str
) -> None:
    """Each precision class stamps EXACTLY what tensorhub would have assumed
    for an unstamped row of that class — read off the declare request."""
    meta = _publish(fake_hub, tmp_path, token)

    expected = HUB_DEFAULTS[token]
    if expected is None:
        assert "placement" not in meta, (
            f"{token!r} constrains nothing; a block here means the hub's "
            "unstamped fallback and the producer disagree")
    else:
        assert meta["placement"] == expected


def test_the_stamp_is_the_only_hub_policy_this_module_restates(
    fake_hub: Any, tmp_path: Path
) -> None:
    """pgw#1286's deletion, asserted rather than remembered: nothing in the
    ladder module names a family root, a family lane preference or an fp8
    compute floor. Those live hub-side and arrive resolved."""
    vendored = [
        name for name in vars(ladder)
        if ("FAMILY" in name.upper() or "ROOT" in name.upper()
            or "MIN_SM" in name.upper())
    ]
    assert vendored == [], f"hub policy re-vendored into ladder.py: {vendored}"
    assert set(ladder.__all__) == {
        "CLASS_BASE", "CLASS_FP8", "CLASS_NVFP4", "CLASS_NVFP4_W4A4",
        "CLASS_SVDQ_FP4", "CLASS_SVDQ_INT4", "Placement",
        "classify_flavor_token", "default_placement", "placement_to_metadata",
    }


def test_a_producer_override_still_wins_over_the_class_default(
    fake_hub: Any, tmp_path: Path
) -> None:
    """The defaults are a FALLBACK, not a ceiling — a producer that knows its
    kernel coverage states it, and the override travels whole. (The override
    attrs themselves must not leak into metadata as prose.)"""
    meta = _publish(
        fake_hub, tmp_path, "svdq-int4",
        placement_sm_allowed="89", placement_engines="nunchaku,triton",
    )

    assert meta["placement"] == {
        "precision_class": "svdq-int4",
        "sm_allowed": [89],
        "engines": ["nunchaku", "triton"],
    }
    assert not [k for k in meta if k.startswith("placement_")]
