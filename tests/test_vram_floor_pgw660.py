"""Every endpoint's hard VRAM floor reaches the hub.

Measurement answers how much VRAM a function WANTS; it cannot answer which card
it cannot run on AT ALL, and only the second question has an answer before a
first build exists.

The builder maps requirements out of the manifest's `resources{}` by NAME, so a
floor that is not emitted does not fail: `min_vram_gb: 80` silently disappears,
`req.VRAMGB` falls back to the RELEASE-level envelope (7 GB for conversion),
and the per-function floor leaves the selection path with no error, no lint and
no build failure. The SM twin of this hole placed sm_89+-only work on sm_80
A100s.

`Resources.compute_capability` is the SM half; this is its VRAM twin. The tests
below are written against the SAME hub keys the builder reads, so a hub-side
rename reds here rather than on a rented pod.
"""

from __future__ import annotations

import pytest

from gen_worker.api.decorators import Resources


# ---------------------------------------------------------------------------
# The emitter, under the key the hub already gates on
# ---------------------------------------------------------------------------


def test_the_floor_projects_under_the_HUBS_OWN_key_with_no_remap() -> None:
    """`function_requirements.go` folds `min_vram_gb` straight into
    `requirement_payload_json`, whence `req.VRAMGB` -> `MinVRAMGB` -> the GPU
    candidate filter. The hub has read this key all along; v2 stopped emitting
    anything that landed there. Same word on both sides, deliberately — the
    one remap in this projection (`ram_gb_hint` -> `ram_gb`) is the exception
    that had to be written down."""
    assert Resources(min_vram_gb=80).manifest_dict() == {
        "gpu": True, "min_vram_gb": 80.0}


def test_an_undeclared_floor_emits_NOTHING() -> None:
    """`omit_defaults`, so every release written before this field has a
    byte-identical payload. "No floor" is spelled by not declaring one."""
    assert Resources().manifest_dict() == {}
    assert "min_vram_gb" not in Resources(vcpus=4).manifest_dict()
    assert Resources().min_vram_gb is None


# ---------------------------------------------------------------------------
# THE REGRESSION ITSELF: a hint is not a gate, and never becomes one
# ---------------------------------------------------------------------------


def test_the_hint_ALONE_still_emits_no_gate_which_IS_the_defect() -> None:
    """This is the v2 hole, stated as an assertion rather than a story.

    `image_lora_finetuner 0.6.6` declares `Resources(vram_gb_hint=48)` and
    believes it has a 48 GB floor. It does not: the builder never reads
    `vram_gb_hint`, so nothing in the projection reaches the candidate filter.
    The row is here so that "the hint is not a gate" can never quietly stop
    being true — if some later change taught the hint to emit a floor, every
    existing advisory declaration in the fleet would silently become a hard
    refusal, which is precisely what pgw#647's freeze forbids."""
    projected = Resources(vram_gb_hint=48).manifest_dict()
    assert projected == {"gpu": True, "vram_gb_hint": 48.0}
    assert "min_vram_gb" not in projected
    assert "vram_gb" not in projected


def test_the_floor_and_the_hint_are_INDEPENDENT_axes() -> None:
    """Declaring a floor does not silently populate the hint, nor the reverse.
    They answer different questions and the payload carries both words."""
    both = Resources(min_vram_gb=80, vram_gb_hint=96).manifest_dict()
    assert both == {"gpu": True, "vram_gb_hint": 96.0, "min_vram_gb": 80.0}


def test_the_conversion_case_that_filed_this_issue_now_emits_its_floor() -> None:
    """te#113: conversion's `modelopt-quantization` declared `min_vram_gb: 80`
    + `compute_capability 8.9` under v1 and emitted NEITHER under v2, so an
    80 GB/sm_89 producer became placeable against a 7 GB release envelope.
    Both halves now ride the same declaration."""
    payload = Resources(
        gpu=True, vcpus=16, libraries=("modelopt",),
        min_vram_gb=80, compute_capability="sm_89",
    ).manifest_dict()
    assert payload["min_vram_gb"] == 80.0
    assert payload["compute_capability"] == 8.9
    assert payload["vcpus"] == 16


# ---------------------------------------------------------------------------
# It is a FLOOR, and the name says so
# ---------------------------------------------------------------------------


def test_there_is_no_second_spelling_for_an_author_to_reach_for() -> None:
    """v1's `vram_gb` must not come back as an alias: two words for one axis
    is how an author declares the advisory one and believes it is the gate.
    (The builder still maps `vram_gb` -> `min_vram_gb` as a FALLBACK for
    hand-written manifests, and its explicit key wins, so this SDK emitting
    only the explicit key can never be shadowed.)"""
    assert not hasattr(Resources(), "vram_gb")
    assert "vram_gb" not in Resources(min_vram_gb=80).manifest_dict()


def test_declaring_a_floor_implies_gpu() -> None:
    """Unlike `min_disk_gb`/`ram_gb_hint` (host-side allocation asks that must
    not rent a card), a VRAM floor is a statement about a GPU — same posture as
    `vram_gb_hint` and `compute_capability`. The builder's accelerator fold
    reads `gpu`, so without this a floor-only declaration would land with no
    accelerator resolved."""
    assert Resources(min_vram_gb=24).gpu is True
    assert Resources(min_vram_gb=24).manifest_dict()["gpu"] is True


# ---------------------------------------------------------------------------
# Contradictions cost a ValueError, never a build
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5])
def test_a_nonpositive_floor_is_a_DECLARATION_time_error(bad: float) -> None:
    with pytest.raises(ValueError, match="min_vram_gb must be positive"):
        Resources(min_vram_gb=bad)


def test_a_hint_BELOW_the_floor_is_refused() -> None:
    """The hint places the FIRST build. Placing it under the value the
    function says it cannot run below is a declaration that contradicts
    itself, and it is refused where it is written rather than shipped."""
    with pytest.raises(ValueError, match="below min_vram_gb"):
        Resources(min_vram_gb=80, vram_gb_hint=24)


def test_a_hint_at_or_above_the_floor_is_fine() -> None:
    """The hint may legitimately exceed the floor — "cannot run below 24, and
    place me on 80 for the first build" is coherent."""
    assert Resources(min_vram_gb=24, vram_gb_hint=80).min_vram_gb == 24.0
    assert Resources(min_vram_gb=24, vram_gb_hint=24).vram_gb_hint == 24.0


def test_an_int_floor_is_normalized_to_float_like_every_other_axis() -> None:
    """One numeric type on the wire; `80` and `80.0` must not be two payloads."""
    assert Resources(min_vram_gb=80).manifest_dict()["min_vram_gb"] == 80.0
    assert isinstance(Resources(min_vram_gb=80).min_vram_gb, float)
