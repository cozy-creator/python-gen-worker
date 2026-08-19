"""pgw#1420 — the ``ltx-2`` vocabulary, and the two hazards it had to clear.

Both are red/green'd here rather than argued in a docstring, because both
review as correct when they are wrong:

* the se#769 CLAMP-DIRECTION hazard (``_merge_int_knob`` floors a row's default
  INTO the platform range, so base-handler bounds ported up into a ModelType
  silently break every distilled sibling), and
* the FINGERPRINT-CAPTURE hazard specific to this family: ``ltx-2`` and
  ``ltx-2-upsampler`` are two roots whose names share a prefix.
"""

from __future__ import annotations

import msgspec
import pytest

from gen_worker.models import Ltx2, Ltx2Defaults, Ltx2Upsampler
from gen_worker.models.defaults_decode import decode_defaults
from gen_worker.models.model_types import (
    LTX2_SCHEDULER_CONFIG,
    Knob,
    MODEL_TYPES,
    model_type_by_name,
    model_type_for_contract,
)

#: The unserved `tensorhub/ltx-2.3-dev` fine-tune base: a 30-step CFG recipe,
#: i.e. the exact "distilled sibling" shape the clamp hazard corrupts. It is a
#: real catalog row (`jobs/catalog/launch_ltx_2026-07-07.toml` clones it to
#: `tensorhub/ltx-2.3-dev`), not a hypothetical.
_DEV_ROW = {
    "guidance": 3.0,
    "audio_guidance": 3.0,
    "sigmas": [1.0, 0.75, 0.5, 0.25],
    "cfg": True,
}


# ── the vocabulary itself ────────────────────────────────────────────────────


def test_the_root_is_the_registered_family_name_not_the_endpoint_slug() -> None:
    """`register_family("ltx-2", ...)` is the family owner's own shipped call;
    `ltx-video-2.3` is an endpoint slug and was never the vocabulary name."""

    assert Ltx2.name == "ltx-2"
    assert model_type_by_name("ltx-2") is Ltx2
    assert model_type_by_name("ltx-video-2.3") is None


def test_the_lane_document_resolves_and_is_not_a_sentinel() -> None:
    contract = Ltx2.canonical_contract
    assert contract is not None
    assert contract.stamp == "ltx-2.diffusers-bf16@1"
    # A `MissingContract` refuses every one of these; a real one answers.
    assert contract.dtype == "bfloat16"
    assert len(contract.digest) == 64


def test_the_upsampler_declares_no_contract_and_no_sentinel() -> None:
    """Absent is honest. A sentinel would be a standing lie with a to-do
    attached, and `Model[Ltx2Upsampler]` states `lanes=()` explicitly."""

    assert Ltx2Upsampler.canonical_contract is None


def test_the_scheduler_config_is_ltx_s_own_and_not_klein_s() -> None:
    """Fetched verbatim from the pinned revision. The three values below are
    exactly where a fill-by-analogy from FLUX.2 Klein would have gone wrong."""

    assert LTX2_SCHEDULER_CONFIG["use_dynamic_shifting"] is False  # Klein: True
    assert LTX2_SCHEDULER_CONFIG["shift"] == 1.0                   # Klein: 3.0
    assert LTX2_SCHEDULER_CONFIG["base_shift"] == 0.95             # Klein: 0.5
    assert LTX2_SCHEDULER_CONFIG["max_shift"] == 2.05              # Klein: 1.15


def test_both_types_are_registered() -> None:
    names = [mt.name for mt in MODEL_TYPES]
    assert "ltx-2" in names and "ltx-2-upsampler" in names


# ── hazard 1: fingerprint capture across a shared name prefix ────────────────


def test_the_ltx_2_fingerprint_does_not_capture_the_upsampler() -> None:
    """`ltx-2.*` is an fnmatch pattern and `ltx-2-upsampler...` shares its
    first five characters. If `.` were treated loosely the auxiliary's stamps
    would classify as the DiT — a wrong `model` column on every upsampler
    artifact, silently."""

    assert model_type_for_contract("ltx-2.diffusers-bf16@1") is Ltx2
    assert model_type_for_contract("ltx-2-upsampler.anything@1") is Ltx2Upsampler
    # and the reverse direction must not hold either
    assert model_type_for_contract("ltx-2-upsampler.anything@1") is not Ltx2


# ── hazard 2: the clamp direction, RED and GREEN ─────────────────────────────


def test_ltx2_declares_no_knob_at_all() -> None:
    """The structural reason this family is immune. A `Knob` is for a value the
    wire exposes and the platform must CLAMP; `ltx-video-2.3` exposes neither
    `steps` nor `guidance` and REJECTS an override with a typed 400 rather than
    narrowing it (ie#345). A knob that can only ever refuse is not a knob."""

    for field in msgspec.structs.fields(Ltx2Defaults):
        assert not isinstance(field.default, Knob), field.name


def test_a_distilled_sibling_row_survives_the_merge_unmodified() -> None:
    """GREEN. The `ltx-2.3-dev` 30-step CFG recipe must come back out exactly
    as written — no floor applied to `guidance`, no clamp on the sigma ladder,
    `cfg` flipped."""

    merged = decode_defaults(Ltx2Defaults, _DEV_ROW, model_name="ltx-2")

    assert merged.guidance == 3.0
    assert merged.audio_guidance == 3.0
    assert merged.sigmas == (1.0, 0.75, 0.5, 0.25)
    assert merged.cfg is True
    # untouched fields fall through to the platform opinion
    assert merged.stage2_sigmas == Ltx2Defaults().stage2_sigmas
    assert merged.max_sequence_length == 1024


def test_the_zero_arg_default_is_the_SERVED_distilled_recipe() -> None:
    """`Defaults()` zero-arg is the platform opinion and must be SERVABLE. The
    checkpoint this endpoint actually serves is the distilled one, so the 8
    fixed sigmas and CFG-off are the right default — not a base-model recipe
    the served weights would run wrong."""

    d = Ltx2Defaults()
    assert len(d.sigmas) == 8
    assert d.sigmas[0] == 1.0 and d.sigmas[-1] == 0.421875
    assert len(d.stage2_sigmas) == 3
    assert d.guidance == 1.0 and d.audio_guidance == 1.0
    assert d.cfg is False


def test_RED_the_same_row_shape_IS_clamped_when_the_field_is_a_knob() -> None:
    """The control. This proves the merge really does clamp — so the green
    above is a property of `Ltx2Defaults` having no knobs, and not of the test
    being unable to fail.

    A knob-bearing struct with a base-model floor of 30 steps, handed the
    distilled row's 10, comes back FLOORED to 30: three times the work, wrong
    regime. That is the defect verbatim, reproduced here on purpose.
    """

    class _KnobbyDefaults(msgspec.Struct, frozen=True):
        steps: Knob[int] = Knob(30, lo=30, hi=80, name="steps")
        guidance: Knob[float] = Knob(4.0, lo=1.0, name="guidance")

    clamped = decode_defaults(
        _KnobbyDefaults, {"steps": {"default": 10}, "guidance": {"default": 0.0}},
        model_name="knobby",
    )

    assert clamped.steps.default == 30, "the clamp hazard must reproduce"
    assert clamped.guidance.default == 1.0, "the clamp hazard must reproduce"


@pytest.mark.parametrize("field", ["sigmas", "stage2_sigmas"])
def test_a_malformed_sigma_ladder_refuses_rather_than_being_coerced(
    field: str,
) -> None:
    from gen_worker.models.defaults_decode import DefaultsDecodeError

    with pytest.raises(DefaultsDecodeError):
        decode_defaults(Ltx2Defaults, {field: "not-a-ladder"}, model_name="ltx-2")
