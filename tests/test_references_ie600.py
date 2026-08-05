"""ie#600 / ie#605: edit/compose references — mechanism tests.

Real payloads through the real validation path; no mocks. The four family
rows below are the ones the endpoint packages declare, so the per-family
behaviour under test is the shipped behaviour — including HiDream rendering
NO label.
"""

from __future__ import annotations

import pytest
from PIL import Image

from gen_worker import (
    EditFamily,
    FamilyGeometry,
    FitMode,
    OutputSize,
    ReferenceRefusal,
    condition_tokens,
    fit_edit_request,
    normalize_references,
    plan_edit,
)


# -- the fleet's four rows (ie#603 label provenance, ie#605 budgets) --------

QWEN = EditFamily(
    name="qwen-image-edit-2511",
    max_references=6,
    budget_basis=(
        "OUR compiled-shape and VRAM budget: every reference is resized to a "
        "constant 1024^2 area, so each adds a FIXED +4096 latent +~188 text "
        "tokens and the attention term grows quadratically (6 refs is ~12x the "
        "1-ref term at 1024^2). Not a validated quality limit"
    ),
    reference_label="Picture {n}",
    label_basis=(
        "proven in the token stream: pipeline_qwenimage_edit_plus.py:239-252 "
        "injects 'Picture N: <|vision_start|>...' before the user text"
    ),
    marginal_latent_tokens=4096,
    marginal_text_tokens=188,
)

KLEIN = EditFamily(
    name="flux2-klein",
    max_references=8,
    budget_basis=(
        "OUR compiled-shape and VRAM budget. BFL's 4 is their hosted-API slot "
        "count (Flux2KleinInputs stops at input_image_4) and does not bind the "
        "open weights we run"
    ),
    reference_label="image {n}",
    label_basis=(
        "BFL-documented USER convention "
        "(docs.bfl.ml/guides/prompting_editing_multi_reference); training basis "
        "UNPUBLISHED. Architecturally supported by the t_coords index channel"
    ),
    marginal_latent_tokens=1024,
)

HIDREAM = EditFamily(
    name="hidream-o1",
    max_references=11,
    budget_basis=(
        "the largest count the HiDream report publishes a measurement for "
        "(UniSubject '9-11 Subjects', arXiv 2605.11061 section 8) — an "
        "evaluation envelope we adopt as our budget, not a capability limit"
    ),
    reference_label=None,
    pipeline_appended_references=1,
)

COMPOSE_ONLY = EditFamily(
    name="compose-only",
    max_references=4,
    budget_basis="test row",
    has_edit_target=False,
)


def _image(width: int = 64, height: int = 64) -> Image.Image:
    return Image.new("RGB", (width, height), (128, 64, 32))


A, B, C = "asset-a", "asset-b", "asset-c"


# -- the declaration itself is gated ---------------------------------------


def test_a_cap_with_no_stated_reason_is_refused_at_declaration():
    with pytest.raises(ValueError, match="budget_basis"):
        EditFamily(name="x", max_references=3, budget_basis="  ")


def test_a_rendered_label_needs_its_provenance():
    with pytest.raises(ValueError, match="label_basis"):
        EditFamily(
            name="x", max_references=3, budget_basis="ours",
            reference_label="image {n}",
        )


def test_a_label_must_carry_the_position():
    with pytest.raises(ValueError, match=r"\{n\}"):
        EditFamily(
            name="x", max_references=3, budget_basis="ours",
            reference_label="Picture", label_basis="test",
        )


# -- normalization: both forms, one ordered list ---------------------------


def test_list_form_keeps_the_callers_order():
    refs = normalize_references([A, B, C], offset=1)
    assert [r.asset for r in refs] == [A, B, C]
    assert [r.position for r in refs] == [2, 3, 4]
    assert all(r.name is None for r in refs)


def test_map_form_is_ordered_by_sorted_key():
    refs = normalize_references({"pose": A, "background": B, "face": C}, offset=1)
    assert [r.name for r in refs] == ["background", "face", "pose"]
    assert [r.asset for r in refs] == [B, C, A]
    assert [r.position for r in refs] == [2, 3, 4]


def test_duplicate_name_is_refused():
    with pytest.raises(ReferenceRefusal) as exc:
        normalize_references({"Face": A, "face": B})
    assert exc.value.code == "duplicate_reference_name"


def test_illegal_name_is_refused():
    with pytest.raises(ReferenceRefusal) as exc:
        normalize_references({"1st-image": A})
    assert exc.value.code == "malformed_reference_name"


# -- the {name} rewrite, per family ----------------------------------------


def test_qwen_renders_picture_n_with_the_edit_target_at_position_1():
    request = plan_edit(
        family=QWEN, mode=FitMode.EDIT,
        prompt="the woman in the pose of {pose}, face from {face}, on {background}",
        image="target",
        references={"pose": A, "face": B, "background": C},
    )
    # sorted keys -> background=2, face=3, pose=4
    assert request.prompt.rendered == (
        "the woman in the pose of Picture 4, face from Picture 3, on Picture 2"
    )
    assert request.images == ("target", C, B, A)
    assert request.prompt.original.startswith("the woman in the pose of {pose}")


def test_klein_renders_image_n():
    request = plan_edit(
        family=KLEIN, mode=FitMode.EDIT,
        prompt="apply the pattern from {pattern} onto the plate",
        image="target", references={"pattern": A},
    )
    assert request.prompt.rendered == "apply the pattern from image 2 onto the plate"


def test_hidream_renders_no_label_and_refuses_the_rewrite():
    # ie#603: HiDream's condition tokens carry no index channel, and its own
    # paper section 2.4 publishes an entity-description caption recipe. An
    # ordinal there would be invented against a published counter-indication.
    with pytest.raises(ReferenceRefusal) as exc:
        plan_edit(
            family=HIDREAM, mode=FitMode.EDIT,
            prompt="place the subject from {subject} in a garden",
            image="target", references={"subject": A},
        )
    assert exc.value.code == "positional_labels_unsupported"
    assert "entity descriptions" in str(exc.value)


def test_hidream_list_form_still_works_naming_is_strictly_additive():
    request = plan_edit(
        family=HIDREAM, mode=FitMode.EDIT,
        prompt="place the woman in the red coat into the garden",
        image="target", references=[A, B],
    )
    assert request.images == ("target", A, B)
    assert not request.prompt.rewritten


# -- escaping ---------------------------------------------------------------


def test_double_braces_are_literal_braces():
    request = plan_edit(
        family=QWEN, mode=FitMode.EDIT,
        prompt='render the JSON {{"a": 1}} onto the sign from {logo}',
        image="target", references={"logo": A},
    )
    assert request.prompt.rendered == 'render the JSON {"a": 1} onto the sign from Picture 2'


def test_an_unescaped_json_prompt_is_refused_not_silently_passed():
    with pytest.raises(ReferenceRefusal) as exc:
        plan_edit(
            family=QWEN, mode=FitMode.EDIT, prompt='render {"a": 1} on the sign',
            image="target", references={"logo": A},
        )
    assert exc.value.code == "malformed_reference_name"


def test_a_lone_brace_is_refused():
    with pytest.raises(ReferenceRefusal) as exc:
        plan_edit(
            family=QWEN, mode=FitMode.EDIT, prompt="a 50% } discount",
            image="target", references={"logo": A},
        )
    assert exc.value.code == "malformed_reference_name"


# -- refusals happen BEFORE dispatch ---------------------------------------


def test_unknown_name_is_refused_and_lists_the_declared_ones():
    with pytest.raises(ReferenceRefusal) as exc:
        plan_edit(
            family=QWEN, mode=FitMode.EDIT, prompt="use the {fase} from here",
            image="target", references={"face": A, "pose": B},
        )
    assert exc.value.code == "unknown_reference_name"
    assert exc.value.name == "fase"
    assert "face, pose" in str(exc.value)


def test_a_name_in_the_prompt_with_a_list_payload_is_refused():
    with pytest.raises(ReferenceRefusal) as exc:
        plan_edit(
            family=QWEN, mode=FitMode.EDIT, prompt="use the {face} from here",
            image="target", references=[A, B],
        )
    assert exc.value.code == "named_references_not_declared"


def test_declared_but_unreferenced_is_accepted_and_reported():
    request = plan_edit(
        family=QWEN, mode=FitMode.EDIT, prompt="in the pose of {pose}",
        image="target", references={"pose": A, "style": B},
    )
    assert request.prompt.unreferenced == ("style",)
    assert request.prompt.substitutions == (("pose", 2),)


def test_names_are_case_insensitive_on_lookup():
    request = plan_edit(
        family=QWEN, mode=FitMode.EDIT, prompt="in the pose of {POSE}",
        image="target", references={"Pose": A},
    )
    assert request.prompt.rendered == "in the pose of Picture 2"


# -- caps: reject, never truncate ------------------------------------------


def test_over_cap_is_refused_not_truncated():
    with pytest.raises(ReferenceRefusal) as exc:
        plan_edit(
            family=QWEN, mode=FitMode.EDIT, prompt="x",
            image="target", references=[A] * 6,
        )
    assert exc.value.code == "too_many_references"
    # The refusal states OUR reason, not a vendor's number.
    assert "compiled-shape and VRAM budget" in str(exc.value)


def test_at_cap_is_accepted():
    request = plan_edit(
        family=QWEN, mode=FitMode.EDIT, prompt="x", image="target",
        references=[A] * 5,
    )
    assert len(request.images) == 6


def test_klein_accepts_eight():
    request = plan_edit(
        family=KLEIN, mode=FitMode.EDIT, prompt="x", image="target",
        references=[A] * 7,
    )
    assert len(request.images) == 8


def test_the_hidream_cap_counts_the_pipeline_appended_layout_image():
    # create_layout_reference_images APPENDS a synthetic layout canvas, so with
    # layout_bboxes set the pipeline sees N+1. The cap is POST-append: 11 total
    # means 10 user references when a layout is requested.
    ten = plan_edit(
        family=HIDREAM, mode=FitMode.EDIT, prompt="x", image="target",
        references=[A] * 9, pipeline_appends=1,
    )
    assert ten.total_references == 11
    with pytest.raises(ReferenceRefusal) as exc:
        plan_edit(
            family=HIDREAM, mode=FitMode.EDIT, prompt="x", image="target",
            references=[A] * 10, pipeline_appends=1,
        )
    assert exc.value.code == "too_many_references"
    assert "the pipeline appends 1 more" in str(exc.value)


def test_without_layout_the_same_family_takes_eleven_user_images():
    request = plan_edit(
        family=HIDREAM, mode=FitMode.EDIT, prompt="x", image="target",
        references=[A] * 10,
    )
    assert request.total_references == 11


# -- mode is declared, never inferred --------------------------------------


def test_edit_without_a_target_is_refused():
    with pytest.raises(ReferenceRefusal) as exc:
        plan_edit(family=QWEN, mode=FitMode.EDIT, prompt="x", references=[A])
    assert exc.value.code == "edit_target_missing"


def test_compose_refuses_an_edit_target():
    with pytest.raises(ReferenceRefusal) as exc:
        plan_edit(
            family=QWEN, mode=FitMode.COMPOSE, prompt="x", image="target",
            references=[A],
        )
    assert exc.value.code == "compose_takes_no_edit_target"


def test_compose_needs_at_least_one_reference():
    with pytest.raises(ReferenceRefusal) as exc:
        plan_edit(family=QWEN, mode=FitMode.COMPOSE, prompt="x", references=[])
    assert exc.value.code == "references_missing"


def test_compose_positions_start_at_one():
    request = plan_edit(
        family=QWEN, mode=FitMode.COMPOSE, prompt="{left} beside {right}",
        references={"left": A, "right": B},
    )
    assert request.prompt.rendered == "Picture 1 beside Picture 2"
    assert request.images == (A, B)


def test_a_compose_only_family_refuses_edit():
    with pytest.raises(ReferenceRefusal) as exc:
        plan_edit(
            family=COMPOSE_ONLY, mode=FitMode.EDIT, prompt="x", image="target",
        )
    assert exc.value.code == "edit_mode_unsupported"


# -- reproducibility: both prompts survive ---------------------------------


def test_both_prompts_are_carried_so_a_run_is_reproducible_from_its_record():
    original = "the woman in the pose of {pose} on {background}"
    request = plan_edit(
        family=QWEN, mode=FitMode.EDIT, prompt=original, image="target",
        references={"pose": A, "background": B},
    )
    assert request.prompt.original == original
    assert request.prompt.rendered != original
    # name -> marshalled position, so the run can be replayed exactly.
    assert dict(request.prompt.substitutions) == {"pose": 3, "background": 2}


# -- the ie#599 seam: one stage over one normalized request ----------------

GEOMETRY = FamilyGeometry(
    name="qwen-image-edit-2511",
    native_area=1024 * 1024,
    buckets=((1024, 1024), (1248, 832), (832, 1248)),
)


def test_edit_fits_the_targets_framing():
    request = plan_edit(
        family=QWEN, mode=FitMode.EDIT, prompt="brighten it", image="target",
        references=[A],
    )
    plan = fit_edit_request(request, [_image(1500, 1000), _image(600, 600)], GEOMETRY)
    assert plan.mode is FitMode.EDIT
    assert plan.bucket == (1248, 832)
    assert plan.source_size == (1500, 1000)


def test_compose_refuses_to_inherit_the_first_references_aspect():
    from gen_worker.api.errors import ValidationError

    request = plan_edit(
        family=QWEN, mode=FitMode.COMPOSE, prompt="x", references=[A, B],
    )
    with pytest.raises(ValidationError, match="owns output geometry"):
        fit_edit_request(request, [_image(1500, 1000), _image(600, 600)], GEOMETRY)


def test_compose_takes_an_explicit_output_bucket():
    request = plan_edit(
        family=QWEN, mode=FitMode.COMPOSE, prompt="x", references=[A, B],
    )
    plan = fit_edit_request(
        request, [_image(1500, 1000), _image(600, 600)], GEOMETRY,
        output_size=OutputSize.NATIVE, preset=(1024, 1024),
    )
    assert plan.bucket == (1024, 1024)


def test_decoded_image_count_must_match_the_marshalled_list():
    from gen_worker.api.errors import ValidationError

    request = plan_edit(
        family=QWEN, mode=FitMode.EDIT, prompt="x", image="target", references=[A],
    )
    with pytest.raises(ValidationError, match="marshalled references"):
        fit_edit_request(request, [_image()], GEOMETRY)


# -- cost ------------------------------------------------------------------


def test_qwen_marginal_cost_is_constant_per_reference():
    assert condition_tokens(QWEN, 1) == 4284
    assert condition_tokens(QWEN, 6) == 6 * 4284


def test_hidream_has_no_constant_marginal_cost():
    # It resizes every reference as K grows, so a linear estimate would lie.
    assert condition_tokens(HIDREAM, 4) is None
