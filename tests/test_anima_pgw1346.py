"""pgw#1346 B3b — Anima is declared, on the tier its serving path actually uses.

The claims this file tests, each against the thing it is a claim ABOUT:

1. **The tuned schema is the endpoint's, BY VALUE and BY NAME.** ``guidance``
   and ``negative`` are values; ``num_inference_steps`` is a NAME, and the
   endpoint's live ``RuntimeFormula`` resolves it by same-named lookup, so
   re-spelling it would silently break the formula rather than fail loudly.
2. **The ie#740 serving floors survive the Slot retirement** (K1) — including
   the one that is known to be too low, which is carried unchanged on purpose.
3. **The tokenizer siblings are eager models with NO tuned schema** (K5/K8), so
   they reach the hub's vocabulary with nothing, which is the intended answer.
4. **Anima's ladder is `flow_match_euler_discrete` with a static shift of 3.0**
   — differenced against a transcription of DiffSynth's own formula. This is
   the measurement that removes Anima from B3's "explicit-sigma ladders owed"
   column, and it is done at pgw#1346 B2's instrument: relative agreement, no
   ULP bound against a torch-derived reference.
5. **The eager tier is a declaration with no binding** (K11), asserted so the
   gap is a fact in the suite rather than a surprise at migration time.
"""

from __future__ import annotations

import msgspec
import pytest

from gen_worker.families import family_for
from gen_worker.model.catalog import anima_serve as an
from gen_worker.model.catalog.anima import (
    ANIMA,
    ANIMA_QWEN3_TOKENIZER,
    ANIMA_T5_TOKENIZER,
    PRESETS,
    TEXT_ENCODER,
    TRANSFORMER,
)
from gen_worker.model.spec import GraphModelSpec, ModelSpec

#: The endpoint's own recipe values (anima/src/anima/main.py `AnimaDefaults`),
#: transcribed HERE so the test compares the declaration against the product
#: decision rather than against itself.
ENDPOINT_NEUTRAL_STEPS = 35
ENDPOINT_NEUTRAL_GUIDANCE = 4.5
ENDPOINT_NEUTRAL_NEGATIVE = ""
#: The turbo distill's forced regime (`_TURBO_STEPS` / `_TURBO_CFG`).
ENDPOINT_TURBO_STEPS = 10
ENDPOINT_TURBO_CFG = 1.0

#: The endpoint's `_ANIMA_ASPECTS` grid, transcribed as (width, height).
ENDPOINT_PRESETS: tuple[tuple[int, int], ...] = (
    (1024, 1024), (1536, 1536), (1024, 768), (768, 1024),
    (1536, 1024), (1024, 1536), (1360, 768), (768, 1360),
)


def _diffsynth_z_image_sigmas(steps: int, shift: float = 3.0) -> tuple[float, ...]:
    """DiffSynth's ``set_timesteps_z_image``, transcribed.

    The REFERENCE, written out rather than imported, because ``diffsynth`` is
    not a gen-worker dependency — which is itself one of the two reasons Anima
    cannot be graph-tiered here. Transcribing it is what makes the comparison
    honest: the formula is short, it is quoted in ``anima_serve``'s docstring,
    and a reader can check both against the upstream file.

        sigmas = linspace(1.0, 0.0, N + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    """

    raw = [1.0 - index / steps for index in range(steps)]
    return tuple(shift * sigma / (1.0 + (shift - 1.0) * sigma) for sigma in raw)


# ------------------------------------------------------------- the tuned schema


def test_the_tuned_schema_is_the_endpoints_recipe_by_value() -> None:
    neutral = an.AnimaTuned()
    assert neutral.num_inference_steps == ENDPOINT_NEUTRAL_STEPS
    assert neutral.guidance == ENDPOINT_NEUTRAL_GUIDANCE
    assert neutral.negative == ENDPOINT_NEUTRAL_NEGATIVE
    assert ANIMA.tuned is an.AnimaTuned


def test_the_steps_field_carries_the_wire_name_because_a_formula_reads_it() -> None:
    """pgw#654 gap #4, asserted rather than commented.

    The anima endpoint declares ``RuntimeFormula("a + b*num_inference_steps")``
    and the formula resolves its terms by SAME-NAMED lookup across the payload
    and the resolved recipe. A tuned schema spelling the field ``steps`` leaves
    that term unresolvable, with nothing failing until a pod evaluates it.
    """

    names = {row.name for row in msgspec.structs.fields(an.AnimaTuned)}
    assert "num_inference_steps" in names
    assert "steps" not in names
    assert {row.name for row in msgspec.structs.fields(an.AnimaLoraTuned)} >= {
        "num_inference_steps"
    }


def test_the_lora_overlay_gives_the_turbo_distills_recipe_a_home() -> None:
    """The endpoint holds 10 and 1.0 as module constants and says they belong
    to the ADAPTER's own kind="lora" metadata. This schema is that home."""

    overlay = an.AnimaLoraTuned()
    assert overlay.num_inference_steps is None
    assert overlay.guidance is None
    assert overlay.trigger_words == ()
    typed = an.AnimaLoraTuned(
        num_inference_steps=ENDPOINT_TURBO_STEPS, guidance=ENDPOINT_TURBO_CFG
    )
    assert typed.num_inference_steps == 10 and typed.guidance == 1.0
    assert ANIMA.lora_tuned is an.AnimaLoraTuned


def test_the_tuned_schemas_are_published_under_the_models_own_name() -> None:
    """K8: an eager model WITH a tuned schema does reach the hub vocabulary."""

    assert family_for("anima") is an.AnimaTuned
    assert ANIMA.name == "anima"


# ------------------------------------------------------------ the ie#740 floors


def test_the_ie740_serving_floors_are_preserved_by_value() -> None:
    assert ANIMA.layouts == {"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")}
    requirements = ANIMA.layout_requirements
    assert requirements["cozy.fp8-rowwise@1"].minimum.min_sm == 89
    assert requirements["plain.bf16@1"].minimum.min_vram_gb == 8.0
    assert requirements["cozy.fp8-rowwise@1"].minimum.min_vram_gb == 0.0
    assert requirements["plain.bf16@1"].minimum.min_sm == 0


def test_the_bf16_floor_is_carried_unchanged_although_it_is_known_low() -> None:
    """A by-value migration migrates the number, including a wrong one.

    ie#706's census measured a 10.6 GiB peak against this declared 8. The
    migration's job is to move the declaration, not to invent a replacement —
    and an under-declared MINIMUM costs a degrade rung, not a refusal. This test
    exists so the 8 cannot be "tidied up" to a measured-looking number without
    someone deciding to.
    """

    assert ANIMA.layout_requirements["plain.bf16@1"].minimum.min_vram_gb == 8.0


# --------------------------------------------------------- the K5 aux siblings


def test_the_tokenizer_siblings_are_eager_models_with_no_tuned_schema() -> None:
    """K5's answer, made concrete: an auxiliary model is a ``Model``.

    And K8's corollary: with no tuned schema it registers NOTHING, so these two
    names never enter the hub's family vocabulary. A tokenizer answers no
    inference question and an empty schema under its name would be a word
    nothing can stamp.
    """

    for aux in (ANIMA_QWEN3_TOKENIZER, ANIMA_T5_TOKENIZER):
        assert isinstance(aux, ModelSpec) and not isinstance(aux, GraphModelSpec)
        assert aux.tuned is None and aux.lora_tuned is None
        assert aux.runners == ()
        assert family_for(aux.name) is None


def test_the_tokenizer_siblings_declare_their_layouts_undeclarable_by_value() -> None:
    """K2: DECLARED undeclarable, not undeclared — verbatim from the Slot.

    ``DEFAULT_LAYOUT`` is ``"bf16"``, so an undeclared model would silently
    claim a tensor-layout contract for bytes holding no tensors at all. That is
    worse than losing the field, which is exactly why the axis exists.
    """

    for aux in (ANIMA_QWEN3_TOKENIZER, ANIMA_T5_TOKENIZER):
        assert aux.layouts is None
        assert aux.layout_requirements == {}
        assert aux.layouts_undeclarable == (
            "tokenizer files only — no tensors, so no tensor-layout contract "
            "describes these bytes"
        )


def test_the_three_anima_models_are_distinct_names() -> None:
    names = {ANIMA.name, ANIMA_QWEN3_TOKENIZER.name, ANIMA_T5_TOKENIZER.name}
    assert names == {"anima", "anima_qwen3_tokenizer", "anima_t5_tokenizer"}


# ---------------------------------------------------------------- the schedule


@pytest.mark.parametrize("steps", [1, 4, 10, 20, 30, 35, 40, 50])
def test_animas_ladder_is_flow_match_euler_with_a_static_shift_of_three(
    steps: int,
) -> None:
    """The measurement that takes Anima OUT of B3's owed-scheduler column.

    ``FlowMatchScheduler("Z-Image")`` reads as a bespoke scheduler and is not
    one. Instrument per pgw#1346 B2: RELATIVE agreement against a transcribed
    reference, never a ULP bound — although here both sides are IEEE double
    arithmetic over the same closed form, so the agreement is far tighter than
    the bar.
    """

    reference = _diffsynth_z_image_sigmas(steps)
    resolved = an.schedule_for(steps)
    # Our ladder appends the terminal zero that DiffSynth's `step()` supplies
    # with its `sigma_ = 0` guard, so the tables line up entry for entry once
    # that zero is set aside.
    assert len(resolved.sigmas) == steps + 1
    assert resolved.sigmas[-1] == 0.0
    for index, (ours, theirs) in enumerate(zip(resolved.sigmas[:-1], reference, strict=True)):
        assert abs(ours - theirs) <= 2e-4 * max(1.0, abs(theirs)), index


def test_the_ladder_does_not_consult_resolution() -> None:
    """Static shift, so 512^2 and 1536^2 walk identical sigmas.

    Unlike FLUX.1-dev's dynamic shift, which interpolates on sequence length.
    Anything keying a schedule cache on resolution here is caching a constant.
    """

    assert an.scheduler().use_dynamic_shifting is False
    assert an.scheduler().shift == 3.0
    small = an.schedule_for(35)
    assert small.sigmas == an.schedule_for(35).sigmas
    # And the closed form needs no sequence length at all: passing one to the
    # scheduler is not even possible through the family's own entry point.
    assert an.schedule_for(35).num_train_timesteps == 1000


def test_the_ladder_is_monotone_and_terminates_at_zero() -> None:
    sigmas = an.schedule_for(35).sigmas
    assert sigmas[0] == pytest.approx(1.0)
    assert all(later < earlier for earlier, later in zip(sigmas, sigmas[1:], strict=False))
    assert sigmas[-1] == 0.0


def test_the_declared_scheduler_block_is_diffsynths_z_image_template() -> None:
    assert an.SCHEDULER == {
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "use_dynamic_shifting": False,
    }


# ------------------------------------------------------------------- the shapes


@pytest.mark.parametrize(("width", "height"), ENDPOINT_PRESETS)
def test_every_preset_is_a_whole_number_of_tokens(width: int, height: int) -> None:
    """The endpoint's own division factor is 16 = VAE stride x patch, and every
    preset is a multiple of it — so no served size is silently rounded up."""

    assert width % an.TOKEN_STRIDE == 0 and height % an.TOKEN_STRIDE == 0
    rows, cols = an.latent_grid(width, height)
    assert (rows, cols) == (height // 8, width // 8)
    assert an.denoiser_tokens(width, height) == (rows // 2) * (cols // 2)


def test_the_declared_presets_are_the_endpoints() -> None:
    assert PRESETS == ENDPOINT_PRESETS


def test_the_latent_shape_is_batch_one_sixteen_channels() -> None:
    """B=1 on every served request: the pipeline exposes no images-per-prompt
    knob at all, and its CFG runs two SEQUENTIAL forwards."""

    assert an.latent_shape(1536, 1536) == (1, 16, 192, 192)
    assert an.LATENT_CHANNELS == TRANSFORMER["in_channels"] == 16


def test_the_text_width_is_the_encoders_and_not_the_dits() -> None:
    """1024 vs 2048 — the mistake this pair of constants exists to prevent.

    Anima reads Qwen3-0.6B's LAST hidden state (not an intermediate stack the
    way FLUX.2-klein does), so cross-attention context is 1x the encoder width,
    while the DiT's own residual stream is twice that.
    """

    assert an.TEXT_WIDTH == 1024 == TEXT_ENCODER["hidden_size"]
    assert TRANSFORMER["crossattn_emb_channels"] == an.TEXT_WIDTH
    assert TRANSFORMER["model_channels"] == 2048
    assert TEXT_ENCODER["hidden_states_layer"] == -1
    assert TEXT_ENCODER["max_length"] == an.TEXT_TOKENS == 512


# ------------------------------------------------------------------ K11, stated


def test_the_eager_tier_is_a_declaration_with_no_binding() -> None:
    """pgw#1346 K11, asserted so it is a fact in the suite.

    ``ModelExport`` refuses a document with no runners, so codegen cannot render
    a ``Model`` subclass for an eager model and no endpoint can annotate a
    handler parameter with one. The declaration is real — it publishes a tuned
    schema and carries the serving floors — but ``inst.tuned`` is unreachable
    until the eager tier gets a binding. This gates every eager model in the
    program, so it must not be discovered one endpoint at a time.
    """

    from gen_worker.model.catalog import _FAMILIES

    assert ANIMA.runners == ()
    assert not isinstance(ANIMA, GraphModelSpec)
    assert not any(name.startswith("Anima") for name in _FAMILIES)
