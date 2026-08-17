"""pgw#1346 B3a — Qwen-Image is declared, and the declaration is the truth.

The claims this file tests are the ones the B3a verdict rests on:

1. **The bucket axis is the endpoint's own fourteen presets**, recomputed from
   its two preset grids rather than transcribed from the declaration.
2. **The edit arm is a different MODEL, not an instance** — measured against
   the pinned diffusers constructor, which is where B1's class/instance rule
   actually bites.
3. **CFG is a CALL COUNT** — two sequential batch-1 forwards per guided step,
   with the norm-preserving rescale the pipeline applies.
4. **The ie#740 floors survive the Slot retirement BY VALUE.**
5. **The ladder is stretched**, and the SDK's own scheduler would not stretch
   it — the reason :mod:`gen_worker.model.flow_ladders` exists.
6. **The loop runs**, hubless and cardless, through the real typed callable.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import pytest

from gen_worker.model.catalog import QwenImage
from gen_worker.model.catalog import qwen_image_serve as qi
from gen_worker.model.catalog._packed_shape import pack_shape
from gen_worker.model.catalog.qwen_image import QWEN_IMAGE, SCHEDULER, TRANSFORMER
from gen_worker.model.runtime import tuned_fields
from gen_worker.model.scheduler import FlowMatchEulerDiscrete

torch = pytest.importorskip("torch")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor

#: The qwen-image endpoint's own preset grids (`qwen_image/presets.py`),
#: transcribed HERE so the test derives the buckets from the product decision
#: rather than from the declaration it is checking.
QWEN_ASPECTS: tuple[tuple[int, int], ...] = (
    (1328, 1328), (1472, 1104), (1104, 1472), (1584, 1056),
    (1056, 1584), (1664, 928), (928, 1664),
)
QWEN_ASPECTS_1MP: tuple[tuple[int, int], ...] = (
    (1024, 1024), (1152, 864), (864, 1152), (1248, 832),
    (832, 1248), (1280, 720), (720, 1280),
)

#: `Qwen/Qwen-Image-Edit-2511`'s published transformer config, verbatim — the
#: arm the batch plan proposed folding in as an instance.
EDIT_CONFIG: dict[str, object] = {
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": False,
    "in_channels": 64,
    "joint_attention_dim": 3584,
    "num_attention_heads": 24,
    "num_layers": 60,
    "out_channels": 16,
    "patch_size": 2,
    "zero_cond_t": True,
}


def _tuned(**overrides: object) -> qi.QwenImageTuned:
    return qi.QwenImageTuned(**overrides)  # type: ignore[arg-type]


def _prompt(tokens: int = qi.TEXT_TOKENS) -> Tensor:
    return torch.zeros(1, tokens, qi.JOINT_DIM)


# ------------------------------------------------------------------ the grid


def test_the_shape_axis_is_the_endpoints_own_two_preset_grids() -> None:
    """Fourteen presets, fourteen packed coordinates — recomputed, not trusted.

    This is the join between an ENDPOINT product grid (aspect ratios and
    megapixel tiers) and a FAMILY bucket axis. They are different vocabularies
    on purpose (greenfield B5), and this is the arithmetic that relates them.
    """

    derived = tuple(
        sorted({pack_shape(w, h) for w, h in QWEN_ASPECTS + QWEN_ASPECTS_1MP})
    )
    assert derived == qi.SHAPE_BUCKETS
    assert len(qi.SHAPE_BUCKETS) == 14
    # ...and the declared class count is the same fourteen, one per row: the
    # endpoint's own `aot/transformer-<w>x<h>.mint.json` set is fourteen files.
    assert len(QWEN_IMAGE.variants()) == 14


def test_transposed_presets_are_two_classes_because_the_patch_grid_is_baked() -> None:
    """Why this family's axis is a (w, h) pair and klein's is a token count.

    1472x1104 and 1104x1472 are the same number of tokens and two different
    patch grids, and the grid enters the traced call as PYTHON INTS inside
    ``img_shapes`` — so the rope tables are baked from it and the two are two
    programs. FLUX.2-klein's rope coordinates arrive as tensors, which is why
    its transposed presets collapse and these do not.
    """

    wide, tall = pack_shape(1472, 1104), pack_shape(1104, 1472)
    assert qi.packed_tokens(wide) == qi.packed_tokens(tall)
    assert qi.patch_grid(wide) != qi.patch_grid(tall)
    assert qi.img_shapes(wide) == [[(1, 69, 92)]]
    assert qi.img_shapes(tall) == [[(1, 92, 69)]]
    assert {wide, tall} <= set(qi.SHAPE_BUCKETS)


def test_a_bucket_coordinate_is_the_token_stride_squared() -> None:
    """1024px at stride 16 is 64x64 = 4096 tokens, the family's anchor."""

    assert qi.TOKEN_STRIDE == 16
    assert qi.patch_grid(pack_shape(1024, 1024)) == (64, 64)
    assert qi.packed_tokens(pack_shape(1024, 1024)) == 4096
    assert qi.PACKED_CHANNELS == qi.VAE_CHANNELS * qi.PATCH * qi.PATCH == 64
    assert TRANSFORMER["in_channels"] == qi.PACKED_CHANNELS


# -------------------------------------------------- the edit arm is not this one


def test_the_edit_arm_is_a_different_model_and_not_an_instance() -> None:
    """B1's class/instance rule, applied where it actually bites.

    ``zero_cond_t`` IS a constructor parameter of
    ``QwenImageTransformer2DModel``, so the edit checkpoint's config builds a
    different module — a different ``ModelSpec`` by construction, since an
    instance carries only weights, ``tuned`` and a ref label.

    The mirror case is in the same assertion and is why the rule needs
    measuring rather than eyeballing: the t2i checkpoint publishes
    ``pooled_projection_dim``, which is NOT a constructor parameter in the
    pinned diffusers and therefore changes nothing — a config key difference is
    not automatically an architecture difference.
    """

    diffusers = pytest.importorskip("diffusers")
    accepted = set(
        inspect.signature(diffusers.QwenImageTransformer2DModel.__init__).parameters
    ) - {"self"}
    assert "zero_cond_t" in accepted
    assert EDIT_CONFIG["zero_cond_t"] is True
    assert TRANSFORMER.get("zero_cond_t") is None
    # The published t2i config's own extra key, which is inert.
    assert "pooled_projection_dim" not in accepted
    assert "pooled_projection_dim" not in TRANSFORMER
    # Everything the declaration carries is a real constructor parameter.
    assert set(TRANSFORMER) <= accepted


# ---------------------------------------------------------------- the floors


def test_the_ie740_serving_floors_migrated_by_value() -> None:
    """K1: the retired Slot's requirements axis, moved and NOT re-derived.

    ``sm89+`` is the DECODABLE floor for the rowwise fp8 lane — the rowwise
    GEMM's sm90 is the fast path, not the floor — and 72 GB is the bf16 lane's,
    agreeing with all fourteen ``aot/transformer-*.mint.json``
    ``declared_vram_gb``. Asserted as the parsed NUMBERS a fit check compares.
    """

    requirements = QWEN_IMAGE.layout_requirements
    assert requirements["cozy.fp8-rowwise@1"].minimum.min_sm == 89
    assert requirements["plain.bf16@1"].minimum.min_vram_gb == 72.0
    # Each lane states only its own floor: a requirement leaking across lanes
    # would decline cards that can serve.
    assert requirements["cozy.fp8-rowwise@1"].minimum.min_vram_gb == 0.0
    assert requirements["plain.bf16@1"].minimum.min_sm == 0
    layouts = QWEN_IMAGE.layouts
    assert layouts is not None
    assert tuple(layouts["*"]) == ("cozy.fp8-rowwise@1", "plain.bf16@1")


def test_the_weight_lane_is_not_a_traced_graph_variant() -> None:
    """fp8 is a LOAD-TIME rung (th#546's fit ladder), so it is not a class."""

    denoiser = next(row for row in QWEN_IMAGE.runners if row.name == "denoiser")
    assert denoiser.layouts == ("bf16",)


# --------------------------------------------------------------- the scheduler


def test_the_declared_block_is_the_checkpoints_own_including_the_stretch() -> None:
    """Two of these keys are read by nothing else in the SDK, and both matter."""

    assert QWEN_IMAGE.scheduler is not None
    assert QWEN_IMAGE.scheduler.name == "flow_match_euler_discrete"
    assert SCHEDULER["use_dynamic_shifting"] is True
    assert SCHEDULER["shift_terminal"] == 0.02
    assert SCHEDULER["max_image_seq_len"] == 8192
    assert SCHEDULER["max_shift"] == 0.9


def test_the_served_ladder_is_stretched_and_the_sdk_scheduler_alone_is_not() -> None:
    """``schedule_for`` reads two keys ``instance.scheduler()`` cannot.

    Measured rather than asserted in prose: same declared block, same step
    count, two different ladders — and the stretched one is the one the
    pipeline walks.
    """

    instance = QwenImage.fake(tuned=_tuned())
    shape = pack_shape(1328, 1328)
    served = qi.schedule_for(instance, steps=30, shape=shape)
    unstretched = instance.scheduler().schedule(
        30, image_seq_len=qi.packed_tokens(shape)
    )
    assert isinstance(instance.scheduler(), FlowMatchEulerDiscrete)
    assert served.sigmas != unstretched.sigmas
    assert served.sigmas[-2] == pytest.approx(0.02, abs=1e-12)


def test_the_ladder_consults_the_resolution() -> None:
    """Dynamic shifting is ON, so two presets at one step count differ."""

    instance = QwenImage.fake(tuned=_tuned())
    small = qi.schedule_for(instance, steps=20, shape=pack_shape(720, 1280))
    large = qi.schedule_for(instance, steps=20, shape=pack_shape(1328, 1328))
    assert small.sigmas != large.sigmas


# ------------------------------------------------------------------ the loop


@pytest.mark.parametrize(
    ("guidance", "negative", "forwards"),
    [
        # True CFG: a SECOND sequential batch-1 forward per step.
        (4.0, True, 6),
        # At or below 1.0 there is nothing to oppose, so no second forward.
        (1.0, True, 3),
        # No negative branch supplied at all.
        (4.0, False, 3),
    ],
)
def test_cfg_is_a_call_count_not_a_batch_axis(
    monkeypatch: pytest.MonkeyPatch, guidance: float, negative: bool, forwards: int
) -> None:
    """"Does this lane use CFG" is a statement about how many forwards run.

    Counted at the typed callable, so the count is of REAL calls through the
    binding — and every one of them is batch 1, which is why no CFG axis
    appears in this family's buckets.
    """

    calls: list[int] = []
    real = QwenImage.denoiser

    def counting(self: Any, **kwargs: Any) -> object:
        calls.append(int(kwargs["hidden_states"].shape[0]))
        return real(self, **kwargs)

    monkeypatch.setattr(QwenImage, "denoiser", counting)
    instance = QwenImage.fake(tuned=_tuned(steps=3))
    qi.generate(
        instance,
        shape=pack_shape(1024, 1024),  # type: ignore[arg-type]
        prompt_embeds=_prompt(),
        prompt_mask_ids=qi.prompt_mask([qi.TEXT_TOKENS], device="cpu"),
        negative_embeds=_prompt() if negative else None,
        negative_mask_ids=qi.prompt_mask([1], device="cpu") if negative else None,
        steps=3,
        guidance=guidance,
        seed=7,
    )
    assert calls == [1] * forwards


def test_the_denoiser_is_conditioned_on_the_sigma_not_the_moment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``timestep / 1000`` in the pipeline, so the input IS the sigma.

    Getting this wrong conditions the model on a step a thousand times too
    large and renders noise, so it is asserted at the call rather than trusted
    to a comment.
    """

    seen: list[float] = []
    real = QwenImage.denoiser

    def capturing(self: Any, **kwargs: Any) -> object:
        seen.append(float(kwargs["timestep"][0]))
        return real(self, **kwargs)

    monkeypatch.setattr(QwenImage, "denoiser", capturing)
    instance = QwenImage.fake(tuned=_tuned())
    shape = pack_shape(1024, 1024)
    schedule = qi.schedule_for(instance, steps=4, shape=shape)
    qi.generate(
        instance,
        shape=shape,  # type: ignore[arg-type]
        prompt_embeds=_prompt(),
        prompt_mask_ids=qi.prompt_mask([qi.TEXT_TOKENS], device="cpu"),
        negative_embeds=None,
        negative_mask_ids=None,
        steps=4,
        guidance=1.0,
        seed=7,
    )
    assert seen == pytest.approx(list(schedule.sigmas[:-1]), abs=1e-6)
    assert max(seen) <= 1.0


def test_the_true_cfg_combination_preserves_the_conditional_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline's own rescale, reproduced on known tensors.

    ``comb * (||cond|| / ||comb||)`` per token. Dropping it shifts contrast and
    saturation on every guided render — a difference that reads as a different
    checkpoint rather than as a bug, which is why it is measured here on values
    the test controls rather than left to the fake backing's noise.
    """

    instance = QwenImage.fake(tuned=_tuned())
    shape = pack_shape(1024, 1024)
    tokens = qi.packed_tokens(shape)
    outputs = iter(
        [
            torch.full((1, tokens, qi.PACKED_CHANNELS), 2.0),
            torch.full((1, tokens, qi.PACKED_CHANNELS), 1.0),
        ]
    )
    monkeypatch.setattr(
        QwenImage, "denoiser", lambda self, **kwargs: next(outputs)  # noqa: ARG005
    )

    schedule = qi.schedule_for(instance, steps=2, shape=shape)
    latents = qi.initial_latents(
        shape=shape, batch=1, seed=1, device="cpu", dtype=torch.float32
    )
    index, stepped = next(
        qi.denoise(
            instance,
            shape=shape,  # type: ignore[arg-type]
            latents=latents,
            prompt_embeds=_prompt(),
            prompt_mask_ids=qi.prompt_mask([qi.TEXT_TOKENS], device="cpu"),
            negative_embeds=_prompt(),
            negative_mask_ids=qi.prompt_mask([1], device="cpu"),
            schedule=schedule,
            guidance=4.0,
        )
    )
    # combined = uncond + 4*(cond - uncond) = 5 everywhere; rescaled to the
    # conditional's own per-token norm it is 2 again — the conditional itself.
    assert index == 0
    expected = latents + (schedule.sigmas[1] - schedule.sigmas[0]) * torch.full_like(
        latents, 2.0
    )
    assert torch.allclose(stepped, expected, atol=1e-5)


def test_a_fake_backed_qwen_runs_the_loop_through_the_typed_callable() -> None:
    """The whole loop, hubless and cardless, through the real code path.

    Returns VAE-ready latents rather than pixels — the decode is not a declared
    runner and the affine is the VAE's own ``latents_mean``/``latents_std``.
    The 5-D result is ``AutoencoderKLQwenImage``'s own input shape: it decodes
    video-shaped latents and takes a still image as one frame.
    """

    instance = QwenImage.fake(tuned=_tuned(steps=4))
    seen: list[tuple[int, int]] = []
    latents = qi.generate(
        instance,
        shape=pack_shape(1472, 1104),  # type: ignore[arg-type]
        prompt_embeds=_prompt(),
        prompt_mask_ids=qi.prompt_mask([qi.TEXT_TOKENS], device="cpu"),
        negative_embeds=_prompt(),
        negative_mask_ids=qi.prompt_mask([2], device="cpu"),
        steps=3,
        guidance=4.0,
        seed=7,
        on_step=lambda index, total: seen.append((index, total)),
    )
    assert seen == [(0, 3), (1, 3), (2, 3)]
    assert tuple(latents.shape) == (1, qi.VAE_CHANNELS, 1, 138, 184)


def test_packing_round_trips_at_a_non_square_shape() -> None:
    """pack -> unpack is the identity, at a NON-square size deliberately."""

    shape = pack_shape(1472, 1104)
    rows, cols = qi.patch_grid(shape)
    latents = torch.randn(1, qi.VAE_CHANNELS, rows * qi.PATCH, cols * qi.PATCH)
    packed = qi.pack_latents(latents)
    assert tuple(packed.shape) == (1, rows * cols, qi.PACKED_CHANNELS)
    restored = qi.unpack_latents(packed, shape=shape)
    assert torch.equal(restored, latents.unsqueeze(2))


def test_the_seed_reaches_the_latents() -> None:
    """Asserted at the latents, where a fake backing's output really varies."""

    def noise(seed: int) -> Tensor:
        return qi.initial_latents(
            shape=pack_shape(1024, 1024),
            batch=1,
            seed=seed,
            device="cpu",
            dtype=torch.float32,
        )

    assert not torch.equal(noise(1), noise(2))
    assert torch.equal(noise(1), noise(1))
    assert tuple(noise(1).shape) == (1, 4096, qi.PACKED_CHANNELS)


# ------------------------------------------------------------ the declarations


def test_the_tuned_schema_carries_every_field_the_endpoint_stamps() -> None:
    """Field-for-field ``QwenImageDefaults``, values included.

    ``negative`` is Qwen's single-space convention and not the empty string:
    the pipeline's true-CFG branch treats an empty negative as "no
    unconditional prompt", which is a different render.
    """

    assert set(tuned_fields(qi.QwenImageTuned)) == {
        "steps",
        "guidance",
        "negative",
        "max_guidance",
    }
    neutral = qi.QwenImageTuned()
    assert (neutral.steps, neutral.guidance, neutral.negative) == (30, 4.0, " ")
    assert neutral.max_guidance is None


def test_the_declaration_exposes_one_runner_and_one_loop_stage() -> None:
    """The honest shape: the endpoint compiles the transformer and nothing else."""

    assert tuple(row.name for row in QWEN_IMAGE.runners) == ("denoiser",)
    assert QWEN_IMAGE.loop is not None
    assert tuple(stage.runner for stage in QWEN_IMAGE.loop.stages) == ("denoiser",)


def test_the_prompt_mask_is_int64_and_covers_the_pinned_window() -> None:
    """The pin always hands the denoiser a mask, so the signature is invariant."""

    mask = qi.prompt_mask([11], device="cpu")
    assert tuple(mask.shape) == (1, qi.TEXT_TOKENS)
    assert mask.dtype == torch.int64
    assert int(mask.sum()) == 11
    # Over-length clamps to the pin rather than silently widening the graph.
    assert int(qi.prompt_mask([9999], device="cpu").sum()) == qi.TEXT_TOKENS
