"""pgw#1346 B1 — FLUX.2-klein-4B is declared, and the declaration is the truth.

The claims this file tests are the ones B1's verdict actually rests on, each
against the thing it is a claim ABOUT:

1. **The bucket axis is derived, not transcribed.** The nine declared token
   coordinates are recomputed from the endpoint's fifteen preset sizes. A
   hand-copied table that drifts from the presets is the failure this catches.
2. **The ie#740 serving floors survive the Slot retirement BY VALUE.** sm89 and
   30 GB are production floors (K1); losing one silently is the whole reason
   the declaration grew the axis.
3. **Base and Turbo are ONE model with two instances.** They differ only in
   tuned values and resolve the identical graph classes.
4. **The loop runs**, hubless and cardless, through the real typed callable.
5. **The joint dimension is derived from the text stack**, so the 7680 cannot
   drift from "three Qwen3 layers wide".
"""

from __future__ import annotations

import pytest

from gen_worker.model.catalog import Flux2Klein4b
from gen_worker.model.catalog import flux2_klein_4b_serve as kl
from gen_worker.model.catalog.flux2_klein_4b import (
    FLUX2_KLEIN_4B,
    SCHEDULER,
    TEXT_ENCODER,
    TOKEN_BUCKETS,
    TRANSFORMER,
)

torch = pytest.importorskip("torch")

#: The endpoint's own preset grid (flux.2-klein-4b/src/flux2_klein_4b/presets.py),
#: transcribed HERE so the test derives the buckets from the product decision
#: rather than from the declaration it is checking.
ENDPOINT_PRESETS: tuple[tuple[int, int], ...] = (
    (1024, 1024), (1184, 880), (880, 1184), (1248, 832), (832, 1248),
    (1392, 752), (752, 1392), (1568, 672), (672, 1568),
    (1408, 1408), (1920, 1088), (1088, 1920),
    (2048, 2048), (2560, 1440), (1440, 2560),
)


def _tuned(
    *, steps: int = 28, guidance: float = 4.0, distilled: bool = False
) -> kl.Flux2Klein4bTuned:
    return kl.Flux2Klein4bTuned(steps=steps, guidance=guidance, distilled=distilled)


# ----------------------------------------------------------------- the buckets


def test_the_declared_buckets_are_the_endpoint_presets_token_counts() -> None:
    """Fifteen preset sizes, nine token coordinates — recomputed, not trusted.

    This is the join between an ENDPOINT product grid (aspect ratios and
    megapixel tiers) and a FAMILY bucket axis. They are different vocabularies
    on purpose (greenfield B5), and this is the arithmetic that relates them.
    """

    derived = tuple(sorted({kl.packed_tokens(w, h) for w, h in ENDPOINT_PRESETS}))
    assert derived == TOKEN_BUCKETS
    assert len(ENDPOINT_PRESETS) == 15 and len(TOKEN_BUCKETS) == 9


def test_orientation_pairs_share_a_token_count_but_not_a_latent_grid() -> None:
    """Why the VAE decode cannot be keyed by this axis, asserted rather than said.

    1184x880 and 880x1184 are the same number of denoiser tokens and two
    different decoder output shapes. That is the measured reason the decoder is
    not a declared runner.
    """

    assert kl.packed_tokens(1184, 880) == kl.packed_tokens(880, 1184)
    assert kl.latent_grid(1184, 880) != kl.latent_grid(880, 1184)


def test_a_bucket_coordinate_is_a_square_of_the_token_stride() -> None:
    """1024px at stride 16 is 64x64 = 4096 tokens, the family's anchor."""

    assert kl.TOKEN_STRIDE == 16
    assert kl.latent_grid(1024, 1024) == (64, 64)
    assert kl.packed_tokens(1024, 1024) == 4096


# ------------------------------------------------------------------ the floors


def test_the_ie740_serving_floors_are_preserved_by_value() -> None:
    """K1: the retired Slot's requirements axis, moved and NOT re-derived.

    Both values are production floors. sm89 is the DECODABLE floor for the
    rowwise fp8 lane (the GEMM's sm90 is the fast path, not the floor), and the
    bf16 lane's 30 GB agrees with the endpoint's own
    ``aot/transformer-4b.mint.json`` ``declared_vram_gb``.
    """

    requirements = FLUX2_KLEIN_4B.layout_requirements
    # Parsed by Slot's OWN normalizer, so the floor survives as a number the
    # fit check can compare rather than as a string somebody must re-read.
    assert requirements["cozy.fp8-rowwise@1"].minimum.min_sm == 89
    assert requirements["plain.bf16@1"].minimum.min_vram_gb == 30.0
    # ...and each lane states only its own floor: the fp8 lane declares no VRAM
    # minimum and the bf16 lane no SM one. A requirement leaking across lanes
    # would decline cards that can serve.
    assert requirements["cozy.fp8-rowwise@1"].minimum.min_vram_gb == 0.0
    assert requirements["plain.bf16@1"].minimum.min_sm == 0
    layouts = FLUX2_KLEIN_4B.layouts
    assert layouts is not None
    assert tuple(layouts["*"]) == ("cozy.fp8-rowwise@1", "plain.bf16@1")


def test_the_weight_lane_is_not_a_traced_graph_variant() -> None:
    """fp8 is a LOAD-TIME rung, so it is a serving capability and not a class.

    The endpoint pins no flavor — th#546's fit ladder picks the rung per card —
    so the declaration states fp8 on the model's layout axis (what the code can
    execute) and leaves the runner's traced-variant axis at bf16 alone.
    Conflating the two would mint an artifact per weight lane.
    """

    denoiser = next(r for r in FLUX2_KLEIN_4B.runners if r.name == "denoiser")
    assert denoiser.layouts == ("bf16",)


# --------------------------------------------------------------- the two lanes


def test_base_and_turbo_are_two_instances_of_one_model() -> None:
    """The B1 verdict, asserted: same class, same graph classes, different tuned.

    FLUX.2-klein-4B and FLUX.2-klein-base-4B publish byte-identical transformer
    configs, so the thing that differs between them is entirely the recipe.
    """

    turbo_recipe = _tuned(steps=4, guidance=1.0, distilled=True)
    base_recipe = _tuned(steps=28, guidance=4.0, distilled=False)
    turbo = Flux2Klein4b.fake(tuned=turbo_recipe)
    base = Flux2Klein4b.fake(tuned=base_recipe)

    assert type(turbo) is type(base) is Flux2Klein4b
    # The published recipes, and the instance carries the one it was given.
    assert (turbo_recipe.steps, turbo_recipe.guidance) == (4, 1.0)
    assert (base_recipe.steps, base_recipe.guidance) == (28, 4.0)
    assert turbo_recipe.distilled is True and base_recipe.distilled is False
    assert turbo.tuned == turbo_recipe and base.tuned == base_recipe
    # The graph class each resolves for one bucket is the SAME class.
    assert (
        turbo.variant("denoiser", {"tokens": 4096}).ingress
        == base.variant("denoiser", {"tokens": 4096}).ingress
    )


@pytest.mark.parametrize(
    ("distilled", "guidance", "forwards"),
    [
        # Distilled: one forward per step however high guidance goes.
        (True, 4.0, 2),
        # Undistilled above 1.0: the negative branch is a SECOND sequential
        # forward per step — a call count, which is why every declared graph
        # class stays B=1 and CFG is not a bucket axis.
        (False, 4.0, 4),
        # Undistilled at 1.0: nothing to oppose, so no second forward.
        (False, 1.0, 2),
    ],
)
def test_cfg_is_a_call_count_not_a_batch_axis(
    monkeypatch: pytest.MonkeyPatch, distilled: bool, guidance: float, forwards: int
) -> None:
    """"Does this lane use CFG" is a statement about how many forwards run.

    Counted at the typed callable, on the class, so the count is of REAL calls
    through the binding rather than of a flag the loop set.
    """

    calls = 0
    real = Flux2Klein4b.denoiser

    def counting(self: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Flux2Klein4b, "denoiser", counting)
    instance = Flux2Klein4b.fake(tuned=_tuned(steps=2, distilled=distilled))
    kl.generate(
        instance,
        tokens=4096,
        width=1024,
        height=1024,
        prompt_embeds=torch.zeros(1, kl.TEXT_TOKENS, kl.JOINT_DIM),
        negative_embeds=torch.zeros(1, kl.TEXT_TOKENS, kl.JOINT_DIM),
        steps=2,
        guidance=guidance,
        seed=7,
        distilled=distilled,
    )
    assert calls == forwards


# -------------------------------------------------------------------- the loop


def test_a_fake_backed_klein_runs_the_loop_through_the_typed_callable() -> None:
    """The whole loop, hubless and cardless, through the real code path.

    Not a mock of the SDK — it IS the SDK, with the only part that needs a card
    replaced. Returns VAE-ready latents rather than pixels, which is the
    catalog's declared boundary for this family.
    """

    instance = Flux2Klein4b.fake(tuned=_tuned(steps=4, guidance=1.0, distilled=True))
    seen: list[tuple[int, int]] = []
    latents = kl.generate(
        instance,
        tokens=4096,
        width=1024,
        height=1024,
        prompt_embeds=torch.zeros(1, kl.TEXT_TOKENS, kl.JOINT_DIM),
        negative_embeds=None,
        steps=3,
        guidance=1.0,
        seed=7,
        distilled=True,
        on_step=lambda index, total: seen.append((index, total)),
    )
    assert seen == [(0, 3), (1, 3), (2, 3)]
    assert tuple(latents.shape) == (1, kl.PACKED_CHANNELS, 64, 64)


def test_the_seed_reaches_the_latents() -> None:
    """Asserted at the latents, where a fake backing's output really varies."""

    one = kl.initial_latents(
        width=1024, height=1024, batch=1, seed=1, device="cpu", dtype=torch.float32
    )
    two = kl.initial_latents(
        width=1024, height=1024, batch=1, seed=2, device="cpu", dtype=torch.float32
    )
    again = kl.initial_latents(
        width=1024, height=1024, batch=1, seed=1, device="cpu", dtype=torch.float32
    )
    assert not torch.equal(one, two)
    assert torch.equal(one, again)
    assert tuple(one.shape) == (1, 4096, kl.PACKED_CHANNELS)


def test_packing_round_trips(width: int = 1184, height: int = 880) -> None:
    """pack -> unpack is the identity, at a NON-square size deliberately."""

    rows, cols = kl.latent_grid(width, height)
    latents = torch.randn(1, kl.PACKED_CHANNELS, rows, cols)
    packed = kl.pack_latents(latents)
    assert tuple(packed.shape) == (1, rows * cols, kl.PACKED_CHANNELS)
    assert torch.equal(kl.unpack_latents(packed, rows=rows, cols=cols), latents)


def test_the_rope_grids_are_int64_and_four_wide() -> None:
    """Klein's coordinates are 4-D (T,H,W,L) and batched, unlike FLUX.1's."""

    image = kl.latent_image_ids(64, 64, batch=1, device="cpu")
    text = kl.text_ids(kl.TEXT_TOKENS, batch=1, device="cpu")
    assert tuple(image.shape) == (1, 4096, 4)
    assert tuple(text.shape) == (1, kl.TEXT_TOKENS, 4)
    assert image.dtype == torch.int64 and text.dtype == torch.int64
    # Text carries position on the LAST axis; the image grid on H and W.
    assert int(text[0, 5, 3]) == 5
    assert int(image[0, 65, 1]) == 1 and int(image[0, 65, 2]) == 1


# ------------------------------------------------------------ the declarations


def test_the_joint_dimension_is_three_stacked_qwen3_layers() -> None:
    """7680 is DERIVED, so it cannot drift from the layer list that implies it."""

    assert kl.JOINT_DIM == kl.TEXT_WIDTH * len(kl.TEXT_LAYERS) == 7680
    assert TRANSFORMER["joint_attention_dim"] == kl.JOINT_DIM
    assert TEXT_ENCODER["hidden_states_layers"] == kl.TEXT_LAYERS


def test_stacking_three_layers_produces_the_joint_width() -> None:
    """The stacking is family truth, so it is tested on real tensors."""

    hidden = [torch.zeros(1, kl.TEXT_TOKENS, kl.TEXT_WIDTH) for _ in range(28)]
    for position, index in enumerate(kl.TEXT_LAYERS, start=1):
        hidden[index] = torch.full(
            (1, kl.TEXT_TOKENS, kl.TEXT_WIDTH), float(position)
        )
    stacked = kl.stack_prompt_layers(hidden)
    assert tuple(stacked.shape) == (1, kl.TEXT_TOKENS, kl.JOINT_DIM)
    # Layer order is preserved across the concatenated feature axis.
    assert float(stacked[0, 0, 0]) == 1.0
    assert float(stacked[0, 0, kl.TEXT_WIDTH]) == 2.0
    assert float(stacked[0, 0, 2 * kl.TEXT_WIDTH]) == 3.0


def test_the_packed_channel_count_is_the_denoisers_input_width() -> None:
    """32 VAE latent channels x a 2x2 patch is the transformer's 128."""

    assert kl.PACKED_CHANNELS == kl.LATENT_CHANNELS * kl.PATCH * kl.PATCH == 128
    assert TRANSFORMER["in_channels"] == kl.PACKED_CHANNELS


def test_klein_takes_no_guidance_embedding() -> None:
    """guidance_embeds is FALSE, which is why guidance here means CFG."""

    assert TRANSFORMER["guidance_embeds"] is False


def test_the_scheduler_block_is_the_checkpoints_own() -> None:
    """Declared, so it rides the export digest rather than the math."""

    assert FLUX2_KLEIN_4B.scheduler is not None
    assert FLUX2_KLEIN_4B.scheduler.name == "flow_match_euler_discrete"
    assert SCHEDULER["use_dynamic_shifting"] is True
    assert SCHEDULER["base_image_seq_len"] == 256
    assert SCHEDULER["max_image_seq_len"] == 4096


def test_the_declaration_exposes_one_runner_and_one_loop_stage() -> None:
    """The honest shape: the endpoint compiles the transformer and nothing else."""

    assert tuple(r.name for r in FLUX2_KLEIN_4B.runners) == ("denoiser",)
    assert FLUX2_KLEIN_4B.loop is not None
    assert tuple(s.runner for s in FLUX2_KLEIN_4B.loop.stages) == ("denoiser",)
