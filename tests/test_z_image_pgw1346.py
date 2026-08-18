"""pgw#1346 B3a — Z-Image is declared, and the declaration is the truth.

Z-Image is the first catalog family whose SHAPE axis is not a bucket, and most
of what this file asserts is about why that is honest:

1. **Two declared classes, one per CFG arity** — the endpoint's own two
   ``aot/transformer-cfg-*.mint.json``, with the resolution symbolic inside
   each.
2. **The symbolic extents are only reachable because of two rewrites**, and
   both are differenced against the real upstream implementation rather than
   trusted as transcriptions.
3. **The declared symbol range is the PRESET GRID's range**, asserted on the
   committed export rather than on the source that produced it.
4. **Base and Turbo are ONE model with two instances**, differing in weights
   and in a published scheduler ``shift`` that had to become a tuned field.
5. **The three unusual facts of this loop** — ``1 - sigma`` conditioning, the
   negated output, and ``pos + scale * (pos - neg)`` — asserted on values the
   test controls.
6. **The ie#740 floors survive the Slot retirement BY VALUE.**
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from gen_worker.model.catalog import ZImage
from gen_worker.model.catalog import z_image_serve as zi
from gen_worker.model.catalog.z_image import SCHEDULER, TRANSFORMER, Z_IMAGE
from gen_worker.model.runtime import tuned_fields

torch = pytest.importorskip("torch")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor

#: `Tongyi-MAI/Z-Image-Turbo`'s published transformer config, verbatim. The
#: base checkpoint publishes the same keys plus ``siglip_feat_dim: null``.
TURBO_CONFIG: dict[str, object] = {
    "all_f_patch_size": [1],
    "all_patch_size": [2],
    "axes_dims": [32, 48, 48],
    "axes_lens": [1536, 512, 512],
    "cap_feat_dim": 2560,
    "dim": 3840,
    "in_channels": 16,
    "n_heads": 30,
    "n_kv_heads": 30,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "norm_eps": 1e-05,
    "qk_norm": True,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
}

#: The z-image endpoint's own preset grids (`z_image/presets.py`), transcribed
#: HERE so the declared symbol range is derived from the product decision
#: rather than from the declaration it is checking.
ENDPOINT_PRESETS: tuple[tuple[int, int], ...] = (
    (1024, 1024), (1152, 864), (864, 1152), (1248, 832),
    (832, 1248), (1280, 720), (720, 1280),
    (1408, 1408), (1920, 1088), (1088, 1920),
)


def _tuned(**overrides: object) -> zi.ZImageTuned:
    return zi.ZImageTuned(**overrides)  # type: ignore[arg-type]


def _captions(branches: int) -> Tensor:
    return torch.zeros(branches, zi.TEXT_TOKENS, zi.CAPTION_WIDTH)


# ------------------------------------------------------------- the two classes


def test_the_only_bucket_axis_is_the_cfg_arity() -> None:
    """Two declared classes, and they are the endpoint's own two artifacts.

    Ten preset rows collapse onto ONE program per CFG arm, which is the
    endpoint's ``shape_strategy="dynamic-collapse"``. #730's argument for
    keeping conv families on static buckets does not reach this family:
    ``transformer_z_image.py`` contains zero ``nn.Conv*`` layers.
    """

    assert tuple(bucket.name for bucket in Z_IMAGE.buckets) == ("branches",)
    assert zi.BRANCH_BUCKETS == (1, 2)
    assert len(Z_IMAGE.variants()) == 2


def test_both_arms_export_the_same_call_signature() -> None:
    """The measured reason the CFG arity is a DIMENSION and not a pytree list.

    ⚠️ pgw#1346 B3a, found by the export refusing: the module's own forward
    takes LISTS, and ``torch.export`` flattens a list into one input per
    element — so the two arms produced three flat inputs and five, and the
    export was refused with ``signature_disagreement``. torchcg G2 is right:
    ONE runner is ONE typed binding whose variants differ only in concrete
    dimensions. The declaration's wrapper therefore takes STACKED tensors and
    unbinds inside the traced region, which makes the arity a concrete leading
    dimension. Asserted on the COMMITTED export, so a future edit that
    reintroduces the list shape fails here rather than at the next mint.
    """

    instance = ZImage.fake(tuned=_tuned())
    one = instance.variant("denoiser", {"branches": 1}).ingress
    two = instance.variant("denoiser", {"branches": 2}).ingress
    assert [row.name for row in one.inputs] == [row.name for row in two.inputs]
    assert len(one.inputs) == 3
    assert one.flat_arity == two.flat_arity == 3
    # ...and the arity really is the leading dimension of each input.
    assert [row.shape[0] for row in one.inputs] == [1, 1, 1]
    assert [row.shape[0] for row in two.inputs] == [2, 2, 2]


def test_the_declared_symbol_range_is_the_preset_grids_range() -> None:
    """A dynamic axis whose bounds are invented is a class that misses a row.

    The smallest edge any preset asks for is 720px and the largest 1920, so the
    latent extents run 90..240. Read off the COMMITTED export's own symbol
    table, which is what a mint asserts against.
    """

    edges = [edge for shape in ENDPOINT_PRESETS for edge in shape]
    expected = (min(edges) // zi.VAE_STRIDE, max(edges) // zi.VAE_STRIDE)
    assert zi.latent_bounds() == expected == (90, 240)

    ingress = ZImage.fake(tuned=_tuned()).variant("denoiser", {"branches": 2}).ingress
    bounds = dict(ingress.symbols)
    assert bounds, "the exported program declares no symbolic extent at all"
    # Two symbols, both spanning the grid: the latent rows and columns.
    assert sorted(bounds.values()) == [expected, expected]
    # And they really reach the latent tensor's spatial axes.
    latents = next(row for row in ingress.inputs if row.name == "x")
    assert isinstance(latents.shape[3], str) and isinstance(latents.shape[4], str)
    assert latents.shape[:3] == (2, zi.LATENT_CHANNELS, 1)


def test_every_preset_row_lands_inside_the_declared_range() -> None:
    """The join between the product grid and the one declared class per arm."""

    low, high = zi.latent_bounds()
    for width, height in ENDPOINT_PRESETS:
        rows, cols = zi.latent_grid(width, height)
        assert low <= rows <= high and low <= cols <= high
        # The patchify folds 2x2, so both extents must be even.
        assert rows % zi.PATCH == 0 and cols % zi.PATCH == 0


# ------------------------------------------------- one model, two checkpoints


def test_base_and_turbo_declare_the_identical_architecture() -> None:
    """The B3a verdict for Z-Image: two INSTANCES, not two models."""

    declared = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in TRANSFORMER.items()
    }
    assert declared == TURBO_CONFIG


def test_the_two_checkpoints_differ_in_a_scheduler_value_and_that_is_tuned() -> None:
    """Why ``ZImageTuned`` grows a field the endpoint's schema does not have.

    The base checkpoint publishes ``shift: 6.0`` and the official DMD Turbo one
    ``shift: 3.0``. A declared family has ONE scheduler block, so a
    per-checkpoint shift has exactly one home — and left undeclared, the DMD
    lane would walk the base ladder on a nine-step walk.
    """

    instance = ZImage.fake(tuned=_tuned(shift=3.0))
    declared = zi.schedule_for(instance, steps=9)
    stamped = zi.schedule_for(
        instance, steps=9, shift=cast(zi.ZImageTuned, instance.tuned).shift
    )
    assert SCHEDULER["shift"] == 6.0
    assert declared.sigmas != stamped.sigmas
    # The tuned default is the BASE checkpoint's, so an unstamped instance
    # walks the declaration's own ladder rather than a silently different one.
    assert zi.ZImageTuned().shift == SCHEDULER["shift"]
    assert (
        zi.schedule_for(instance, steps=9, shift=zi.ZImageTuned().shift).sigmas
        == declared.sigmas
    )


def test_the_tuned_schema_carries_the_endpoints_fields_plus_the_shift() -> None:
    """``num_inference_steps`` keeps the WIRE spelling the formula resolves by."""

    assert set(tuned_fields(zi.ZImageTuned)) == {
        "num_inference_steps",
        "guidance",
        "shift",
    }
    neutral = zi.ZImageTuned()
    assert (neutral.num_inference_steps, neutral.guidance) == (28, 4.0)


# ---------------------------------------------------------------- the floors


def test_the_ie740_serving_floors_migrated_by_value() -> None:
    """K1: the retired Slot's requirements axis, moved and NOT re-derived.

    Both of the endpoint's slots declare the identical pair, which is exactly
    the case K1's ruling is built for: two bindings of one model state ONE
    demand. 40 GB agrees with both ``aot/transformer-cfg-*.mint.json``
    ``declared_vram_gb``.
    """

    requirements = Z_IMAGE.layout_requirements
    assert requirements["cozy.fp8-rowwise@1"].minimum.min_sm == 89
    assert requirements["plain.bf16@1"].minimum.min_vram_gb == 40.0
    assert requirements["cozy.fp8-rowwise@1"].minimum.min_vram_gb == 0.0
    assert requirements["plain.bf16@1"].minimum.min_sm == 0
    layouts = Z_IMAGE.layouts
    assert layouts is not None
    assert tuple(layouts["*"]) == ("cozy.fp8-rowwise@1", "plain.bf16@1")


def test_the_weight_lane_is_not_a_traced_graph_variant() -> None:
    """fp8 is a LOAD-TIME rung (th#546's fit ladder), so it is not a class."""

    denoiser = next(row for row in Z_IMAGE.runners if row.name == "denoiser")
    assert denoiser.layouts == ("bf16",)


# --------------------------------------------------------- the graph rewrites


def _bare() -> Any:
    """An object carrying upstream's own helpers and NO weights.

    ``patchify_and_embed`` reaches ``create_coordinate_grid`` (a staticmethod),
    ``_pad_with_ids`` and ``_patchify_image``, none of which touch a parameter —
    so both implementations can be differenced without constructing a 6B model.
    """

    diffusers = pytest.importorskip("diffusers")
    model = diffusers.ZImageTransformer2DModel

    class _Bare:
        # `staticmethod` again on the way in: reading it off the class hands
        # back a plain function, and a plain function assigned to a class
        # attribute would rebind `self` into its first argument.
        create_coordinate_grid = staticmethod(model.create_coordinate_grid)
        _pad_with_ids = model._pad_with_ids
        _patchify_image = model._patchify_image

    return _Bare()


@pytest.mark.parametrize(
    ("rows", "cols", "pad"),
    [
        # 16x16 latent -> 8x8 = 64 image tokens, a multiple of 32: pad 0.
        (16, 16, 0),
        # 8x12 -> 4x6 = 24 tokens: pad 8.
        (8, 12, 8),
        # 8x8 -> 16 tokens: pad 16.
        (8, 8, 16),
    ],
)
def test_the_pad_rewrite_is_value_identical_to_upstream(
    rows: int, cols: int, pad: int
) -> None:
    """ie#637's rewrite, differenced against the real upstream method.

    All three pads the served grid actually needs are covered — 1024x1024 needs
    0, 1248x832 needs 8, 1152x864 needs 16 — which is also the measured reason
    the branch cannot be decided once at trace time: a graph that did would
    serve one row correctly and lie about the other nine.
    """

    from gen_worker.model.catalog.z_image_graph import (
        UPSTREAM_PATCHIFY,
        patchify_and_embed,
    )

    bare = _bare()
    images = [torch.arange(16 * 1 * rows * cols, dtype=torch.float32).reshape(
        16, 1, rows, cols
    )]
    captions = [torch.arange(64 * 8, dtype=torch.float32).reshape(64, 8)]

    theirs = UPSTREAM_PATCHIFY(bare, images, captions, 2, 1)
    ours = patchify_and_embed(bare, images, captions, 2, 1)

    assert len(ours) == len(theirs) == 7
    # The pad this row exercises, confirmed rather than assumed.
    assert int(ours[0][0].shape[0]) - (rows // 2) * (cols // 2) == pad
    for index in range(7):
        mine: Any = ours[index]
        upstream: Any = theirs[index]
        if index == 2:  # the (F, H, W) size tuples
            assert mine == upstream
            continue
        for left, right in zip(mine, upstream, strict=True):
            assert torch.equal(left, right), f"return {index} differs"


def test_the_rope_tables_are_bit_identical_to_upstreams() -> None:
    """ie#630's rewrite: the same table, as buffers instead of baked constants.

    Two real float32 buffers per axis rather than one complex buffer, because a
    module-wide ``.to(bfloat16)`` casts complex tensors too and would silently
    discard the imaginary part. Recomposition must therefore be exact.
    """

    from gen_worker.model.catalog.z_image_graph import (
        UPSTREAM_ROPE,
        BoundRopeEmbedder,
        table_names,
    )

    dims = [32, 48, 48]
    lens = [128, 64, 64]
    ours = BoundRopeEmbedder(theta=256.0, axes_dims=dims, axes_lens=lens)
    theirs = UPSTREAM_ROPE.precompute_freqs_cis(dims, lens, theta=256.0)

    for mine, upstream in zip(ours.freqs_cis, theirs, strict=True):
        assert mine.dtype == upstream.dtype == torch.complex64
        assert torch.equal(mine, upstream)
    # The tables are BUFFERS, which is what keeps them out of the artifact as
    # anonymous constants (DESIGN-RULINGS §1.30).
    assert set(table_names(3)) <= set(dict(ours.named_buffers()))

    # And a gathered call agrees with upstream's own gather.
    ids = torch.stack(
        [torch.arange(7) % 128, torch.arange(7) % 64, torch.arange(7) % 64], dim=-1
    )
    reference = UPSTREAM_ROPE(theta=256.0, axes_dims=dims, axes_lens=lens)
    assert torch.equal(ours(ids), reference(ids))


def test_a_module_wide_bfloat16_cast_does_not_round_the_rope_tables() -> None:
    """The property plain attributes had for free and buffers must buy back."""

    from gen_worker.model.catalog.z_image_graph import BoundRopeEmbedder

    rope = BoundRopeEmbedder(theta=256.0, axes_dims=[32], axes_lens=[64])
    before = rope.freqs_cis[0].clone()
    rope.to(torch.bfloat16)
    assert torch.equal(rope.freqs_cis[0], before)


# ------------------------------------------------------------------ the loop


@pytest.mark.parametrize("branches", [1, 2])
def test_a_fake_backed_z_image_runs_the_loop_through_the_typed_callable(
    branches: int,
) -> None:
    """The whole loop, hubless and cardless, through the real code path.

    ONE forward per step in both arms — the guided arm is a batch-2 pytree, not
    two calls, which is the opposite of Qwen-Image and FLUX.2-klein. Returns
    VAE-ready latents in float32, which is the dtype the pipeline asserts its
    stepped latents keep.
    """

    calls: list[int] = []
    instance = ZImage.fake(tuned=_tuned(num_inference_steps=4))
    real = ZImage.denoiser

    def counting(self: Any, **kwargs: Any) -> object:
        calls.append(int(kwargs["branches"]))
        return real(self, **kwargs)

    seen: list[tuple[int, int]] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ZImage, "denoiser", counting)
        latents = zi.generate(
            instance,
            branches=branches,  # type: ignore[arg-type]
            width=1152,
            height=864,
            captions=_captions(branches),
            steps=3,
            guidance=4.0 if branches == 2 else 1.0,
            seed=7,
            on_step=lambda index, total: seen.append((index, total)),
        )
    assert calls == [branches] * 3
    assert seen == [(0, 3), (1, 3), (2, 3)]
    assert latents.dtype == torch.float32
    assert tuple(latents.shape) == (1, zi.LATENT_CHANNELS, 108, 144)


def test_the_denoiser_is_conditioned_on_one_minus_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``(1000 - t) / 1000`` in the pipeline, and t is ``sigma * 1000``.

    Asserted at the call because the two conventions are numerically plausible
    for each other — a schedule fed ``sigma`` here runs the trajectory
    backwards and still produces an image-shaped tensor.
    """

    seen: list[float] = []
    real = ZImage.denoiser

    def capturing(self: Any, **kwargs: Any) -> object:
        seen.append(float(kwargs["t"][0]))
        return real(self, **kwargs)

    monkeypatch.setattr(ZImage, "denoiser", capturing)
    instance = ZImage.fake(tuned=_tuned())
    schedule = zi.schedule_for(instance, steps=4)
    zi.generate(
        instance,
        branches=1,
        width=1024,
        height=1024,
        captions=_captions(1),
        steps=4,
        guidance=1.0,
        seed=7,
    )
    assert seen == pytest.approx([1.0 - sigma for sigma in schedule.sigmas[:-1]])


def test_the_output_is_negated_and_cfg_extrapolates_from_the_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two lines whose omission produces a confidently wrong image.

    Z-Image's DiT predicts the REVERSE velocity, and its CFG combines as
    ``pos + scale * (pos - neg)`` rather than ``neg + scale * (pos - neg)``.
    Measured on values the test controls: pos=2, neg=1, scale=4 gives 6, and
    the step must move by ``(sigma_next - sigma) * -6``.
    """

    instance = ZImage.fake(tuned=_tuned())
    latents = zi.initial_latents(
        width=1024, height=1024, batch=1, seed=1, device="cpu"
    )
    rows, cols = zi.latent_grid(1024, 1024)
    stacked = torch.stack(
        [
            torch.full((zi.LATENT_CHANNELS, 1, rows, cols), 2.0),
            torch.full((zi.LATENT_CHANNELS, 1, rows, cols), 1.0),
        ],
        dim=0,
    )
    monkeypatch.setattr(ZImage, "denoiser", lambda self, **kwargs: stacked)

    schedule = zi.schedule_for(instance, steps=3)
    index, stepped = next(
        zi.denoise(
            instance,
            branches=2,
            latents=latents,
            captions=_captions(2),
            schedule=schedule,
            guidance=4.0,
        )
    )
    assert index == 0
    delta = schedule.sigmas[1] - schedule.sigmas[0]
    expected = latents + delta * torch.full_like(latents, -6.0)
    assert torch.allclose(stepped, expected, atol=1e-5)


def test_the_positive_row_comes_first_in_the_caption_batch() -> None:
    """The pipeline's own order, and a swap extrapolates AWAY from the prompt."""

    positive = torch.ones(1, zi.TEXT_TOKENS, zi.CAPTION_WIDTH)
    negative = torch.zeros(1, zi.TEXT_TOKENS, zi.CAPTION_WIDTH)
    batched = zi.caption_states(positive, negative)
    assert tuple(batched.shape) == (2, zi.TEXT_TOKENS, zi.CAPTION_WIDTH)
    assert float(batched[0, 0, 0]) == 1.0 and float(batched[1, 0, 0]) == 0.0
    assert torch.equal(zi.caption_states(positive, None), positive)


def test_the_seed_reaches_the_latents() -> None:
    """Asserted at the latents, where a fake backing's output really varies."""

    def noise(seed: int) -> Tensor:
        return zi.initial_latents(
            width=1024, height=1024, batch=1, seed=seed, device="cpu"
        )

    assert not torch.equal(noise(1), noise(2))
    assert torch.equal(noise(1), noise(1))
    assert noise(1).dtype == torch.float32
    assert tuple(noise(1).shape) == (1, zi.LATENT_CHANNELS, 128, 128)


# ------------------------------------------------------------ the declaration


def test_the_declaration_exposes_one_runner_and_one_loop_stage() -> None:
    """The honest shape: the endpoint compiles the transformer and nothing else."""

    assert tuple(row.name for row in Z_IMAGE.runners) == ("denoiser",)
    assert Z_IMAGE.loop is not None
    assert tuple(stage.runner for stage in Z_IMAGE.loop.stages) == ("denoiser",)
    # A set of ONE, keyed by sampler (pgw#1346 K10).
    assert list(Z_IMAGE.schedulers) == ["flow_match_euler"]
    assert Z_IMAGE.schedulers["flow_match_euler"].name == "flow_match_euler_discrete"
    assert SCHEDULER["use_dynamic_shifting"] is False
