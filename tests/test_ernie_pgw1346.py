"""pgw#1346 B3a — ERNIE-Image is declared, and the declaration is the truth.

The claims this file tests are the ones the B3a verdict rests on, each against
the thing it is a claim ABOUT:

1. **Base and Turbo are ONE model with two instances** — asserted against the
   pinned diffusers constructor's real signature, not against the two published
   config files' key sets.
2. **The bucket grid is the endpoint's own 14**, derived from its preset table
   and its CFG fork rather than transcribed from the declaration under test.
3. **CFG is a BATCH AXIS here** — one forward per step in both arms, which is
   what makes ``batch`` a bucket and distinguishes this family from Qwen-Image
   and FLUX.2-klein, where a negative branch is a second call.
4. **The ie#740 floor survives the Slot retirement BY VALUE.**
5. **The loop runs**, hubless and cardless, through the real typed callable.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import pytest

from gen_worker.model.catalog import Ernie
from gen_worker.model.catalog import ernie_serve as es
from gen_worker.model.catalog._packed_shape import pack_shape
from gen_worker.model.catalog.ernie import ERNIE, SCHEDULER, TRANSFORMER

torch = pytest.importorskip("torch")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor

#: The ernie endpoint's own preset table (`ernie/src/ernie/main.py`),
#: transcribed HERE so the test derives the buckets from the product decision
#: rather than from the declaration it is checking.
ENDPOINT_PRESETS: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1200, 896),
    (896, 1200),
    (1264, 848),
    (848, 1264),
    (1376, 768),
    (768, 1376),
)

#: `baidu/ERNIE-Image-Turbo`'s published transformer config, verbatim. The two
#: keys the base checkpoint does not carry are the whole question this file
#: settles.
TURBO_CONFIG: dict[str, object] = {
    "eps": 1e-06,
    "ffn_hidden_size": 12288,
    "hidden_size": 4096,
    "in_channels": 128,
    "lora_rank": 4,
    "num_attention_heads": 32,
    "num_layers": 36,
    "out_channels": 128,
    "patch_size": 1,
    "qk_layernorm": True,
    "rope_axes_dim": [32, 48, 48],
    "rope_theta": 256,
    "text_in_dim": 3072,
    "use_lora": False,
}


def _tuned(**overrides: object) -> es.ErnieTuned:
    return es.ErnieTuned(**overrides)  # type: ignore[arg-type]


# ------------------------------------------------- one model, two checkpoints


def test_base_and_turbo_construct_the_identical_module() -> None:
    """The B3a verdict for ERNIE, measured against the constructor itself.

    ``lora_rank`` and ``use_lora`` are the ONLY keys the Turbo config adds, and
    neither is a parameter of ``ErnieImageTransformer2DModel.__init__`` in the
    pinned diffusers — so they cannot change the module the mint traces, and a
    declaration that passed them would refuse to build. Everything else the two
    checkpoints publish is what this declaration already carries.
    """

    diffusers = pytest.importorskip("diffusers")
    accepted = set(
        inspect.signature(diffusers.ErnieImageTransformer2DModel.__init__).parameters
    ) - {"self"}
    extra = set(TURBO_CONFIG) - set(TRANSFORMER)
    assert extra == {"lora_rank", "use_lora"}
    assert not (extra & accepted)
    # ...and every key the declaration DOES carry is one the constructor takes,
    # so the build cannot fail on a key nobody checked.
    assert set(TRANSFORMER) <= accepted
    # The two configs agree on every key that reaches the constructor.
    shared = {key: TURBO_CONFIG[key] for key in TURBO_CONFIG if key in accepted}
    declared = {key: TRANSFORMER[key] for key in shared}
    assert {k: list(v) if isinstance(v, tuple) else v for k, v in declared.items()} == shared


def test_the_two_lanes_differ_only_in_tuned_values() -> None:
    """Two instances of one class, resolving the identical graph class."""

    base = Ernie.fake(tuned=_tuned(num_inference_steps=28, guidance=4.0))
    turbo = Ernie.fake(tuned=_tuned(num_inference_steps=8, guidance=1.0))

    assert type(base) is type(turbo) is Ernie
    assert base.tuned != turbo.tuned
    assert (
        base.variant("denoiser", {"batch": 1, "shape": 10241024}).ingress
        == turbo.variant("denoiser", {"batch": 1, "shape": 10241024}).ingress
    )


# ------------------------------------------------------------------ the grid


def test_the_shape_axis_is_the_endpoints_own_preset_table() -> None:
    """Seven presets, seven packed coordinates — recomputed, not trusted."""

    derived = tuple(sorted({pack_shape(w, h) for w, h in ENDPOINT_PRESETS}))
    assert derived == es.SHAPE_BUCKETS
    assert len(ENDPOINT_PRESETS) == len(es.SHAPE_BUCKETS) == 7


def test_the_declared_class_count_is_the_endpoints_own_fourteen() -> None:
    """Seven shapes x two CFG arms, and the endpoint says it cannot collapse.

    Its declaration states the reason: ERNIE's latents reach the DiT as spatial
    ``(B, C, H_lat, W_lat)`` through an ``nn.Conv2d`` patch embed, so 1200x896
    and 896x1200 are genuinely different conv graphs.
    """

    assert len(ERNIE.variants()) == 14
    assert es.BATCH_BUCKETS == (1, 2)


def test_transposed_presets_are_two_classes_with_two_latent_grids() -> None:
    """The measured reason there is no honest dedupe on this family."""

    assert pack_shape(1200, 896) == 12000896
    assert es.latent_shape(12000896) == (56, 75)
    assert es.latent_shape(8961200) == (75, 56)
    assert es.unpack_shape(12000896) == (1200, 896)


def test_every_preset_divides_by_the_combined_latent_stride() -> None:
    """The stride is 16 and the presets prove it.

    ``pipeline_ernie_image`` raises when ``height % vae_scale_factor != 0``, and
    1200 / 848 / 1264 / 1376 are divisible by 16 and NOT by 32 — so a 32 would
    refuse this endpoint's own 4:3 preset on every request.
    """

    assert es.LATENT_STRIDE == 16
    for width, height in ENDPOINT_PRESETS:
        assert width % 16 == 0 and height % 16 == 0
    assert any(width % 32 for width, _ in ENDPOINT_PRESETS)


# ---------------------------------------------------------------- the floors


def test_the_ie740_serving_floor_migrated_by_value() -> None:
    """K1: the retired Slot's requirements axis, moved and NOT re-derived.

    32 GB is a production floor recovered from the pre-wipe served release
    (th#1762) — undeclared for long enough that placement bought the cheapest
    card matching nothing. It is asserted as the parsed NUMBER a fit check
    compares, never as the string somebody must re-read.
    """

    requirements = ERNIE.layout_requirements
    assert requirements["plain.bf16@1"].minimum.min_vram_gb == 32.0
    # No SM floor on this lane: ERNIE serves bf16 only, so there is no fp8 rung
    # to guard and none is invented.
    assert requirements["plain.bf16@1"].minimum.min_sm == 0
    layouts = ERNIE.layouts
    assert layouts is not None
    assert tuple(layouts["*"]) == ("plain.bf16@1",)
    assert set(requirements) == {"plain.bf16@1"}


def test_the_weight_lane_is_not_a_traced_graph_variant() -> None:
    """The model's layout axis and the runner's traced-variant axis are different."""

    denoiser = next(row for row in ERNIE.runners if row.name == "denoiser")
    assert denoiser.layouts == ("bf16",)


# ------------------------------------------------------------------ the loop


def test_cfg_is_one_batched_forward_per_step_not_two_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"Does this lane use CFG" is a statement about the traced BATCH here.

    Counted at the typed callable, so the count is of REAL calls through the
    binding. Both arms make exactly one call per step — the guided arm just
    makes it at batch 2, which is why ``batch`` is a bucket axis on this family
    and a call count on Qwen-Image.
    """

    calls: list[int] = []
    real = Ernie.denoiser

    def counting(self: Any, **kwargs: Any) -> object:
        calls.append(int(kwargs["batch"]))
        return real(self, **kwargs)

    monkeypatch.setattr(Ernie, "denoiser", counting)
    instance = Ernie.fake(tuned=_tuned(num_inference_steps=3))
    for batch in (1, 2):
        calls.clear()
        text = torch.zeros(batch, es.TEXT_TOKENS, es.TEXT_WIDTH)
        es.generate(
            instance,
            shape=10241024,
            batch=batch,  # type: ignore[arg-type]
            text_bth=text,
            text_lens=torch.full((batch,), es.TEXT_TOKENS, dtype=torch.long),
            steps=3,
            guidance=4.0,
            seed=7,
        )
        assert calls == [batch] * 3


def test_a_fake_backed_ernie_runs_the_loop_through_the_typed_callable() -> None:
    """The whole loop, hubless and cardless, through the real code path.

    Not a mock of the SDK — it IS the SDK, with the only part that needs a card
    replaced. Returns VAE-ready latents rather than pixels, which is this
    family's declared boundary: the affine that follows is the VAE's own
    BatchNorm statistics, i.e. checkpoint WEIGHTS.
    """

    instance = Ernie.fake(tuned=_tuned(num_inference_steps=4))
    seen: list[tuple[int, int]] = []
    latents = es.generate(
        instance,
        shape=12000896,
        batch=2,
        text_bth=torch.zeros(2, es.TEXT_TOKENS, es.TEXT_WIDTH),
        text_lens=torch.full((2,), es.TEXT_TOKENS, dtype=torch.long),
        steps=3,
        guidance=4.0,
        seed=7,
        on_step=lambda index, total: seen.append((index, total)),
    )
    assert seen == [(0, 3), (1, 3), (2, 3)]
    # ONE sample out of a batch-2 graph: the second half of the batch is the
    # unconditional branch, not a second image.
    assert tuple(latents.shape) == (1, es.LATENT_CHANNELS, 56, 75)


def test_the_seed_reaches_the_latents() -> None:
    """Asserted at the latents, where a fake backing's output really varies."""

    def noise(seed: int) -> Tensor:
        return es.initial_latents(
            shape=10241024, batch=1, seed=seed, device="cpu", dtype=torch.float32
        )

    assert not torch.equal(noise(1), noise(2))
    assert torch.equal(noise(1), noise(1))
    assert tuple(noise(1).shape) == (1, es.LATENT_CHANNELS, 64, 64)


def test_the_uncond_row_comes_first_in_the_text_batch() -> None:
    """The pipeline's own order, and a swap inverts every prompt silently."""

    positive = torch.ones(1, es.TEXT_TOKENS, es.TEXT_WIDTH)
    negative = torch.zeros(1, es.TEXT_TOKENS, es.TEXT_WIDTH)
    batched, lengths = es.text_batch(positive, negative, lengths=(11, 22))
    assert tuple(batched.shape) == (2, es.TEXT_TOKENS, es.TEXT_WIDTH)
    assert float(batched[0, 0, 0]) == 0.0 and float(batched[1, 0, 0]) == 1.0
    assert lengths.tolist() == [11, 22]
    assert lengths.dtype == torch.int64
    # Over-length is clamped to the pin rather than silently traced longer.
    _, clamped = es.text_batch(positive, None, lengths=(9999,))
    assert clamped.tolist() == [es.TEXT_TOKENS]


# ------------------------------------------------------------ the declarations


def test_the_tuned_schema_carries_every_field_the_endpoint_stamps() -> None:
    """Field-for-field ``ErnieDefaults``, including the WIRE spelling.

    ``num_inference_steps`` keeps the wire name because the endpoint's
    ``RuntimeFormula`` resolves terms by same-named lookup over payload-then-
    recipe; renaming it would silently unresolve both lanes' steps term.
    """

    from gen_worker.model.runtime import tuned_fields

    assert set(tuned_fields(es.ErnieTuned)) == {
        "num_inference_steps",
        "guidance",
        "negative",
    }
    neutral = es.ErnieTuned()
    assert (neutral.num_inference_steps, neutral.guidance, neutral.negative) == (
        28,
        4.0,
        "",
    )


def test_the_scheduler_block_is_the_checkpoints_own_static_shift() -> None:
    """Both checkpoints publish the identical block, so one declaration carries it."""

    # A set of ONE, keyed by sampler (pgw#1346 K10).
    assert list(ERNIE.schedulers) == ["flow_match_euler"]
    assert ERNIE.schedulers["flow_match_euler"].name == "flow_match_euler_discrete"
    assert SCHEDULER["shift"] == 4.0
    assert SCHEDULER["use_dynamic_shifting"] is False
    # `shift_terminal` is published as null and is therefore ABSENT: a block
    # holds finite JSON scalars, and 0.0 would be a different ladder.
    assert "shift_terminal" not in SCHEDULER


def test_the_declaration_exposes_one_runner_and_one_loop_stage() -> None:
    """The honest shape: the endpoint compiles the transformer and nothing else."""

    assert tuple(row.name for row in ERNIE.runners) == ("denoiser",)
    assert ERNIE.loop is not None
    assert tuple(stage.runner for stage in ERNIE.loop.stages) == ("denoiser",)
    assert tuple(row.name for row in ERNIE.parameters) == ("steps",)
