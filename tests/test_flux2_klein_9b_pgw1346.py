"""pgw#1346 B3b — FLUX.2-klein-9B is declared, and its config has a SOURCE.

B1 recorded klein-9B as unauthorable: no 9B ``transformer/config.json`` exists
on any authoring box and both 9b endpoints deliberately carry no checkpoint ref
(ie#524/th#980), so the class-level width had nothing to be derived from. This
lane resolved that by reading the SERVING RELEASE'S OWN published configs out of
the hub catalog — seven JSON documents, 4.8 KB, no weights — and committing them
beside this file with their release ids and digests.

So the first thing this file tests is the SOURCE, not the declaration:

1. **The fixture is what it says it is.** Every cached file re-hashes to the
   digest ``PROVENANCE.json`` records. A silently edited fixture fails here
   rather than quietly re-keying the declaration that reads from it.
2. **The declaration is the fixture.** Every architecture number in
   ``TRANSFORMER`` is asserted against the fetched config, field by field — so
   the 9B width is a MEASUREMENT with a receipt, which is exactly what B1 said
   was missing.
3. **9B is not 4B.** The two configs differ in depth and width, which is why
   this is a second ``ModelSpec`` rather than a third klein instance.
4. **Base and Turbo ARE one model.** Their transformer configs differ only in
   ``_name_or_path`` — a provenance string with no architecture in it.
5. **The ie#740 serving floors survive the Slot retirement BY VALUE** (K1):
   sm89 and 44 GB are production floors, asserted as parsed NUMBERS.
6. **The bucket axis is derived**, from the 9b endpoint's own preset grid.
7. **The loop runs**, hubless and cardless, through the real typed callable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from gen_worker.model.catalog import Flux2Klein9b
from gen_worker.model.catalog import flux2_klein_4b_serve as kl
from gen_worker.model.catalog import flux2_klein_9b_serve as kl9
from gen_worker.model.catalog.flux2_klein_4b import FLUX2_KLEIN_4B
from gen_worker.model.catalog.flux2_klein_4b import TRANSFORMER as TRANSFORMER_4B
from gen_worker.model.catalog.flux2_klein_9b import (
    FLUX2_KLEIN_9B,
    TEXT_ENCODER,
    TRANSFORMER,
)

torch = pytest.importorskip("torch")

FIXTURES = Path(__file__).parent / "fixtures" / "flux2_klein_9b"

#: The 9b endpoint's own preset grid (flux.2-klein-9b/src/flux2_klein_9b/presets.py),
#: transcribed HERE so the test derives the buckets from the product decision
#: rather than from the declaration it is checking. Byte-identical to the 4b
#: endpoint's grid, which is why the two models share a bucket axis.
ENDPOINT_PRESETS: tuple[tuple[int, int], ...] = (
    (1024, 1024), (1184, 880), (880, 1184), (1248, 832), (832, 1248),
    (1392, 752), (752, 1392), (1568, 672), (672, 1568),
    (1408, 1408), (1920, 1088), (1088, 1920),
    (2048, 2048), (2560, 1440), (1440, 2560),
)

#: The endpoint's retired `Slot` recipe, migrated by value (th#1116's SCHEMA;
#: the neutral values are BFL's Base card numbers).
ENDPOINT_NEUTRAL_STEPS = 28
ENDPOINT_NEUTRAL_GUIDANCE = 4.0
#: The distilled lane's published recipe (`_TURBO_STEPS` / `_TURBO_GUIDANCE`).
ENDPOINT_TURBO_STEPS = 4
ENDPOINT_TURBO_GUIDANCE = 1.0


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text("utf-8"))


def _provenance() -> Any:
    return _fixture("PROVENANCE.json")


def _tuned(
    *, steps: int = 28, guidance: float = 4.0, distilled: bool = False
) -> kl9.Flux2Klein9bTuned:
    return kl9.Flux2Klein9bTuned(steps=steps, guidance=guidance, distilled=distilled)


# ------------------------------------------------------------- the config SOURCE


def test_every_cached_config_matches_its_recorded_digest() -> None:
    """The provenance is checkable, so the fixture cannot drift silently.

    This is the test that makes the whole declaration trustworthy: the numbers
    below are only a measurement if the bytes they were measured from are the
    bytes the release published.
    """

    checked = 0
    for release in _provenance()["releases"]:
        for name, row in release["files"].items():
            raw = (FIXTURES / name).read_bytes()
            assert len(raw) == row["size_bytes"], name
            assert "sha256:" + hashlib.sha256(raw).hexdigest() == row["digest"], name
            checked += 1
    assert checked == 7


def test_the_fixture_holds_configs_and_no_weights() -> None:
    """A config cache that grew a tensor would be a weights mirror.

    The whole justification for fetching anything was that a config is 542
    bytes. Bounding the directory is what keeps that true.
    """

    files = sorted(path.name for path in FIXTURES.iterdir())
    assert all(name.endswith(".json") for name in files), files
    assert all((FIXTURES / name).stat().st_size < 8192 for name in files)


def test_the_declared_architecture_is_the_published_config() -> None:
    """Every number in ``TRANSFORMER``, against the config it came from.

    Field by field rather than as one dict comparison, because the declaration
    legitimately restates two of them through derived constants
    (``in_channels`` as ``PACKED_CHANNELS``, ``joint_attention_dim`` as
    ``JOINT_DIM``) and a dict equality would hide which one moved.
    """

    published = _fixture("turbo.transformer.config.json")
    for field, value in TRANSFORMER.items():
        expected = published[field]
        if isinstance(expected, list):
            expected = tuple(expected)
        assert value == expected, field
    # And nothing architectural was DROPPED: the only published keys the
    # declaration omits are diffusers bookkeeping and a null.
    omitted = set(published) - set(TRANSFORMER)
    assert omitted == {"_class_name", "_diffusers_version", "_name_or_path", "out_channels"}
    assert published["out_channels"] is None


def test_the_joint_dimension_is_three_stacked_qwen3_layers() -> None:
    """12288 is DERIVED from the text encoder, and both halves are the fixture's.

    Reading only the final Qwen3 layer would produce a tensor of the right rank
    and the wrong meaning — the failure that type-checks — so the width and the
    layer count are asserted against the published text_encoder config.
    """

    text = _fixture("text_encoder.config.json")
    assert TEXT_ENCODER["hidden_size"] == text["hidden_size"] == 4096
    assert TEXT_ENCODER["num_hidden_layers"] == text["num_hidden_layers"] == 36
    assert len(kl.TEXT_LAYERS) == 3
    assert max(kl.TEXT_LAYERS) < text["num_hidden_layers"]
    assert kl9.JOINT_DIM == 3 * text["hidden_size"] == 12288
    assert TRANSFORMER["joint_attention_dim"] == kl9.JOINT_DIM


def test_the_packed_channel_count_is_the_vaes_own() -> None:
    """128 = 32 latent channels x a 2x2 patch, both read from the VAE config."""

    vae = _fixture("vae.config.json")
    assert vae["latent_channels"] == kl.LATENT_CHANNELS == 32
    assert tuple(vae["patch_size"]) == (kl.PATCH, kl.PATCH) == (2, 2)
    assert kl.PACKED_CHANNELS == 128 == TRANSFORMER["in_channels"]


def test_the_scheduler_block_is_the_checkpoints_own() -> None:
    """Declared, so it rides the export digest rather than the math.

    9B and 4B ship BYTE-IDENTICAL scheduler configs (same blob digest), which
    is why the 9B declaration imports 4B's block instead of restating it.
    """

    published = _fixture("scheduler.scheduler_config.json")
    # A set of ONE, keyed by sampler (pgw#1346 K10).
    assert list(FLUX2_KLEIN_9B.schedulers) == ["flow_match_euler"]
    scheduler = FLUX2_KLEIN_9B.schedulers["flow_match_euler"]
    assert scheduler.name == "flow_match_euler_discrete"
    for field, value in scheduler.parameters.items():
        assert value == published[field], field
    assert published["_class_name"] == "FlowMatchEulerDiscreteScheduler"


# -------------------------------------------------------- 9B is not 4B, and why


def test_nine_b_is_a_different_architecture_from_four_b() -> None:
    """The measurement B1 could not make, made — and it refutes one model.

    An instance carries only ref, tuned, backing and label
    (``model/runtime.py::_materialize``), so a differing architecture config is
    a different ``ModelSpec`` by construction rather than by taste.
    """

    differs = {
        field
        for field in set(TRANSFORMER) | set(TRANSFORMER_4B)
        if TRANSFORMER.get(field) != TRANSFORMER_4B.get(field)
    }
    assert differs == {
        "num_layers",
        "num_single_layers",
        "num_attention_heads",
        "joint_attention_dim",
    }
    assert (TRANSFORMER["num_layers"], TRANSFORMER_4B["num_layers"]) == (8, 5)
    assert (TRANSFORMER["num_single_layers"], TRANSFORMER_4B["num_single_layers"]) == (24, 20)
    assert (TRANSFORMER["num_attention_heads"], TRANSFORMER_4B["num_attention_heads"]) == (32, 24)
    assert (TRANSFORMER["joint_attention_dim"], TRANSFORMER_4B["joint_attention_dim"]) == (12288, 7680)
    assert FLUX2_KLEIN_9B.name != "flux2_klein_4b"


def test_base_and_turbo_share_one_architecture_and_differ_only_in_provenance() -> None:
    """The OTHER half of the split: two checkpoints, ONE 9B model.

    ``_name_or_path`` is the directory the release was converted from. It is
    not architecture, and a declaration keyed on it would be keyed on a build
    machine's filesystem.
    """

    turbo = _fixture("turbo.transformer.config.json")
    base = _fixture("base.transformer.config.json")
    assert set(turbo) - set(base) == set()
    differs = {field for field in turbo if turbo[field] != base.get(field)}
    assert differs == {"_name_or_path"}
    # And the DISTILLATION is the thing that actually differs, which is a
    # `tuned` fact: it rides model_index.json, not the architecture.
    assert _fixture("turbo.model_index.json")["is_distilled"] is True
    assert "is_distilled" not in _fixture("base.model_index.json")


def test_base_and_turbo_are_two_instances_of_one_model() -> None:
    """Two tuned values, one set of graph classes — asserted, not asserted-about."""

    base = Flux2Klein9b.fake(
        tuned=_tuned(steps=ENDPOINT_NEUTRAL_STEPS, guidance=ENDPOINT_NEUTRAL_GUIDANCE)
    )
    turbo = Flux2Klein9b.fake(
        tuned=_tuned(
            steps=ENDPOINT_TURBO_STEPS, guidance=ENDPOINT_TURBO_GUIDANCE, distilled=True
        )
    )
    assert type(base) is type(turbo) is Flux2Klein9b
    assert base.tuned != turbo.tuned
    base_tuned, turbo_tuned = base.tuned, turbo.tuned
    assert isinstance(base_tuned, kl9.Flux2Klein9bTuned)
    assert isinstance(turbo_tuned, kl9.Flux2Klein9bTuned)
    assert base_tuned.steps == 28 and base_tuned.guidance == 4.0
    assert turbo_tuned.steps == 4 and turbo_tuned.distilled is True
    for tokens in kl9.packed_tokens(1024, 1024), kl9.packed_tokens(2048, 2048):
        assert (
            base.variant("denoiser", {"tokens": tokens}).layout
            == turbo.variant("denoiser", {"tokens": tokens}).layout
        )


# --------------------------------------------------------------- the bucket axis


def test_the_declared_buckets_are_the_endpoint_presets_token_counts() -> None:
    """Fifteen preset sizes, nine token coordinates — recomputed, not trusted."""

    derived = tuple(sorted({kl9.packed_tokens(w, h) for w, h in ENDPOINT_PRESETS}))
    axis = FLUX2_KLEIN_9B.axis_values["tokens"]
    assert derived == axis
    assert len(ENDPOINT_PRESETS) == 15 and len(axis) == 9
    # The nine coordinates the endpoint's own mint declaration records.
    assert axis == (4056, 4070, 4089, 4096, 4116, 7744, 8160, 14400, 16384)


def test_orientation_pairs_share_a_token_count_but_not_a_latent_grid() -> None:
    """Why the VAE decode cannot be keyed by this axis, at the second width too."""

    assert kl9.packed_tokens(1184, 880) == kl9.packed_tokens(880, 1184)
    assert kl.latent_grid(1184, 880) != kl.latent_grid(880, 1184)


# ------------------------------------------------------------ the ie#740 floors


def test_the_ie740_serving_floors_are_preserved_by_value() -> None:
    """K1: a production floor migrates as a NUMBER, never as a re-typed string.

    44 GB is 9B's own envelope and is NOT 4B's 30 — losing the difference
    silently would place a 9B serve on a card that cannot hold it.
    """

    assert FLUX2_KLEIN_9B.layouts == {"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")}
    requirements = FLUX2_KLEIN_9B.layout_requirements
    assert set(requirements) == {"cozy.fp8-rowwise@1", "plain.bf16@1"}
    assert requirements["cozy.fp8-rowwise@1"].minimum.min_sm == 89
    assert requirements["plain.bf16@1"].minimum.min_vram_gb == 44.0
    # Each lane declares ONE of the two terms, and the other stays unset — a
    # floor invented on the axis a lane did not name is still an invented floor.
    assert requirements["cozy.fp8-rowwise@1"].minimum.min_vram_gb == 0.0
    assert requirements["plain.bf16@1"].minimum.min_sm == 0


def test_the_weight_lane_is_not_a_traced_graph_variant() -> None:
    """The fp8 lane is a WEIGHT contract; the traced classes are bf16 only.

    Restated at this width because the two axes are easy to conflate: the model
    DEMANDS two layout contracts of its bytes, and declares graph variants for
    one compute dtype.
    """

    layouts = {layout for _, _, layout in FLUX2_KLEIN_9B.variants()}
    assert layouts == {"bf16"}
    assert len(FLUX2_KLEIN_9B.variants()) == 9


# ------------------------------------------------------------------- the loop


def test_a_fake_backed_klein_runs_the_loop_through_the_typed_callable() -> None:
    """Hubless, cardless, weightless — and the real code path."""

    instance = Flux2Klein9b.fake(tuned=_tuned(steps=4))
    width, height = 1024, 1024
    tokens = kl9.packed_tokens(width, height)
    seen: list[int] = []
    latents = kl9.generate(
        instance,
        tokens=tokens,  # type: ignore[arg-type]
        width=width,
        height=height,
        prompt_embeds=torch.zeros(1, kl.TEXT_TOKENS, kl9.JOINT_DIM),
        negative_embeds=None,
        steps=4,
        guidance=1.0,
        seed=7,
        on_step=lambda index, total: seen.append(index),
    )
    rows, cols = kl.latent_grid(width, height)
    assert latents.shape == (1, kl.PACKED_CHANNELS, rows, cols)
    assert seen == [0, 1, 2, 3]


def test_cfg_is_a_call_count_not_a_batch_axis() -> None:
    """A negative branch is a SECOND sequential forward, never a doubled batch.

    Klein's transformer takes no guidance embedding, so guidance here is
    classifier-free — which is why every declared class is B=1.
    """

    instance = Flux2Klein9b.fake(tuned=_tuned())
    width, height = 1024, 1024
    tokens = kl9.packed_tokens(width, height)
    embeds = torch.zeros(1, kl.TEXT_TOKENS, kl9.JOINT_DIM)
    schedule = kl9.schedule_for(instance, steps=3, width=width, height=height)
    steps = list(
        kl9.denoise(
            instance,
            tokens=tokens,  # type: ignore[arg-type]
            width=width,
            height=height,
            latents=kl.initial_latents(
                width=width, height=height, batch=1, seed=0,
                device=embeds.device, dtype=embeds.dtype,
            ),
            prompt_embeds=embeds,
            negative_embeds=embeds,
            schedule=schedule,
            guidance=4.0,
        )
    )
    assert [index for index, _ in steps] == [0, 1, 2]
    assert all(latents.shape[0] == 1 for _, latents in steps)
    assert TRANSFORMER["guidance_embeds"] is False


def test_the_declaration_exposes_one_runner_and_one_loop_stage() -> None:
    """The composition, stated: one denoiser, repeated `steps` times."""

    assert tuple(runner.name for runner in FLUX2_KLEIN_9B.runners) == ("denoiser",)
    loop = FLUX2_KLEIN_9B.staged_loop
    assert tuple((stage.runner, stage.repeat) for stage in loop.stages) == (
        ("denoiser", "steps"),
    )
    assert tuple(p.name for p in FLUX2_KLEIN_9B.parameters) == ("steps",)


def test_the_tuned_schema_is_its_own_and_carries_the_endpoints_values() -> None:
    """A tuned schema is published under the MODEL's name, so it cannot be shared.

    And the neutral values are the endpoint's retired `Flux2Klein9bDefaults`,
    migrated by value rather than re-chosen.
    """

    assert FLUX2_KLEIN_9B.tuned is kl9.Flux2Klein9bTuned
    # A tuned schema is published under the MODEL's own name, so 4B's cannot
    # be reused here even though the two structs are field-identical. mypy
    # already proves the classes are unrelated; this asserts the REGISTRATION
    # consequence, which is the part that would actually break the hub.
    assert FLUX2_KLEIN_9B.tuned is not FLUX2_KLEIN_4B.tuned
    assert FLUX2_KLEIN_9B.name != FLUX2_KLEIN_4B.name
    neutral = kl9.Flux2Klein9bTuned()
    assert neutral.steps == ENDPOINT_NEUTRAL_STEPS
    assert neutral.guidance == ENDPOINT_NEUTRAL_GUIDANCE
    assert neutral.distilled is False
    assert FLUX2_KLEIN_9B.lora_tuned is kl9.Flux2Klein9bLoraTuned
