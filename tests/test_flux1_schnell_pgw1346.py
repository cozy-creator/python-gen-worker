"""pgw#1346 B1 — FLUX.1-schnell is declared, and every number has a SOURCE.

B1 recorded schnell as unauthorable: its architecture and schedule live in the
checkpoint, the endpoints deliberately carry no checkpoint ref (ie#524/th#980),
and guessing a scheduler block silently re-keys a family because that block
rides the export digest. Resolved by caching the serving release's own
published configs through the hub — the route the endpoints already resolve
against — under ``tests/fixtures/flux1_schnell/``.

So the claims here are mostly one claim, made specific: **the declaration is
the fixture.**

1. The fixture is what it says it is (re-hashed against its provenance).
2. Every architecture number in the declaration equals the published one.
3. schnell differs from dev in EXACTLY ONE architecture field, which is what
   makes the "import dev's blocks" sharing legitimate rather than convenient.
4. That one field is a GRAPH difference — it removes an input from the traced
   call — which is what makes schnell a model and not an instance of dev.
5. The schedule is the release's own: static shift, where dev's is dynamic.
6. The loop runs, hubless and cardless, through the real typed callables.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from gen_worker.model.catalog import Flux1Schnell
from gen_worker.model.catalog import flux1_schnell_serve as sc
from gen_worker.model.catalog.flux1_dev import (
    CLIP_TEXT,
    T5_TEXT,
    TRANSFORMER as DEV_TRANSFORMER,
    VAE,
)
from gen_worker.model.catalog.flux1_schnell import (
    FLUX1_SCHNELL,
    SCHEDULER,
    TOKEN_BUCKETS,
    TRANSFORMER,
)

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "flux1_schnell"

#: The endpoint's own preset grid (flux.1-schnell/src/flux1_schnell/main.py),
#: transcribed HERE so the buckets are derived from the product decision rather
#: than from the declaration under test.
ENDPOINT_PRESETS: tuple[tuple[int, int], ...] = (
    (1024, 1024), (1152, 864), (864, 1152), (1248, 832),
    (832, 1248), (1344, 768), (768, 1344),
)


def _fetcher() -> Any:
    """The shared authoring utility, imported the way the fences are."""

    spec = importlib.util.spec_from_file_location(
        "_pgw1346_fetch", REPO / "scripts" / "fetch_model_configs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _published(name: str) -> Any:
    return json.loads((FIXTURES / f"schnell.{name}.json").read_text())


# ------------------------------------------------------------------ the source


def test_every_cached_config_matches_its_recorded_digest() -> None:
    """The provenance is checkable, so the fixture cannot drift silently.

    Uses the SHARED verifier rather than a local re-implementation: one
    definition of "the fixture is what it says it is", so a second family
    cannot land a weaker version of this check.
    """

    checked, problems = _fetcher().verify_fixture(FIXTURES)
    assert problems == []
    assert checked == 6


def test_red_the_verifier_catches_a_tampered_fixture(tmp_path: Path) -> None:
    """The check can go RED, which is the only thing that makes it a check.

    A verifier nobody has watched fail is a verifier nobody knows is wired up.
    """

    fetch = _fetcher()
    for source in FIXTURES.iterdir():
        (tmp_path / source.name).write_bytes(source.read_bytes())
    assert fetch.verify_fixture(tmp_path)[1] == []

    target = tmp_path / "schnell.scheduler.scheduler_config.json"
    document = json.loads(target.read_text())
    document["shift"] = 3.0  # dev's value, silently substituted
    target.write_text(json.dumps(document))

    _, problems = fetch.verify_fixture(tmp_path)
    assert any("hashes to" in problem for problem in problems)


def test_red_the_fetcher_refuses_a_weight_bearing_path() -> None:
    """The weights fence is by NAME, so a plausible .json cannot slip through."""

    fetch = _fetcher()
    for path in (
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "text_encoder_2/model.safetensors.index.json",
        "transformer/diffusion_pytorch_model.safetensors",
    ):
        with pytest.raises(fetch.FetchRefused):
            fetch._check_path(path)
    # ...and the ones a declaration actually needs are allowed.
    fetch._check_path("transformer/config.json")
    fetch._check_path("scheduler/scheduler_config.json")


def test_red_an_unvouched_file_in_the_fixture_is_refused(tmp_path: Path) -> None:
    """A file no provenance row covers is how a weight would arrive."""

    fetch = _fetcher()
    for source in FIXTURES.iterdir():
        (tmp_path / source.name).write_bytes(source.read_bytes())
    (tmp_path / "sneaked.json").write_text("{}")
    _, problems = fetch.verify_fixture(tmp_path)
    assert any("sneaked.json" in problem for problem in problems)


def test_the_fixture_holds_configs_and_no_weights() -> None:
    """The weights-locality rule, asserted rather than promised."""

    files = sorted(p.name for p in FIXTURES.iterdir() if p.is_file())
    assert all(name.endswith(".json") for name in files)
    total = sum((FIXTURES / name).stat().st_size for name in files)
    assert total < 64 * 1024, f"{total} bytes is too much to be configs"
    provenance = json.loads((FIXTURES / "PROVENANCE.json").read_text())
    assert provenance["releases"][0]["repo"] == "tensorhub/flux1-schnell"
    assert provenance["releases"][0]["checkpoint_id"].startswith("sha256:")


# ------------------------------------------------------------ the architecture


def test_the_declared_architecture_is_the_published_one() -> None:
    """Every field the release publishes, equal to what the catalog declares."""

    published = _published("transformer.config")
    for field, value in published.items():
        if field.startswith("_"):
            continue
        assert TRANSFORMER[field] == value, field


def test_schnell_differs_from_dev_in_exactly_one_architecture_field() -> None:
    """The measurement the whole sharing design rests on.

    If this ever grows a second entry, importing dev's block is no longer
    honest and schnell owes its own — so this test is the tripwire on that.
    """

    published = _published("transformer.config")
    differing = sorted(
        field
        for field, value in published.items()
        if not field.startswith("_") and DEV_TRANSFORMER.get(field) != value
    )
    assert differing == ["guidance_embeds"]
    assert DEV_TRANSFORMER["guidance_embeds"] is True
    assert TRANSFORMER["guidance_embeds"] is False


def test_the_shared_component_blocks_really_are_shared() -> None:
    """dev's VAE, CLIP and T5 blocks describe schnell's components too.

    Compared as tuples because JSON has no tuple: a list/tuple mismatch is a
    representation difference and asserting on it would be asserting about
    ``json.loads``, not about the checkpoint.
    """

    def same(declared: Any, published: Any) -> bool:
        if isinstance(declared, (tuple, list)) and isinstance(published, (tuple, list)):
            return tuple(declared) == tuple(published)
        return bool(declared == published)

    vae = _published("vae.config")
    for field, value in VAE.items():
        if field in vae:
            assert same(value, vae[field]), f"vae.{field}"

    clip = _published("text_encoder.config")
    for field, value in CLIP_TEXT.items():
        if field in clip:
            assert same(value, clip[field]), f"clip.{field}"

    # T5 is shared too, but three fields are DELIBERATE serving pins rather
    # than architecture: an encoder-only, cacheless, dropout-free eval config.
    # They are stated here so the exception is recorded, not silently skipped.
    serving_pins = {"dropout_rate", "is_encoder_decoder", "use_cache"}
    t5 = _published("text_encoder_2.config")
    for field, value in T5_TEXT.items():
        if field in t5 and field not in serving_pins:
            assert same(value, t5[field]), f"t5.{field}"
    assert T5_TEXT["use_cache"] is False and T5_TEXT["is_encoder_decoder"] is False


def test_the_component_tree_is_the_published_pipeline() -> None:
    """Four components, and the declaration covers the three it runs."""

    index = _published("model_index")
    assert index["_class_name"] == "FluxPipeline"
    assert index["transformer"][1] == "FluxTransformer2DModel"
    assert index["text_encoder"][1] == "CLIPTextModel"
    assert index["text_encoder_2"][1] == "T5EncoderModel"
    assert index["vae"][1] == "AutoencoderKL"


# --------------------------------------------------------------- the scheduler


def test_the_scheduler_block_is_the_releases_own_and_is_static() -> None:
    """The block rides the export digest, so it is asserted field by field.

    Static shift is schnell's one scheduler difference from dev, and it is the
    reason B1 owed no scheduler math: ``flow_match_euler_discrete`` already
    implements both arms.
    """

    published = _published("scheduler.scheduler_config")
    for field, value in published.items():
        if field.startswith("_"):
            continue
        assert SCHEDULER[field] == value, field
    assert SCHEDULER["use_dynamic_shifting"] is False
    assert SCHEDULER["shift"] == 1.0
    assert FLUX1_SCHNELL.scheduler is not None
    assert FLUX1_SCHNELL.scheduler.name == "flow_match_euler_discrete"


# ----------------------------------------------------------------- the buckets


def test_the_declared_buckets_are_the_endpoint_presets_token_counts() -> None:
    """Seven preset sizes, four token coordinates — recomputed, not trusted."""

    derived = tuple(sorted({sc.packed_tokens(w, h) for w, h in ENDPOINT_PRESETS}))
    assert derived == TOKEN_BUCKETS
    assert len(ENDPOINT_PRESETS) == 7 and len(TOKEN_BUCKETS) == 4


def test_transposed_presets_collapse_onto_one_graph_class() -> None:
    """ie#685: FLUX.1's rope arrives as tensors, so the graph keys on COUNT.

    And the same fact is why the VAE decode is NOT a declared runner: the two
    orientations need different decoder output shapes.
    """

    assert sc.packed_tokens(1152, 864) == sc.packed_tokens(864, 1152)
    assert sc.latent_grid(1152, 864) != sc.latent_grid(864, 1152)


def test_a_square_preset_is_the_familiar_anchor() -> None:
    assert sc.TOKEN_STRIDE == 16
    assert sc.packed_tokens(1024, 1024) == 4096


# ------------------------------------------------------------------ the floors


def test_the_ie740_serving_floor_is_preserved_by_value() -> None:
    """K1, and the ABSENCE of an fp8 lane is preserved too.

    Schnell's endpoint offers only bf16, so inventing an sm floor here would
    decline cards that serve it today.
    """

    requirements = FLUX1_SCHNELL.layout_requirements
    assert requirements["plain.bf16@1"].minimum.min_vram_gb == 36.0
    assert requirements["plain.bf16@1"].minimum.min_sm == 0
    layouts = FLUX1_SCHNELL.layouts
    assert layouts is not None
    assert tuple(layouts["*"]) == ("plain.bf16@1",)


# -------------------------------------------------------------------- the loop


def test_the_denoiser_call_carries_no_guidance_input() -> None:
    """The graph difference that makes schnell a MODEL, asserted on the export.

    dev's denoiser takes seven parameters; schnell's takes six, and the missing
    one is ``guidance``. This is the whole class/instance argument in one
    assertion.
    """

    document = json.loads(
        (
            REPO / "src" / "gen_worker" / "model" / "catalog" / "_generated"
            / "flux1_schnell.export.json"
        ).read_text()
    )
    denoiser = next(r for r in document["runners"] if r["name"] == "denoiser")
    for variant in denoiser["variants"]:
        params = [entry["param"] for entry in variant["ingress"]["inputs"]]
        assert "guidance" not in params
        assert params == [
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "img_ids",
            "txt_ids",
        ]
        # Six, where dev's is seven. The arity IS the difference.
        assert variant["ingress"]["flat_arity"] == 6


def test_a_fake_backed_schnell_runs_the_loop_through_the_typed_callables() -> None:
    """The whole loop, hubless and cardless, through the real code path."""

    instance = Flux1Schnell.fake()
    seen: list[tuple[int, int]] = []
    latents = sc.generate(
        instance,
        tokens=4096,
        width=1024,
        height=1024,
        clip_ids=sc.clip_token_ids([1, 2, 3], device="cpu"),
        t5_ids=sc.t5_token_ids([4, 5, 6], device="cpu"),
        steps=4,
        seed=7,
        on_step=lambda index, total: seen.append((index, total)),
    )
    assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]
    assert tuple(latents.shape) == (1, sc.LATENT_CHANNELS, 128, 128)


def test_the_text_pin_is_schnells_own_256() -> None:
    """th#1126: dev's 512 bought a 512-token encode for nothing."""

    assert sc.TEXT_TOKENS == 256
    ids = sc.t5_token_ids([1, 2, 3], device="cpu")
    assert tuple(ids.shape) == (1, 256)


def test_the_seed_reaches_the_latents() -> None:
    one = sc.initial_latents(
        width=1024, height=1024, batch=1, seed=1, device="cpu", dtype=torch.float32
    )
    two = sc.initial_latents(
        width=1024, height=1024, batch=1, seed=2, device="cpu", dtype=torch.float32
    )
    again = sc.initial_latents(
        width=1024, height=1024, batch=1, seed=1, device="cpu", dtype=torch.float32
    )
    assert not torch.equal(one, two)
    assert torch.equal(one, again)


def test_packing_round_trips_at_a_rectangular_size() -> None:
    """Non-square deliberately: schnell's grid is where dev's helpers stop."""

    rows, cols = sc.latent_grid(1152, 864)
    latents = torch.randn(1, sc.LATENT_CHANNELS, rows * sc.PATCH, cols * sc.PATCH)
    packed = sc.pack_latents(latents)
    assert tuple(packed.shape) == (1, rows * cols, sc.LATENT_CHANNELS * 4)
    assert torch.equal(sc.unpack_latents(packed, rows=rows, cols=cols), latents)


def test_the_declaration_runs_three_stages_and_no_decoder() -> None:
    """The honest shape, and the endpoint's own (`targets=("transformer",)`)."""

    assert tuple(r.name for r in FLUX1_SCHNELL.runners) == ("clip", "denoiser", "t5")
    assert FLUX1_SCHNELL.loop is not None
    assert tuple(s.runner for s in FLUX1_SCHNELL.loop.stages) == (
        "clip", "t5", "denoiser",
    )


def test_the_step_ceiling_is_the_distillation_contract() -> None:
    """1-4 steps (ie#462), declared as a parameter bound rather than a comment."""

    steps = next(p for p in FLUX1_SCHNELL.parameters if p.name == "steps")
    assert (steps.minimum, steps.maximum) == (1, 4)
    assert sc.Flux1SchnellTuned().steps == 4
