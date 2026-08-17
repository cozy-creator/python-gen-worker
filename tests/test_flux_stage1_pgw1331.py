"""pgw#1331 — Flux serves end to end with no model library on the request path.

Four claims, and each is tested against the thing it is a claim ABOUT:

1. **The bare math is the same math.** The scheduler and the packing arithmetic
   are compared to diffusers' own, numerically, not by reading them. A
   re-implementation nobody differenced is a rewrite with a bug in it.
2. **The whole composition is graph classes.** All four runners export, and the
   mint bridge turns each into a torchcg class declaration whose ingress digest
   is the COMMITTED export's — which is what makes the typed binding and the
   artifact one contract seen from two sides.
3. **The request path holds no model library.** Statically, by the fence;
   at run time, in a subprocess that installs the blocker, imports the surface
   and generates an image — because an in-process check can never observe the
   property (the test interpreter has already imported everything).
4. **The gaps are shaped like refusals.** A family whose scheduler has no
   implementation gets no ``scheduler()`` method; a mint whose digest drifts is
   refused; an unlisted guarded import is refused.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gen_worker.family import mint as family_mint
from gen_worker.family.catalog import Flux1Dev, Sdxl
from gen_worker.family.catalog import flux1_dev_serve as fx
from gen_worker.family.catalog.flux1_dev import FLUX1_DEV, SCHEDULER
from gen_worker.family.errors import FamilyError, FamilyRefusal
from gen_worker.family.scheduler import (
    IMPLEMENTED,
    FlowMatchEulerDiscrete,
    Schedule,
    SchedulerKind,
    parse_kind,
)
from gen_worker.serve import role

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]


def _fence() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_pgw1331_fence", REPO / "scripts" / "lint_serve_role_closure.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- the bare math


@pytest.mark.parametrize("steps", [1, 4, 20, 28, 50])
@pytest.mark.parametrize("resolution", [768, 1024])
def test_the_flow_match_schedule_matches_diffusers_to_one_float32_ulp(steps: int, resolution: int) -> None:
    """Our closed form and diffusers' object produce the SAME sigma ladder.

    This is the evidence for "scheduler as bare typed math" being a MOVE rather
    than a rewrite. If it ever stops holding, every image this family serves has
    silently changed, and no other test in the repo would notice.
    """

    numpy = pytest.importorskip("numpy")
    diffusers = pytest.importorskip("diffusers")
    flux = pytest.importorskip("diffusers.pipelines.flux.pipeline_flux")

    sequence = fx.packed_tokens(resolution)
    theirs = diffusers.FlowMatchEulerDiscreteScheduler(**dict(SCHEDULER))
    mu = flux.calculate_shift(
        sequence,
        theirs.config.base_image_seq_len,
        theirs.config.max_image_seq_len,
        theirs.config.base_shift,
        theirs.config.max_shift,
    )
    theirs.set_timesteps(sigmas=numpy.linspace(1.0, 1 / steps, steps), mu=mu)

    ours = FlowMatchEulerDiscrete.from_block(SCHEDULER).schedule(
        steps, image_seq_len=sequence
    )

    assert FlowMatchEulerDiscrete.from_block(SCHEDULER).mu(sequence) == pytest.approx(mu)
    assert len(ours) == steps
    # The tolerance is ONE float32 ULP, and it is that because of a measured
    # difference rather than a guessed one: diffusers rounds its shifted ladder
    # through float32 tensors while this stays in float64 until a caller casts,
    # so at most one entry per ladder lands a single ULP apart (measured worst
    # case 6e-8 at sigma 0.513, which IS one ULP there). Every other entry is
    # exact. A looser tolerance would hide a real formula difference; a tighter
    # one would be asserting that two float widths agree, which they cannot.
    assert list(ours.sigmas) == pytest.approx(
        [float(value) for value in theirs.sigmas], rel=2e-7, abs=1e-9
    )
    assert list(ours.timesteps) == pytest.approx(
        [float(value) for value in theirs.timesteps], rel=2e-7, abs=1e-6
    )
    assert numpy.float32(ours.sigmas[0]) == numpy.float32(1.0)
    assert ours.sigmas[-1] == 0.0


def test_the_euler_step_is_one_line_of_arithmetic() -> None:
    """``x + (sigma_next - sigma) * v``, and the terminal sigma lands on zero."""

    schedule = Schedule(sigmas=(1.0, 0.5, 0.0), num_train_timesteps=1000)
    sample = torch.ones(2, 3)
    velocity = torch.full((2, 3), 4.0)
    assert torch.equal(schedule.step(0, velocity, sample), sample + (0.5 - 1.0) * velocity)
    assert torch.equal(schedule.step(1, velocity, sample), sample + (0.0 - 0.5) * velocity)
    with pytest.raises(FamilyError) as caught:
        schedule.step(2, velocity, sample)
    assert caught.value.reason is FamilyRefusal.SCHEDULER_INVALID


@pytest.mark.parametrize("resolution", [768, 1024])
def test_the_packing_arithmetic_is_bitwise_the_pipeline_s(resolution: int) -> None:
    """Pack, unpack and the rope ids all equal ``FluxPipeline``'s, exactly."""

    pipeline = pytest.importorskip("diffusers").FluxPipeline
    edge = fx.latent_edge(resolution)
    latents = torch.randn(2, fx.LATENT_CHANNELS, edge, edge)

    packed = fx.pack_latents(latents)
    assert torch.equal(packed, pipeline._pack_latents(latents, 2, fx.LATENT_CHANNELS, edge, edge))
    assert packed.shape[1] == fx.packed_tokens(resolution)
    assert torch.equal(fx.unpack_latents(packed, edge=edge), latents)
    assert torch.equal(
        fx.latent_image_ids(edge, device="cpu", dtype=torch.float32),
        pipeline._prepare_latent_image_ids(None, edge // 2, edge // 2, "cpu", torch.float32),
    )


def test_a_schedule_terminating_anywhere_but_zero_is_refused() -> None:
    """The terminal zero is part of the ladder, not a special case in ``step``."""

    with pytest.raises(FamilyError) as caught:
        Schedule(sigmas=(1.0, 0.5), num_train_timesteps=1000)
    assert "terminate at sigma 0.0" in str(caught.value)


def test_dynamic_shifting_refuses_to_guess_a_sequence_length() -> None:
    """Defaulting it would serve every resolution one resolution's schedule."""

    scheduler = FlowMatchEulerDiscrete.from_block(SCHEDULER)
    assert scheduler.use_dynamic_shifting
    with pytest.raises(FamilyError) as caught:
        scheduler.schedule(4)
    assert "image_seq_len" in str(caught.value)


@pytest.mark.parametrize(
    "block, wanted",
    [
        ({"use_dynamic_shifting": 1}, "a boolean"),
        ({"num_train_timesteps": 1.5}, "an integer"),
        ({"base_shift": "0.5"}, "a real number"),
    ],
)
def test_a_scheduler_block_is_parsed_not_coerced(block: dict[str, Any], wanted: str) -> None:
    """``use_dynamic_shifting: 1`` means something the author did not write."""

    with pytest.raises(FamilyError) as caught:
        FlowMatchEulerDiscrete.from_block(block)
    assert wanted in str(caught.value)


def test_an_unimplemented_scheduler_is_an_absent_method_not_a_fallback() -> None:
    """SDXL declares ``euler_discrete``; the SDK implements no math for it.

    The gap's SHAPE is the point: ``Sdxl`` has no ``scheduler()``, so an author
    who wants one is told by their own type checker. A base-class fallback that
    returned "some scheduler" would put a model library back on the path at the
    one moment nobody is looking.
    """

    assert Flux1Dev.SCHEDULER is SchedulerKind.FLOW_MATCH_EULER_DISCRETE
    assert Sdxl.SCHEDULER is SchedulerKind.EULER_DISCRETE
    assert isinstance(Flux1Dev.fake().scheduler(), FlowMatchEulerDiscrete)
    assert not hasattr(Sdxl, "scheduler")
    assert set(IMPLEMENTED) == {SchedulerKind.FLOW_MATCH_EULER_DISCRETE}
    with pytest.raises(FamilyError):
        parse_kind("ddim")


def test_the_declared_block_is_the_scheduler_s_only_source_of_constants() -> None:
    """A constant hardcoded in the math would be a second declaration of it."""

    scheduler = Flux1Dev.fake().scheduler()
    assert scheduler.base_shift == SCHEDULER["base_shift"]
    assert scheduler.max_shift == SCHEDULER["max_shift"]
    assert scheduler.max_image_seq_len == SCHEDULER["max_image_seq_len"]
    # …and the block rides the export digest, so re-declaring it re-identifies
    # the family rather than silently changing every request.
    assert dict(Flux1Dev.SCHEDULER_PARAMETERS) == dict(SCHEDULER)


# ------------------------------------------------------- the whole composition


def test_the_family_declares_every_stage_of_the_pipeline() -> None:
    """Two text encoders, the denoiser and the decoder — nothing left eager."""

    assert [runner.name for runner in FLUX1_DEV.runners] == [
        "clip",
        "decoder",
        "denoiser",
        "t5",
    ]
    assert [row[0] for row in Flux1Dev.LOOP] == ["clip", "t5", "denoiser", "decoder"]
    # The text encoders bucket on nothing: their token lengths are pinned by the
    # architecture and by the family, so a bucket would generate classes nothing
    # selects.
    assert FLUX1_DEV.runner("clip").axes == ()
    assert FLUX1_DEV.runner("t5").axes == ()
    assert FLUX1_DEV.runner("denoiser").axes == ("resolution",)


def test_every_declared_variant_carries_a_committed_ingress() -> None:
    """The binding a handler compiles against covers all six classes."""

    rows = family_mint.variants_of(FLUX1_DEV)
    assert len(rows) == 6
    for runner, bucket, layout in rows:
        variant = Flux1Dev.EXPORT.runner(runner.name).variant(bucket, layout)
        assert str(variant.ingress_digest)
        assert variant.outputs


def test_a_fake_backed_flux_generates_an_image_through_the_typed_callables() -> None:
    """The whole loop, hubless and cardless, through the real code path.

    Not a mock of the SDK — it IS the SDK, with the only part that needs a card
    replaced (greenfield B8). So this exercises the packing, the schedule, four
    typed callables and the ingress each of them resolves.
    """

    instance = Flux1Dev.fake()
    seen: list[tuple[int, int]] = []
    image = fx.generate(
        instance,
        resolution=1024,
        clip_ids=fx.clip_token_ids([1, 2, 3], device="cpu"),
        t5_ids=fx.t5_token_ids([4, 5, 6], device="cpu"),
        steps=3,
        guidance=3.5,
        seed=7,
        on_step=lambda index, total: seen.append((index, total)),
    )
    assert seen == [(0, 3), (1, 3), (2, 3)]
    assert tuple(image.shape) == (1, 3, 1024, 1024)
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
    # Deterministic: a fake backing is a function of the declaration and the
    # seed, so a receipt can assert on it.
    again = fx.generate(
        instance,
        resolution=1024,
        clip_ids=fx.clip_token_ids([1, 2, 3], device="cpu"),
        t5_ids=fx.t5_token_ids([4, 5, 6], device="cpu"),
        steps=3,
        guidance=3.5,
        seed=7,
    )
    assert torch.equal(image, again)


def test_a_different_seed_is_a_different_starting_point() -> None:
    """The seed reaches the latents; it is not decorative.

    Asserted at the LATENTS and not at the image, and the reason is worth
    stating: a fake backing's outputs are a function of the declaration, not of
    the call, so the decoder returns the same tensor whatever it is handed. The
    seed's effect is real and observable exactly where it is applied. A test
    that asserted it at the image would be asserting a property the fake
    backing does not have, and would go green only once a real backing arrived.
    """

    one = fx.initial_latents(
        resolution=768, batch=1, seed=1, device="cpu", dtype=torch.float32
    )
    two = fx.initial_latents(
        resolution=768, batch=1, seed=2, device="cpu", dtype=torch.float32
    )
    again = fx.initial_latents(
        resolution=768, batch=1, seed=1, device="cpu", dtype=torch.float32
    )
    assert not torch.equal(one, two)
    assert torch.equal(one, again)
    assert tuple(one.shape) == (1, fx.packed_tokens(768), fx.LATENT_CHANNELS * 4)


# ------------------------------------------------------------- the mint bridge


def test_the_mint_bridge_declares_torchcg_classes_from_the_declaration_alone() -> None:
    """Trace -> keying block -> class hash, with no endpoint and no weights.

    Everything below ``Engine.compile`` runs on CPU under fake tensors, which is
    the whole reason a family can be minted without downloading a checkpoint:
    cell identity is checkpoint-free (§4.27) and the constants arrive at ARM
    time from the store (pgw#1329).
    """

    from gen_worker import aot_mint

    hashes: dict[str, str] = {}
    for traced, spec, row in family_mint.traced_classes(FLUX1_DEV, only=("decoder",)):
        declaration = aot_mint.tcg_graph_class_spec(row, spec).declare()
        hashes[row.name] = str(declaration.class_hash)
        assert spec.family == "flux1_dev"
        assert spec.target == traced.runner
        assert dict(spec.class_dims) == traced.bucket
        assert spec.specialization == {"layout": traced.layout}
        # THE drift property: the mint's ingress is the committed export's,
        # because both come from the same tracer (torchcg G16).
        committed = Flux1Dev.EXPORT.runner(traced.runner).variant(
            traced.bucket, traced.layout
        )
        assert str(committed.ingress_digest) == traced.ingress.digest()
        row.release()

    assert sorted(hashes) == ["decoder.resolution1024.bf16", "decoder.resolution768.bf16"]
    # Two buckets are two CLASSES: a runner armed at 1024 must not answer a 768
    # call, and the identity is what makes that structural rather than checked.
    assert len(set(hashes.values())) == 2


def test_a_mint_whose_digest_drifted_from_the_export_is_refused() -> None:
    """The export is the source; a mint that disagrees is the thing that is wrong."""

    row = family_mint.MintedVariant(
        runner="decoder",
        bucket=(("resolution", 1024),),
        layout="bf16",
        graph_class="decoder.resolution1024.bf16",
        key="cg-key-v1-" + "0" * 56,
        artifact=Path("/nowhere.tar.gz"),
        metadata={},
        ingress_digest="not-the-committed-one",
        compile_s=1.0,
        reuse_s=0.0,
    )
    with pytest.raises(FamilyError) as caught:
        family_mint.assert_matches_export([row], Flux1Dev.EXPORT)
    assert "never becomes the source" in str(caught.value)
    family_mint.assert_matches_export([], Flux1Dev.EXPORT)


def test_minting_an_eager_only_family_is_refused_rather_than_invented() -> None:
    from gen_worker.family.spec import Family, TunedValues

    class _Tuned(TunedValues, frozen=True):
        steps: int = 1

    eager = Family(name="pgw1331_eager_only", tuned=_Tuned)
    try:
        with pytest.raises(FamilyError) as caught:
            family_mint.mint_family(eager, out_dir=Path("/nowhere"), work=Path("/nowhere"))
        assert "eager-only" in str(caught.value)
    finally:
        from gen_worker.families import base as families_base

        families_base._REGISTRY.pop("pgw1331_eager_only", None)


def test_a_runner_the_family_does_not_declare_is_refused() -> None:
    with pytest.raises(FamilyError) as caught:
        family_mint.variants_of(FLUX1_DEV, only=("unet",))
    assert "declares no runner 'unet'" in str(caught.value)


def test_the_mint_bridge_is_declared_mint_machinery() -> None:
    """A serve-role module that could reach it would be a pod that can compile."""

    assert "gen_worker.family.mint" in role.MINT_MACHINERY


# ------------------------------------------------------------------- the fence


def test_the_model_free_surface_reaches_no_model_library() -> None:
    fence = _fence()
    roots = fence._declared_tuple("MODEL_FREE_MODULES")
    libraries = fence._declared_tuple("FORBIDDEN_LIBRARIES")
    optional = fence._declared_tuple("OPTIONAL_SERVE_IMPORTS")
    assert tuple(roots) == role.MODEL_FREE_MODULES
    assert tuple(libraries) == role.FORBIDDEN_LIBRARIES
    assert tuple(optional) == role.OPTIONAL_SERVE_IMPORTS
    assert fence.check_model_free(roots, libraries, optional, within=role.SERVE_ROLE_MODULES) == []
    assert fence.main([]) == 0


def test_red_the_fence_fires_on_a_declaration_that_names_a_model_library() -> None:
    """The declaration reaches diffusers only through FUNCTION-LOCAL imports.

    So this one root proves both halves: that the library walk fires, and that
    it still follows lazy imports — which is the shape the coupling takes.
    """

    fence = _fence()
    problems = fence.check_model_free(
        ("gen_worker.family.catalog.flux1_dev",), role.FORBIDDEN_LIBRARIES, ()
    )
    assert [line for line in problems if "diffusers" in line]
    assert [line for line in problems if "transformers" in line]


def test_red_an_unlisted_guarded_import_is_refused() -> None:
    """The hatch is a closed list, not a door any ``try: import`` walks through."""

    fence = _fence()
    problems = fence.check_model_free(
        ("gen_worker.family.catalog._generated.flux1_dev",), role.FORBIDDEN_LIBRARIES, ()
    )
    assert [line for line in problems if "guarded import" in line]


def test_red_a_listed_hatch_nobody_uses_is_refused() -> None:
    """An enumerated hatch nobody reaches is a hatch nobody is checking."""

    fence = _fence()
    problems = fence.check_model_free(
        role.MODEL_FREE_MODULES,
        role.FORBIDDEN_LIBRARIES,
        (*role.OPTIONAL_SERVE_IMPORTS, "gen_worker.view"),
    )
    assert [line for line in problems if "no serve-role module reaches it" in line]


def test_red_a_model_free_module_outside_the_serve_role_is_refused() -> None:
    """Model-free but not mint-free is the smaller of the two properties."""

    fence = _fence()
    problems = fence.check_model_free(
        ("gen_worker.view",), role.FORBIDDEN_LIBRARIES, (), within=role.SERVE_ROLE_MODULES
    )
    assert [line for line in problems if "not in SERVE_ROLE_MODULES" in line]


def test_the_model_free_set_is_spliced_into_the_serve_role_set() -> None:
    """One list. Retyping it is how pgw#824's drift starts."""

    assert set(role.MODEL_FREE_MODULES) <= set(role.SERVE_ROLE_MODULES)
    fence = _fence()
    assert tuple(fence._declared_tuple("SERVE_ROLE_MODULES")) == role.SERVE_ROLE_MODULES


def test_the_fence_selftest_passes() -> None:
    assert _fence().selftest() == 0


def test_type_checking_imports_are_not_followed() -> None:
    """They never execute; reporting them would report a cost nobody pays."""

    fence = _fence()
    binding = REPO / "src/gen_worker/family/catalog/_generated/flux1_dev.py"
    assert "from torch import Tensor" in binding.read_text()
    _, _, libraries = fence._imports(binding, "gen_worker.family.catalog._generated.flux1_dev")
    assert "torch" not in libraries


# --------------------------------------------------------- the runtime blocker


_DONE_TEST = r"""
import sys

from gen_worker.serve import guard, role

role.declare(role.ServeRole.ADOPT_ONLY)
guard.install()

for name in role.FORBIDDEN_LIBRARIES:
    try:
        __import__(name)
    except guard.ModelLibraryUnavailable as exc:
        assert exc.blocked == name, (name, exc.blocked)
    else:
        raise SystemExit("IMPORTED " + name)

# The whole serving surface, in a process that cannot acquire a model library.
from gen_worker.family.catalog import Flux1Dev
from gen_worker.family.catalog import flux1_dev_serve as fx
from gen_worker.family.scheduler import FlowMatchEulerDiscrete

instance = Flux1Dev.fake()
assert isinstance(instance.scheduler(), FlowMatchEulerDiscrete)
image = fx.generate(
    instance,
    resolution=768,
    clip_ids=fx.clip_token_ids([1, 2, 3], device="cpu"),
    t5_ids=fx.t5_token_ids([4, 5], device="cpu"),
    steps=3,
    guidance=3.5,
    seed=11,
)
assert tuple(image.shape) == (1, 3, 768, 768), image.shape

leaked = [name for name in role.FORBIDDEN_LIBRARIES if name in sys.modules]
if leaked:
    raise SystemExit("LEAKED " + ",".join(leaked))
assert guard.libraries_present() == ()
print("OK")
"""


def test_the_request_path_generates_in_a_process_that_cannot_import_diffusers() -> None:
    """pgw#1331's done-test, and it must be a SUBPROCESS.

    The test interpreter has already imported diffusers (the parity tests above
    need it), so an in-process assertion could never observe the property. This
    one boots a fresh adopt-only interpreter, installs the blocker, imports the
    typed family surface, and generates a whole image — then checks that no
    model library ever entered ``sys.modules``.
    """

    done = subprocess.run(
        [sys.executable, "-c", _DONE_TEST],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert done.returncode == 0, done.stderr[-4000:]
    assert done.stdout.strip().splitlines()[-1] == "OK"


def test_the_blocker_refuses_to_promise_something_already_false() -> None:
    """This interpreter HAS diffusers; installing now would be a lie."""

    from gen_worker.serve import guard

    previous = role.current()
    try:
        role._reset_for_test(role.ServeRole.ADOPT_ONLY)
        assert "diffusers" in sys.modules or "transformers" in sys.modules
        with pytest.raises(RuntimeError) as caught:
            guard.install()
        assert "already imported" in str(caught.value)
    finally:
        guard._uninstall_for_test()
        role._reset_for_test(previous)


def test_the_forbidden_libraries_are_the_ones_the_issue_names() -> None:
    assert role.FORBIDDEN_LIBRARIES == ("diffusers", "transformers")


# ------------------------------------------------------------- the measurement


def test_the_after_arm_of_the_measurement_loads_no_model_library() -> None:
    """The benchmark's own claim, asserted where CI can see it.

    The numbers move with the machine; what must not move is which libraries
    each arm ends up holding, because that is the property, and the arm lists
    are what the measurement quotes.
    """

    from gen_worker.benchmarks import request_path_imports as measurement

    after = measurement.measure("after", measurement.AFTER, repeat=1)
    assert not after.diffusers_loaded
    assert not after.transformers_loaded
    assert after.import_s < after.torch_s
    assert json.dumps([target for target in measurement.BEFORE])  # the arm is nameable
