"""pgw#1332 — the typed Family SDK, proven through its own path end to end.

The suite deliberately runs the REAL pipeline for a toy family: declare ->
fake-tensor export -> ``family_export_v1`` -> codegen -> import the generated
module -> call a typed binding. No hand-written snapshot stands in for the
export and no stub stands in for the generated module, because the defect class
this SDK exists to remove is *the generated surface disagreeing with the thing
it was generated from* — and a fixture that skips either end cannot see it.

What is faked is exactly one thing: the weights. That is the point of the FAKE
backing (greenfield B8), and it is what makes every row below runnable on a CI
box with no GPU, no hub and no checkpoint.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import msgspec
import pytest

from gen_worker import RequestContext, endpoint
from gen_worker.api.decorators import ATTR
from gen_worker.family import (
    Bucket,
    BucketMap,
    CallExample,
    DualBacking,
    EagerBacking,
    FakeBacking,
    Family,
    FamilyError,
    FamilyExport,
    FamilyRefusal,
    GraphFamily,
    Loop,
    LoopKind,
    Parameter,
    Runner,
    SessionState,
    Stage,
    Tuned,
    TunedValues,
    bind_families,
    fake_kwargs,
    resolve,
    resolve_tuned,
    tuned_fields,
    tuned_payload_fields,
)
from gen_worker.family.codegen import class_name, render_module
from gen_worker.family.drift import assert_recipe
from gen_worker.family.export import export_family
from gen_worker.family.snapshot import EXPORT_VERSION

from harness.family_toys_pgw1332 import TOY_AR, TOY_DIFFUSION, WIDTH, ToyTuned

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# The whole pipeline, once, shared by everything below.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_export() -> FamilyExport:
    """The toy family's DECLARATION-TIME export. A real trace, not a fixture."""
    return export_family(TOY_DIFFUSION)


@pytest.fixture(scope="module")
def toy_binding(toy_export: FamilyExport, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The generated module, imported. Generation is the code under test."""
    root = tmp_path_factory.mktemp("pgw1332_bindings")
    package = root / "pgw1332_generated"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "toy_diffusion.export.json").write_text(toy_export.dumps())
    (package / "toy_diffusion.py").write_text(
        render_module(
            toy_export,
            spec_module="harness.family_toys_pgw1332",
            spec_attr="TOY_DIFFUSION",
        )
    )
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("pgw1332_generated.toy_diffusion")
    finally:
        sys.path.remove(str(root))
    return module.ToyDiffusion


# ---------------------------------------------------------------------------
# The declaration refuses what a generator would have to paper over
# ---------------------------------------------------------------------------


def _reason(exc: pytest.ExceptionInfo[FamilyError]) -> FamilyRefusal:
    return exc.value.reason


def test_a_family_name_a_generator_could_not_emit_is_refused() -> None:
    """torchcg G1: codegen owns NO escaping rule, so the grammar refuses here.

    A hyphen is the live case — every legacy vocabulary name in this tree has
    one — and the refusal is what makes "a generator never mangles" true rather
    than aspirational.
    """
    with pytest.raises(FamilyError) as exc:
        Family(name="toy-diffusion", tuned=ToyTuned)
    assert _reason(exc) is FamilyRefusal.IDENTIFIER_INVALID

    with pytest.raises(FamilyError) as reserved:
        Family(name="class", tuned=ToyTuned)
    assert _reason(reserved) is FamilyRefusal.IDENTIFIER_INVALID


def test_an_eager_only_family_cannot_smuggle_in_runners() -> None:
    """The two tiers are type-honest: `Family` has no graph classes, period."""
    with pytest.raises(FamilyError) as exc:
        Family(
            name="toy_eager",
            tuned=ToyTuned,
            runners=(TOY_DIFFUSION.runner("denoiser"),),
        )
    assert _reason(exc) is FamilyRefusal.FAMILY_INVALID


def test_a_tuned_schema_with_no_tuned_fields_is_refused() -> None:
    """An empty schema is a struct nobody can stamp — an authoring mistake that
    would otherwise surface as silently-missing catalog values at serve time."""

    class Empty(TunedValues, frozen=True):
        pass

    with pytest.raises(FamilyError) as exc:
        Family(name="toy_empty", tuned=Empty)
    assert _reason(exc) is FamilyRefusal.TUNED_INVALID


def test_a_REFUSED_declaration_leaves_no_registration_behind() -> None:
    """Registration is a process-global side effect and runs LAST, on purpose.

    A declaration that raises after claiming its name would make the next
    import of the CORRECTED declaration read as a collision — one nothing can
    adjudicate, because both sides are the same family.
    """
    import gen_worker.families as families

    class RefusedTuned(TunedValues, frozen=True):
        steps: int = 1

    with pytest.raises(FamilyError):
        GraphFamily(
            name="toy_refused",
            tuned=RefusedTuned,
            buckets=(Bucket("resolution", (64,)),),
            runners=(TOY_DIFFUSION.runner("denoiser"),),
            loop=Loop(stages=(Stage("nonexistent"),)),
        )
    assert families.family_for("toy_refused") is None

    # ...and the corrected declaration registers cleanly rather than colliding.
    GraphFamily(
        name="toy_refused",
        tuned=RefusedTuned,
        buckets=(Bucket("resolution", (64, 128)),),
        runners=(TOY_DIFFUSION.runner("denoiser"),),
        loop=Loop(stages=(Stage("denoiser"),)),
    )
    assert families.family_for("toy_refused") is RefusedTuned


def test_a_loop_that_stages_an_undeclared_runner_is_refused() -> None:
    with pytest.raises(FamilyError) as exc:
        GraphFamily(
            name="toy_badloop",
            tuned=ToyTuned,
            buckets=(Bucket("resolution", (64,)),),
            runners=(TOY_DIFFUSION.runner("denoiser"),),
            loop=Loop(stages=(Stage("nonexistent"),)),
        )
    assert _reason(exc) is FamilyRefusal.LOOP_INVALID


def test_a_declared_runner_no_stage_runs_is_refused() -> None:
    with pytest.raises(FamilyError) as exc:
        GraphFamily(
            name="toy_unused",
            tuned=ToyTuned,
            buckets=(Bucket("resolution", (64,)),),
            runners=(
                TOY_DIFFUSION.runner("decoder"),
                TOY_DIFFUSION.runner("denoiser"),
            ),
            loop=Loop(stages=(Stage("denoiser", repeat="steps"),)),
            parameters=(Parameter("steps", minimum=1, maximum=4),),
        )
    assert _reason(exc) is FamilyRefusal.LOOP_INVALID


def test_a_host_owned_loop_refuses_a_repeat_count() -> None:
    """torchcg G14: an AR family's iteration is data-dependent, and a count in a
    document would be read by a second implementation as a real bound."""
    with pytest.raises(FamilyError) as exc:
        Loop(
            stages=(Stage("decode", repeat="steps"),),
            kind=LoopKind.HOST,
            session_state=SessionState.HOST,
        )
    assert _reason(exc) is FamilyRefusal.LOOP_INVALID


def test_a_runner_bucketing_on_an_undeclared_axis_is_refused() -> None:
    with pytest.raises(FamilyError) as exc:
        GraphFamily(
            name="toy_badaxis",
            tuned=ToyTuned,
            buckets=(Bucket("resolution", (64,)),),
            runners=(
                Runner(
                    "denoiser",
                    build=lambda layout: None,
                    example=lambda bucket, layout: CallExample(params=("x",)),
                    axes=("tokens",),
                ),
            ),
            loop=Loop(stages=(Stage("denoiser"),)),
        )
    assert _reason(exc) is FamilyRefusal.BUCKET_INVALID


# ---------------------------------------------------------------------------
# The export: what the declaration actually traced
# ---------------------------------------------------------------------------


def test_the_export_is_total_over_every_bucket_at_every_layout(
    toy_export: FamilyExport,
) -> None:
    """torchcg G6/G15: this is what makes a generated `Literal` exhaustive."""
    assert toy_export.axis_values == {"resolution": (64, 128)}
    for runner in toy_export.runners:
        for layout in runner.layouts:
            buckets = {
                variant.bucket for variant in runner.variants if variant.layout == layout
            }
            assert len(buckets) == 2, f"{runner.name} at {layout} is not total"


def test_the_export_carries_no_class_hash_and_no_checkpoint_field(
    toy_export: FamilyExport,
) -> None:
    """A declaration-time export cannot know a class hash (it folds `target` and
    device placement), and it is CLASS-LEVEL so no checkpoint fact can appear.

    Asserted over the canonical BYTES, not over the object graph: a field that
    leaked in through a nested `as_dict` would be invisible to an attribute
    check and perfectly visible here.
    """
    body = toy_export.canonical().decode("ascii")
    for forbidden in ("class_hash", "checkpoint", "weights", "ref\":", "sm\":", "toolchain"):
        assert forbidden not in body, f"the export leaked {forbidden!r}"


def test_the_export_round_trips_through_its_own_document(
    toy_export: FamilyExport,
) -> None:
    again = FamilyExport.loads(toy_export.dumps())
    assert again.digest() == toy_export.digest()
    assert again.as_dict() == toy_export.as_dict()


def test_an_unknown_export_version_is_refused_rather_than_best_effort_read(
    toy_export: FamilyExport,
) -> None:
    document = toy_export.as_dict()
    document["v"] = EXPORT_VERSION + 1
    with pytest.raises(FamilyError) as exc:
        FamilyExport.decode(document)
    assert _reason(exc) is FamilyRefusal.SNAPSHOT_VERSION_UNSUPPORTED


def test_an_unknown_field_is_refused_rather_than_ignored(
    toy_export: FamilyExport,
) -> None:
    document = toy_export.as_dict()
    document["tuned_values"] = {"steps": 4}
    with pytest.raises(FamilyError) as exc:
        FamilyExport.decode(document)
    assert _reason(exc) is FamilyRefusal.SNAPSHOT_INVALID


def test_a_bucket_gap_is_refused_as_incomplete_coverage(
    toy_export: FamilyExport,
) -> None:
    document = toy_export.as_dict()
    for runner in document["runners"]:
        if runner["name"] == "denoiser":
            runner["variants"] = runner["variants"][:1]
    with pytest.raises(FamilyError) as exc:
        FamilyExport.decode(document)
    assert _reason(exc) is FamilyRefusal.BUCKET_COVERAGE_INCOMPLETE


def test_variants_that_disagree_about_the_signature_are_refused(
    toy_export: FamilyExport,
) -> None:
    """torchcg G2: one runner is one binding. Two variants that project onto
    different signatures cannot share a generated callable, and picking one
    would make the binding lie about the other."""
    document = toy_export.as_dict()
    for runner in document["runners"]:
        if runner["name"] != "denoiser":
            continue
        second = runner["variants"][1]
        second["ingress"]["inputs"][0]["dtype"] = "bfloat16"
        second["ingress_digest"] = _redigest(second["ingress"])
    with pytest.raises(FamilyError) as exc:
        FamilyExport.decode(document)
    assert _reason(exc) is FamilyRefusal.SIGNATURE_DISAGREEMENT


def _redigest(ingress: dict[str, Any]) -> str:
    from gen_worker._vendor.torchcg.ingress import CallIngress

    return CallIngress.decode(ingress).digest()


# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------


def test_generation_is_a_pure_function_of_the_document(
    toy_export: FamilyExport,
) -> None:
    """Same document in, same bytes out — which is what makes the CI fence a
    byte comparison a two-minute job can afford."""
    first = render_module(toy_export, spec_module="m", spec_attr="F")
    again = render_module(FamilyExport.loads(toy_export.dumps()), spec_module="m", spec_attr="F")
    assert first == again


def test_no_generated_code_holds_a_string_runner_lookup(toy_binding: Any) -> None:
    """torchcg G7: the name resolves to an identity BEFORE anything runs, so a
    handler cannot reach a runner through a string it typed."""
    source = Path(sys.modules[toy_binding.__module__].__file__ or "").read_text()
    assert "ctx.runner(" not in source
    assert ".runner(" not in source
    for handle in ("denoiser", "decoder"):
        assert f"def {handle}(" in source


def test_the_bucket_axis_generates_a_closed_literal(toy_binding: Any) -> None:
    module = sys.modules[toy_binding.__module__]
    from typing import get_args

    assert set(get_args(module.ToyDiffusionResolution)) == {64, 128}
    assert set(get_args(module.ToyDiffusionLayout)) == {"bf16"}


def test_the_generated_class_carries_the_class_level_facts_and_no_others(
    toy_binding: Any, toy_export: FamilyExport
) -> None:
    assert toy_binding.FAMILY == "toy_diffusion"
    assert toy_binding.EXPORT_DIGEST == toy_export.digest()
    assert toy_binding.LOOP == (("denoiser", "counted", "steps"), ("decoder", "once", ""))
    assert toy_binding.LOOP_KIND == "staged"
    assert toy_binding.SCHEDULER == "euler_discrete"
    assert dict(toy_binding.SCHEDULER_PARAMETERS) == {"shift": 3.0}
    assert toy_binding.PARAMETERS == (("steps", 1, 100),)
    assert toy_binding.Tuned is ToyTuned


def test_class_name_is_generatable_without_an_escaping_rule() -> None:
    assert class_name("flux1_dev") == "Flux1Dev"
    assert class_name("sdxl") == "Sdxl"
    assert class_name("toy_diffusion") == "ToyDiffusion"


# ---------------------------------------------------------------------------
# The instance: the value a handler parameter receives
# ---------------------------------------------------------------------------


def test_a_family_class_has_no_public_constructor(toy_binding: Any) -> None:
    """torchcg G12: a family is checkpoint-free, so a bare one cannot be called.

    The refusal names the three ways to get a real one, because "no public
    constructor" with no alternative is a dead end rather than a design.
    """
    with pytest.raises(FamilyError) as exc:
        toy_binding()
    assert "instance(" in str(exc.value) and "fake()" in str(exc.value)


def test_two_instances_of_one_family_are_independent(toy_binding: Any) -> None:
    """torchcg G13: `flux_a: Flux1Dev, flux_b: Flux1Dev` is two checkpoints."""
    a = toy_binding.fake(tuned=ToyTuned(steps=4))
    b = toy_binding.fake(tuned=ToyTuned(steps=40))
    assert a is not b
    assert a.tuned.steps == 4 and b.tuned.steps == 40
    # And nothing at class level moved with either of them.
    assert toy_binding.Tuned is ToyTuned


def test_tuned_values_ride_the_instance_not_the_context(toy_binding: Any) -> None:
    """Paul, 2026-08-17: the delivery address moved; the values are still
    catalog-stamped per release slot exactly as th#1116 stamps them."""
    instance = toy_binding.fake(tuned=ToyTuned(steps=12, guidance=1.5))
    assert instance.tuned.steps == 12
    assert instance.tuned.guidance == 1.5
    assert not hasattr(instance, "defaults")


def test_with_tuned_returns_a_new_instance_rather_than_mutating(
    toy_binding: Any,
) -> None:
    """Rebinding in place would let one request's resolution leak into the next
    — the cross-axis bleed the three-axis split exists to prevent."""
    base = toy_binding.fake(tuned=ToyTuned(steps=4))
    tweaked = base.with_tuned(ToyTuned(steps=9))
    assert base.tuned.steps == 4
    assert tweaked.tuned.steps == 9
    assert tweaked is not base


def test_a_tuned_struct_of_the_wrong_family_is_refused(toy_binding: Any) -> None:
    class Other(TunedValues, frozen=True):
        steps: int = 1

    with pytest.raises(FamilyError) as exc:
        toy_binding.fake(tuned=Other())
    assert _reason(exc) is FamilyRefusal.TUNED_INVALID


def test_an_instance_without_a_worker_says_what_to_do_instead(
    toy_binding: Any,
) -> None:
    with pytest.raises(FamilyError) as exc:
        toy_binding.instance("owner/repo@2026.08")
    assert "fake()" in str(exc.value)


def test_the_on_load_hook_runs_when_an_instance_materializes() -> None:
    """Greenfield B6: today's `setup()` bodies get a home, and handlers stay
    stateless."""
    seen: list[str] = []

    class HookTuned(TunedValues, frozen=True):
        steps: int = 1

    Family(name="toy_hooked", tuned=HookTuned, on_load=lambda inst: seen.append(inst.ref))
    # The hook is declared on the family; the instance path is what runs it, so
    # this asserts the wiring through the generated base rather than the field.
    from gen_worker.family.runtime import FamilyBinding

    class Hooked(FamilyBinding):
        __slots__ = ()
        FAMILY = "toy_hooked"
        Tuned = HookTuned
        SPEC = Family(name="toy_hooked", tuned=HookTuned, on_load=lambda i: seen.append(i.ref))

    Hooked._materialize(ref="local/x", tuned=HookTuned(), backing=FakeBacking("toy_hooked"))
    assert seen == ["local/x"]


# ---------------------------------------------------------------------------
# Backings: one signature, three implementations
# ---------------------------------------------------------------------------


def test_the_fake_backing_returns_shape_correct_deterministic_tensors(
    toy_binding: Any,
) -> None:
    """Greenfield B8. Determinism is what lets a test assert on the value at
    all; the shape is what makes every line after the call real."""
    instance = toy_binding.fake()
    hidden = torch.zeros(1, 1, WIDTH, dtype=torch.float32)
    timestep = torch.zeros((), dtype=torch.float32)
    first = instance.denoiser(resolution=64, hidden_states=hidden, timestep=timestep)
    again = instance.denoiser(resolution=64, hidden_states=hidden, timestep=timestep)
    assert tuple(first.shape) == (1, 1, WIDTH)
    assert torch.equal(first, again)
    bigger = instance.denoiser(
        resolution=128,
        hidden_states=torch.zeros(1, 2, WIDTH, dtype=torch.float32),
        timestep=timestep,
    )
    assert tuple(bigger.shape) == (1, 2, WIDTH)
    assert not torch.equal(first.flatten()[:1], bigger.flatten()[:1])


def test_one_handler_body_serves_eager_and_fake_with_no_code_change(
    toy_binding: Any,
) -> None:
    """The dual-backing claim, stated as the only test that can prove it: the
    SAME closure runs against both and neither branch mentions a backing."""

    def handler(instance: Any) -> Any:
        hidden = torch.zeros(1, 1, WIDTH, dtype=torch.float32)
        latents = instance.denoiser(
            resolution=64, hidden_states=hidden, timestep=torch.zeros((), dtype=torch.float32)
        )
        return instance.decoder(resolution=64, latents=latents)

    faked = handler(toy_binding.fake())
    eager = handler(
        toy_binding.adopt(
            ref="local/toy@1",
            eager={
                "denoiser": TOY_DIFFUSION.runner("denoiser").build("bf16"),
                "decoder": TOY_DIFFUSION.runner("decoder").build("bf16"),
            },
        )
    )
    assert tuple(faked.shape) == tuple(eager.shape) == (1, 1, 3)


def test_a_bucket_the_family_never_declared_is_refused_not_approximated(
    toy_binding: Any,
) -> None:
    """torchcg G8: the lookup is exact. Ranking a live call is a SEPARATE
    contract (`ingress_selection_v1`), and this one never approximates."""
    instance = toy_binding.fake()
    with pytest.raises(FamilyError) as exc:
        instance.denoiser(
            resolution=999,
            hidden_states=torch.zeros(1, 1, WIDTH, dtype=torch.float32),
            timestep=torch.zeros((), dtype=torch.float32),
        )
    assert _reason(exc) is FamilyRefusal.CALL_INVALID


def test_a_compiled_backing_serves_only_its_traced_variant(toy_binding: Any) -> None:
    """torchcg G15: a compiled backing accepts exactly its traced bucket and
    layout. The dual backing serves everything else EAGERLY rather than quietly
    running the wrong graph."""
    calls: list[tuple[int, ...]] = []

    class _Armed:
        def __call__(self, *feeds: Any) -> Any:
            calls.append(tuple(int(d) for d in feeds[0].shape))
            return torch.zeros(1, 1, WIDTH, dtype=torch.float32)

    from gen_worker.family.backing import CompiledBacking

    variant = toy_binding.EXPORT.runner("denoiser").variant({"resolution": 64}, "bf16")
    compiled = CompiledBacking({CompiledBacking.key("denoiser", variant): _Armed()})
    instance = toy_binding._materialize(
        ref="local/toy@1",
        tuned=ToyTuned(),
        backing=DualBacking(
            eager=EagerBacking(
                {
                    "denoiser": TOY_DIFFUSION.runner("denoiser").build("bf16"),
                    "decoder": TOY_DIFFUSION.runner("decoder").build("bf16"),
                }
            ),
            compiled=compiled,
        ),
    )
    hidden = torch.zeros(1, 1, WIDTH, dtype=torch.float32)
    timestep = torch.zeros((), dtype=torch.float32)
    instance.denoiser(resolution=64, hidden_states=hidden, timestep=timestep)
    assert calls == [(1, 1, WIDTH)], "the armed variant did not answer its own call"
    assert instance.backing_kind("denoiser", {"resolution": 64}).value == "compiled"

    instance.denoiser(
        resolution=128,
        hidden_states=torch.zeros(1, 2, WIDTH, dtype=torch.float32),
        timestep=timestep,
    )
    assert calls == [(1, 1, WIDTH)], "an unarmed bucket reached the compiled runner"
    assert instance.backing_kind("denoiser", {"resolution": 128}).value == "eager"


def test_a_miss_with_no_backing_refuses_and_says_the_hub_routes(
    toy_binding: Any,
) -> None:
    """§4.28/§4.30: an adopt-only pod REFUSES and the hub routes. It never mints
    on demand, and no worker-initiated mint request exists to make."""
    from gen_worker.family.backing import CompiledBacking

    instance = toy_binding._materialize(
        ref="local/toy@1",
        tuned=ToyTuned(),
        backing=DualBacking(eager=None, compiled=CompiledBacking({})),
    )
    with pytest.raises(FamilyError) as exc:
        instance.denoiser(
            resolution=64,
            hidden_states=torch.zeros(1, 1, WIDTH, dtype=torch.float32),
            timestep=torch.zeros((), dtype=torch.float32),
        )
    assert _reason(exc) is FamilyRefusal.BACKING_MISSING
    assert "never mints on demand" in str(exc.value)


# ---------------------------------------------------------------------------
# Sessions (AR families)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ar_binding(tmp_path_factory: pytest.TempPathFactory) -> Any:
    export = export_family(TOY_AR)
    root = tmp_path_factory.mktemp("pgw1332_ar")
    package = root / "pgw1332_ar"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "toy_ar.export.json").write_text(export.dumps())
    (package / "toy_ar.py").write_text(
        render_module(export, spec_module="harness.family_toys_pgw1332", spec_attr="TOY_AR")
    )
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module("pgw1332_ar.toy_ar").ToyAr
    finally:
        sys.path.remove(str(root))


def test_a_host_loop_generates_typed_callables_and_no_driver(ar_binding: Any) -> None:
    """torchcg G14: the per-step classes and the state owner, and nothing else.
    A generator that synthesized a loop bound would have invented a fact the
    recipe deliberately refused to state."""
    import ast

    module = sys.modules[ar_binding.__module__]
    source = Path(module.__file__ or "").read_text()
    assert ar_binding.LOOP_KIND == "host"
    assert ar_binding.SESSION_STATE == "host"
    assert ar_binding.PARAMETERS == ()
    assert "def prefill(" in source and "def decode(" in source
    # Parsed, not grepped: the claim is about generated CODE, and a docstring
    # containing the word "for" is not a driver.
    tree = ast.parse(source)
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.comprehension))
    ], "the generator emitted iteration for a loop the recipe refused to bound"


def test_a_session_owns_its_state_and_refuses_to_outlive_it(ar_binding: Any) -> None:
    instance = ar_binding.fake()
    with instance.session() as session:
        session.state["kv"] = torch.zeros(1, 4)
        assert "kv" in session.state
    with pytest.raises(FamilyError) as exc:
        session.state["kv"] = None
    assert _reason(exc) is FamilyRefusal.SESSION_INVALID


def test_a_session_is_single_use(ar_binding: Any) -> None:
    instance = ar_binding.fake()
    with instance.session() as session:
        pass
    with pytest.raises(FamilyError) as exc:
        session._enter()
    assert _reason(exc) is FamilyRefusal.SESSION_INVALID


def test_a_staged_family_refuses_a_decode_session(toy_binding: Any) -> None:
    with pytest.raises(FamilyError) as exc:
        with toy_binding.fake().session():
            pass
    assert _reason(exc) is FamilyRefusal.SESSION_INVALID


# ---------------------------------------------------------------------------
# The drift assertion (torchcg G16)
# ---------------------------------------------------------------------------


def _recipe_from(export: FamilyExport, *, drift: bool = False) -> Any:
    """A minimal mint-shaped `recipe_v1` over the same runners.

    Built here rather than fixtured because the DIFFERENCE between the drifted
    and undrifted document has to be one field, visible in the diff.
    """
    from gen_worker._vendor.torchcg.recipe import (
        BucketAxis,
        GraphClassHash,
        GraphClassVariant,
        Loop as RecipeLoop,
        LoopStep,
        Recipe,
        RecipeParameter,
        RecipeRunner,
        SchedulerName,
        Scheduler as RecipeScheduler,
    )

    runners = []
    for index, runner in enumerate(export.runners):
        variants = tuple(
            GraphClassVariant(
                class_hash=GraphClassHash(f"{index:x}{position:015x}"),
                ingress_digest=variant.ingress_digest,
                ingress=variant.ingress,
                layout=variant.layout,
                bucket=variant.bucket,
            )
            for position, variant in enumerate(runner.variants)
        )
        runners.append(
            RecipeRunner(name=runner.name, axes=runner.axes, variants=variants)
        )
    if drift:
        runners = runners[1:]
    return Recipe(
        family=export.family,
        buckets=tuple(BucketAxis(name=name, values=values) for name, values in export.buckets),
        runners=tuple(runners),
        loop=RecipeLoop(
            kind=export.loop.kind,
            session_state=export.loop.session_state,
            stages=tuple(
                LoopStep(
                    runner=stage.runner, repeat=stage.repeat, parameter=stage.parameter
                )
                for stage in export.loop.stages
                if not drift or stage.runner != export.runners[0].name
            ),
        ),
        parameters=tuple(
            RecipeParameter(name=row.name, minimum=row.minimum, maximum=row.maximum)
            for row in export.parameters
        ),
        scheduler=(
            None
            if export.scheduler is None
            else RecipeScheduler(
                name=SchedulerName(export.scheduler.name),
                parameters=export.scheduler.parameters,
            )
        ),
    )


def test_a_recipe_that_restates_the_declaration_is_accepted(
    toy_export: FamilyExport,
) -> None:
    assert_recipe(toy_export, _recipe_from(toy_export))


def test_a_recipe_missing_a_declared_runner_is_declaration_drift(
    toy_export: FamilyExport,
) -> None:
    """The failure this exists for: a declaration changed in a PR, the bindings
    regenerated, and the artifacts on the fleet were minted from the previous
    one. Nothing about the artifact says it is stale."""
    with pytest.raises(FamilyError) as exc:
        assert_recipe(toy_export, _recipe_from(toy_export, drift=True))
    assert _reason(exc) is FamilyRefusal.DECLARATION_DRIFT


def test_a_recipe_for_another_family_is_refused(toy_export: FamilyExport) -> None:
    ar = export_family(TOY_AR)
    with pytest.raises(FamilyError) as exc:
        assert_recipe(toy_export, _recipe_from(ar))
    assert _reason(exc) is FamilyRefusal.DECLARATION_DRIFT


# ---------------------------------------------------------------------------
# Tuned resolution and the product grid
# ---------------------------------------------------------------------------


class _TunedIn(msgspec.Struct):
    prompt: str = ""
    steps: Tuned[int] = None
    guidance: Tuned[float] = None
    seed: int | None = None


def test_payload_values_win_over_tuned_and_none_falls_through() -> None:
    """Greenfield B3. The rule used to be hand-written once per field per
    handler, which is once per place it can be written backwards."""
    tuned = ToyTuned(steps=28, guidance=6.0)
    assert resolve_tuned(_TunedIn(steps=8), tuned, tuned_fields(ToyTuned)) == {
        "steps": 8,
        "guidance": 6.0,
    }
    assert resolve_tuned(_TunedIn(), tuned, tuned_fields(ToyTuned)) == {
        "steps": 28,
        "guidance": 6.0,
    }


def test_the_tuned_annotation_is_derivable_and_changes_no_wire_shape() -> None:
    """`Tuned[int]` has to be distinguishable from a plain optional or nothing
    could tell a field that DEFERS from one that merely has no default — and it
    has to decode and schematize identically, or the annotation would make the
    payload schema lie about what a client may send."""
    assert tuned_payload_fields(_TunedIn) == ("steps", "guidance")
    assert "seed" not in tuned_payload_fields(_TunedIn)

    class Plain(msgspec.Struct):
        prompt: str = ""
        steps: int | None = None
        guidance: float | None = None
        seed: int | None = None

    assert msgspec.json.decode(b'{"steps": 5}', type=_TunedIn).steps == 5
    assert msgspec.json.schema(_TunedIn)["$defs"]["_TunedIn"]["properties"] == (
        msgspec.json.schema(Plain)["$defs"]["Plain"]["properties"]
    )


def test_resolve_takes_a_payload_and_an_instance_and_needs_no_field_list(
    toy_binding: Any,
) -> None:
    """The shape a handler actually wants: one call, and a field added to the
    payload participates without anybody updating a second list."""
    instance = toy_binding.fake(tuned=ToyTuned(steps=28, guidance=6.0))
    assert resolve(_TunedIn(steps=8), instance) == {"steps": 8, "guidance": 6.0}
    assert resolve(_TunedIn(), instance) == {"steps": 28, "guidance": 6.0}


def test_tuned_fields_excludes_the_version_stamp() -> None:
    assert tuned_fields(ToyTuned) == ("steps", "guidance")


def test_catalog_values_decode_onto_the_instance_and_loras_overlay_them(
    toy_binding: Any,
) -> None:
    """th#1116's wire is unchanged; only the delivery address moved.

    The stamped document is the hub's, the merge rule is `api/slot`'s (one
    implementation, or a LoRA retunes differently depending on which path
    decoded it), and the result lands on the instance because it is a
    checkpoint-level fact.
    """
    from gen_worker.family import tuned_from_catalog

    neutral = tuned_from_catalog(toy_binding)
    assert isinstance(neutral, ToyTuned) and neutral.steps == 4  # the schema default
    stamped = tuned_from_catalog(toy_binding, '{"steps": 12, "guidance": 1.5}')
    assert isinstance(stamped, ToyTuned)
    assert (stamped.steps, stamped.guidance) == (12, 1.5)
    instance = toy_binding.fake(tuned=stamped)
    assert instance.tuned.steps == 12

    with pytest.raises(FamilyError) as exc:
        tuned_from_catalog(toy_binding, '{"steps": "twelve"}')
    assert _reason(exc) is FamilyRefusal.TUNED_INVALID

    with pytest.raises(FamilyError) as unknown:
        tuned_from_catalog(toy_binding, '{"not_a_field": 1}')
    assert _reason(unknown) is FamilyRefusal.TUNED_INVALID, (
        "the exported schema is closed (additionalProperties: false), so an "
        "unknown key is version skew rather than something to ignore"
    )


def test_a_product_grid_that_targets_an_undeclared_bucket_fails_the_build() -> None:
    """Greenfield B5: the failure is at BUILD, not on a pod."""
    with pytest.raises(FamilyError) as exc:
        BucketMap(TOY_DIFFUSION, "resolution", {"square": 64, "wide": 999})
    assert _reason(exc) is FamilyRefusal.GRID_INVALID


def test_the_warm_plan_derives_from_the_product_grid() -> None:
    grid = BucketMap(
        TOY_DIFFUSION, "resolution", {"square": 64, "portrait": 128, "landscape": 128}
    )
    assert grid.bucket_for("portrait") == 128
    assert grid.warm_plan() == (64, 128)
    with pytest.raises(FamilyError) as exc:
        grid.bucket_for("panorama")
    assert _reason(exc) is FamilyRefusal.GRID_INVALID


# ---------------------------------------------------------------------------
# @endpoint(families=...) — declaration, refusals, injection
# ---------------------------------------------------------------------------


class _In(msgspec.Struct):
    prompt: str = ""
    steps: int | None = None


class _Out(msgspec.Struct):
    shape: str = ""


def _handler_class(toy_binding: Any, body: str, name: str) -> Any:
    """Build a handler class annotated with the RUNTIME-generated family class.

    A real endpoint writes `toy: Flux1Dev` against a committed binding module.
    These tests generate the binding inside a fixture, so the class object has
    to reach the annotation through a namespace — the alternative is annotating
    `Any`, which would silently opt out of the very check being exercised.
    """

    namespace: dict[str, Any] = {
        "RequestContext": RequestContext,
        "_In": _In,
        "_Out": _Out,
        "Toy": toy_binding,
        "torch": torch,
        "WIDTH": WIDTH,
        "resolve_tuned": resolve_tuned,
    }
    exec(body, namespace)
    return namespace[name]


def test_a_handler_parameter_binds_a_resolved_instance(toy_binding: Any) -> None:
    """Paul's ruling, end to end: the generated class is the TYPE and the
    injected value is an instance carrying graph + weights + tuned values."""

    Generate = endpoint(families={"toy": toy_binding})(
        _handler_class(
            toy_binding,
            "class Generate:\n"
            "    def generate(self, ctx: RequestContext, p: _In, toy: Toy) -> _Out:\n"
            "        steps = resolve_tuned(p, toy.tuned, ('steps',))['steps']\n"
            "        out = toy.denoiser(\n"
            "            resolution=64,\n"
            "            hidden_states=torch.zeros(1, 1, WIDTH, dtype=torch.float32),\n"
            "            timestep=torch.zeros((), dtype=torch.float32),\n"
            "        )\n"
            "        return _Out(shape=f'{tuple(out.shape)}@{steps}')\n",
            "Generate",
        )
    )
    declared = getattr(Generate, ATTR).families
    assert declared == {"toy": toy_binding}
    kwargs = fake_kwargs(Generate)
    assert set(kwargs) == {"toy"}
    result = Generate().generate(None, _In(), **kwargs)  # type: ignore[arg-type]
    assert result.shape == f"(1, 1, {WIDTH})@4"


def test_two_parameters_of_one_family_get_two_independent_instances(
    toy_binding: Any,
) -> None:
    Compare = endpoint(families={"left": toy_binding, "right": toy_binding})(
        _handler_class(
            toy_binding,
            "class Compare:\n"
            "    def compare(self, ctx: RequestContext, p: _In, left: Toy, right: Toy) -> _Out:\n"
            "        return _Out(shape=f'{left.ref}|{right.ref}')\n",
            "Compare",
        )
    )
    kwargs = fake_kwargs(Compare)
    assert kwargs["left"] is not kwargs["right"]
    hidden = torch.zeros(1, 1, WIDTH, dtype=torch.float32)
    timestep = torch.zeros((), dtype=torch.float32)
    call: Any = kwargs["left"]
    left = call.denoiser(resolution=64, hidden_states=hidden, timestep=timestep)
    call = kwargs["right"]
    right = call.denoiser(resolution=64, hidden_states=hidden, timestep=timestep)
    assert not torch.equal(left, right), (
        "two parameters of one family returned identical tensors; a handler that "
        "swapped them would be invisible"
    )


def test_a_declared_family_with_no_handler_parameter_is_refused(
    toy_binding: Any,
) -> None:
    """A declared family nothing consumes is a prefetch for weights the handler
    never touches."""
    with pytest.raises(ValueError, match="has no such parameter"):

        @endpoint(families={"toy": toy_binding})
        class Missing:
            def generate(self, ctx: RequestContext, p: _In) -> _Out:
                return _Out()


def test_a_family_parameter_that_is_not_declared_is_refused(toy_binding: Any) -> None:
    """The reverse direction, and the one that actually bites: nothing
    prefetches it, and the failure would surface on a pod."""
    namespace: dict[str, Any] = {
        "RequestContext": RequestContext,
        "_In": _In,
        "_Out": _Out,
        "Toy": toy_binding,
    }
    exec(
        "class Undeclared:\n"
        "    def generate(self, ctx: RequestContext, p: _In, toy: Toy) -> _Out:\n"
        "        return _Out()\n",
        namespace,
    )
    with pytest.raises(ValueError, match="does not declare it"):
        endpoint(namespace["Undeclared"])


def test_a_family_parameter_must_be_annotated_with_its_family(
    toy_binding: Any, ar_binding: Any
) -> None:
    namespace: dict[str, Any] = {
        "RequestContext": RequestContext,
        "_In": _In,
        "_Out": _Out,
        "Ar": ar_binding,
    }
    exec(
        "class Crossed:\n"
        "    def generate(self, ctx: RequestContext, p: _In, toy: Ar) -> _Out:\n"
        "        return _Out()\n",
        namespace,
    )
    with pytest.raises(TypeError, match="families binds"):
        endpoint(families={"toy": toy_binding})(namespace["Crossed"])


def test_binding_a_declaration_instead_of_its_generated_type_is_refused() -> None:
    class Wrong:
        def generate(self, ctx: RequestContext, p: _In, toy: object) -> _Out:
            return _Out()

    with pytest.raises(TypeError, match="DECLARATION"):
        endpoint(families={"toy": TOY_DIFFUSION})(Wrong)


def test_a_job_binds_families_exactly_as_an_endpoint_does(toy_binding: Any) -> None:
    """`@endpoint`/`@job` portability is a REQUIREMENT, not a style (jobs.py).

    A job that could not take a family instance would make promotion a rewrite
    for precisely the bodies most likely to want one.
    """
    from gen_worker import job
    from gen_worker.api.jobs import JOB_ATTR

    namespace: dict[str, Any] = {
        "RequestContext": RequestContext,
        "_In": _In,
        "_Out": _Out,
        "Toy": toy_binding,
    }
    exec(
        "def render(ctx: RequestContext, p: _In, toy: Toy) -> _Out:\n"
        "    return _Out(shape=toy.ref)\n",
        namespace,
    )
    body = namespace["render"]
    declared = job(families={"toy": toy_binding})(body)
    assert getattr(declared, JOB_ATTR).families == {"toy": toy_binding}
    kwargs = fake_kwargs(declared)
    assert declared(None, _In(), **kwargs).shape.startswith("fake:")


def test_a_job_still_refuses_an_undeclared_extra_parameter() -> None:
    from gen_worker import job

    with pytest.raises(TypeError, match=r"plus declared family instances"):

        @job
        def render(ctx: RequestContext, p: _In, extra: str) -> _Out:
            return _Out()


def test_bind_families_refuses_an_instance_of_the_wrong_family(
    toy_binding: Any, ar_binding: Any
) -> None:
    with pytest.raises(FamilyError) as exc:
        bind_families({"toy": toy_binding}, {"toy": ar_binding.fake()})
    assert _reason(exc) is FamilyRefusal.BACKING_MISSING


def test_bind_families_refuses_a_missing_instance(toy_binding: Any) -> None:
    with pytest.raises(FamilyError) as exc:
        bind_families({"toy": toy_binding}, {})
    assert _reason(exc) is FamilyRefusal.BACKING_MISSING


def test_the_discovery_manifest_carries_the_families_a_function_binds(
    toy_binding: Any, toy_export: FamilyExport
) -> None:
    """Placement reads this to prefetch weights and verify the VRAM fit BEFORE a
    request lands — which is the entire reason static declaration is the
    default."""
    from gen_worker.registry import extract_specs

    Generate = endpoint(families={"toy": toy_binding})(
        _handler_class(
            toy_binding,
            "class Generate:\n"
            "    def generate(self, ctx: RequestContext, p: _In, toy: Toy) -> _Out:\n"
            "        return _Out()\n",
            "Generate",
        )
    )
    (spec,) = extract_specs(Generate, walked_module="tests.pgw1332")
    assert spec.families == {"toy": toy_binding}
    block = [
        {
            "parameter": name,
            "family": str(getattr(family, "FAMILY", "")),
            "export_digest": str(getattr(family, "EXPORT_DIGEST", "")),
        }
        for name, family in sorted(spec.families.items())
    ]
    assert block == [
        {
            "parameter": "toy",
            "family": "toy_diffusion",
            "export_digest": toy_export.digest(),
        }
    ]


# ---------------------------------------------------------------------------
# The catalog, and the fence that keeps its two committed halves honest
# ---------------------------------------------------------------------------


CATALOG = Path(__file__).resolve().parents[1] / "src/gen_worker/family/catalog"
GENERATED = CATALOG / "_generated"
CATALOG_SPECS = {
    "flux1_dev": "gen_worker.family.catalog.flux1_dev:FLUX1_DEV",
    "sdxl": "gen_worker.family.catalog.sdxl:SDXL",
}


@pytest.mark.parametrize("family", sorted(CATALOG_SPECS))
def test_the_committed_binding_is_what_its_committed_export_implies(family: str) -> None:
    """THE fence. Codegen is pure, so this is a byte comparison — and it catches
    the case that actually happens: an export regenerated and committed while
    the binding beside it was not."""
    export = FamilyExport.loads((GENERATED / f"{family}.export.json").read_bytes())
    module, _, attr = CATALOG_SPECS[family].partition(":")
    expected = render_module(export, spec_module=module, spec_attr=attr)
    assert (GENERATED / f"{family}.py").read_text() == expected, (
        f"{family}.py is not what {family}.export.json implies; regenerate with "
        f"`gen-worker family generate <path>/{family}.export.json --spec "
        f"{CATALOG_SPECS[family]}`"
    )


@pytest.mark.parametrize("family", sorted(CATALOG_SPECS))
def test_the_committed_export_is_canonical_and_class_level(family: str) -> None:
    raw = (GENERATED / f"{family}.export.json").read_text()
    export = FamilyExport.loads(raw)
    assert raw == export.dumps(), "the committed document is not canonical"
    assert str(export.family) == family
    body = export.canonical().decode("ascii")
    assert "class_hash" not in body and "checkpoint" not in body


def test_the_family_facade_does_not_import_the_exporter() -> None:
    """The claim `scripts/lint_serving_process_compiles.py` records, EXECUTED.

    `gen_worker.family.export` runs `torch.export`, and th#1299 keeps that off
    any serving process. Its safety here is structural — nothing on the serve
    path reaches it — so this asserts the structure rather than the intent: the
    facade, the generated bindings and the catalog all import cleanly with the
    exporter made unimportable.
    """
    import os
    import subprocess

    program = """
import sys


class Refuse:
    def find_spec(self, name, path=None, target=None):
        if name == "gen_worker.family.export":
            raise AssertionError("the serve path imported the declaration exporter")
        return None


sys.meta_path.insert(0, Refuse())

import gen_worker.family  # noqa: F401
from gen_worker.family.catalog import Sdxl

assert Sdxl.fake().tuned.steps == 28
print("ok")
"""
    root = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_the_catalog_imports_without_torch_or_diffusers() -> None:
    """pgw#1328: an adopt-only serve role holds the bindings and must not
    acquire model code by importing one. Proved by EXECUTING the import with
    both made unimportable, not by reading the import list."""
    import os
    import subprocess

    program = """
import sys


class Refuse:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "diffusers"):
            raise AssertionError(f"the catalog reached {name}")
        return None


sys.meta_path.insert(0, Refuse())
try:
    __import__("torch")
except AssertionError:
    pass
else:
    raise SystemExit("the guard never fired -- this test proves nothing")

from gen_worker.family.catalog import Flux1Dev, Sdxl

assert Sdxl.FAMILY == "sdxl" and Flux1Dev.FAMILY == "flux1_dev"
# The DECLARATION loads too, and that is the stronger property: a catalog
# declaration imports diffusers only inside `build`, so it is model-code-free
# until a mint or an eager backing actually needs the architecture.
assert Sdxl.SPEC is not None and Sdxl.SPEC.name == "sdxl"
assert Sdxl.EXPORT.runner("denoiser").layouts == ("bf16",)
print("ok")
"""
    root = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


@pytest.mark.parametrize("family", sorted(CATALOG_SPECS))
def test_every_catalog_family_is_callable_hubless(family: str) -> None:
    """The catalog's own B8 proof: a fake instance of each shipped family
    answers its declared runners with the right shapes."""
    from gen_worker.family import catalog

    binding = getattr(catalog, class_name(family))
    instance = binding.fake()
    variant = binding.EXPORT.runner("decoder").variant({"resolution": 1024}, "bf16")
    (spec,) = variant.ingress.inputs
    latents = torch.zeros(*[int(d) for d in spec.shape], dtype=getattr(torch, spec.dtype))
    image = instance.decoder(resolution=1024, latents=latents)
    assert tuple(image.shape) == (1, 3, 1024, 1024)


def test_the_catalog_index_and_its_exports_agree() -> None:
    from gen_worker.family import catalog

    assert sorted(catalog.__all__) == sorted(catalog._FAMILIES)
    for name in catalog.__all__:
        assert getattr(catalog, name) is not None
    with pytest.raises(AttributeError, match="the catalog has no"):
        catalog.NoSuchFamily  # noqa: B018


def test_a_family_owns_its_tuned_schema_registration() -> None:
    """The naming-collision ruling, asserted: the family registers its schema,
    and the retired free-standing decorator is gone."""
    import gen_worker.families as families

    assert not hasattr(families, "family"), (
        "the `@family(...)` class decorator is back; pgw#1332 resolved the name "
        "in favour of the family-owned schema"
    )
    from gen_worker.family.catalog.sdxl import SDXL, SdxlLoraTuned, SdxlTuned

    assert families.family_for("sdxl") is SdxlTuned
    assert families.family_for("sdxl", kind="lora") is SdxlLoraTuned
    assert SDXL.tuned is SdxlTuned
    schema = families.export_json_schema("sdxl")
    assert schema["additionalProperties"] is False
    assert json.dumps(schema)  # the exported contract stays JSON-serializable
