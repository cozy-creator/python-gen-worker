"""SDK authoring surface: the decorators, the job contract, what a handler emits.

Sections keep their incident id; the full narratives live in the tracker.
"""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

import msgspec
import pytest
from harness.progress_wait import Cadence, await_progress

from gen_worker import (
    AxisClass,
    Compile,
    CompileAxis,
    DynamicDim,
    Hub,
    JobContext,
    RequestContext,
    Resources,
    Slot,
    TextLengthExceededError,
    endpoint,
    job,
    pad_text_sequence,
)
from gen_worker.api.compile_axis import extract_payload_axes, warm_guidance_values
from gen_worker.api.errors import (
    JobProgressStalledError,
    MediaNotDeclaredError,
    NonMonotonicProgressError,
    PublishNotDeclaredError,
)
from gen_worker.api.slot import resolve_slot
from gen_worker.api.tree import (
    KIND_CONFIG,
    KIND_WEIGHTS,
    derive_components,
    is_introspectable,
    validate_no_sibling_parts,
)
from gen_worker.executor import Executor
from gen_worker.families.base import GenerationDefaults, register_family
from gen_worker.jobs import (
    DEFAULT_PHASE_BUDGET_S,
    JobDispatch,
    ProgressWatch,
    execute_job,
)
from gen_worker.jobs import execute_job as _real_execute_job
from gen_worker.lifecycle_intents import IntentRegistry
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec, extract_job_spec, extract_specs
from gen_worker.view import UnknownSamplerError, clone_scheduler, for_request

# ============================================================================
# pgw#647 — SDK v2 surface: derived component tree, sibling-as-part lint,
#   CompileAxis payload classes, the text-sequence lint, Resources v2,
#   derived config schema, per-request ...
# ============================================================================

class _V2Defaults(GenerationDefaults, frozen=True):
    steps: int = 28
    guidance: float = 6.0


register_family("v2-testfam", _V2Defaults)


class _In(msgspec.Struct):
    prompt: str = ""
    model: str = ""


class _Out(msgspec.Struct):
    ok: bool = True


class _FakePipeline:
    """Diffusers-shaped: self-describes via _get_signature_keys."""

    @classmethod
    def _get_signature_keys(cls, obj):
        return {"unet", "vae", "text_encoder", "tokenizer", "scheduler"}, set()

    @classmethod
    def from_pretrained(cls, path, **kw):
        return cls()


def test_tree_derives_and_classifies_parts():
    tree = derive_components(_FakePipeline)
    assert tree == {
        "unet": KIND_WEIGHTS,
        "vae": KIND_WEIGHTS,
        "text_encoder": KIND_WEIGHTS,
        "tokenizer": KIND_CONFIG,
        "scheduler": KIND_CONFIG,
    }


def test_str_and_plain_classes_are_not_introspectable():
    assert not is_introspectable(str)
    assert derive_components(str) is None

    class Plain:
        pass

    assert derive_components(Plain) is None


def test_real_diffusers_pipeline_tree_derives():
    diffusers = pytest.importorskip("diffusers")
    tree = derive_components(diffusers.StableDiffusionXLPipeline)
    assert tree is not None
    assert tree.get("unet") == KIND_WEIGHTS
    assert tree.get("vae") == KIND_WEIGHTS
    assert tree.get("scheduler") == KIND_CONFIG
    assert tree.get("tokenizer") == KIND_CONFIG


def test_sibling_component_slot_is_rejected():
    slots: dict[str, Any] = {
        "pipeline": Slot(_FakePipeline),
        "vae": Slot(_FakePipeline),  # 'vae' IS a part of pipeline's tree
    }
    with pytest.raises(ValueError, match="COMPONENT.*catalog data"):
        validate_no_sibling_parts("Owner.gen", slots, {})


def test_sibling_str_slot_next_to_tree_is_rejected():
    slots: dict[str, Any] = {
        "pipeline": Slot(_FakePipeline),
        "turbo_lora": Slot(str),
    }
    with pytest.raises(ValueError, match="sibling-as-part|adapters riding"):
        validate_no_sibling_parts("Owner.gen", slots, {})


def test_all_str_multi_slot_escape_hatch_survives():
    slots: dict[str, Any] = {"weights": Slot(str), "mmproj": Slot(str)}
    validate_no_sibling_parts("Owner.gen", slots, {})  # no raise


def test_registry_records_derived_tree():
    @endpoint(models={"pipeline": Slot(_FakePipeline, selected_by="model")})
    class Gen:
        def setup(self, pipeline: _FakePipeline) -> None:
            self.pipeline = pipeline

        def generate(self, ctx: RequestContext[_V2Defaults], p: _In) -> _Out:
            return _Out()

    (spec,) = extract_specs(Gen)
    assert spec.slot_components["pipeline"]["unet"] == KIND_WEIGHTS
    assert spec.defaults_type is _V2Defaults
    assert spec.slot_family["pipeline"] == "v2-testfam"


class _AxisIn(msgspec.Struct):
    prompt: str = ""
    guidance_scale: Annotated[float, CompileAxis(classes=(
        AxisClass("cfg_off", match=lambda v: v == 0, warm=0.0),
        AxisClass("cfg_on", match=lambda v: v != 0, warm=5.0),
    ))] = 5.0
    aspect: Annotated[str, CompileAxis(classes="enum")] = "1:1"


def test_axis_extraction_and_classification():
    with pytest.raises(ValueError):
        extract_payload_axes("owner", _AxisIn)  # plain str is not enumerable

    class _EnumIn(msgspec.Struct):
        guidance_scale: Annotated[float, CompileAxis(classes=(
            AxisClass("cfg_off", match=lambda v: v == 0, warm=0.0),
            AxisClass("cfg_on", match=lambda v: v != 0, warm=5.0),
        ))] = 5.0

    axes = extract_payload_axes("owner", _EnumIn)
    assert [a.field for a in axes] == ["guidance_scale"]
    axis = axes[0]
    assert axis.class_names == ("cfg_off", "cfg_on")
    assert axis.classify(0.0) == "cfg_off"
    assert axis.classify(7.5) == "cfg_on"
    assert warm_guidance_values(axes) == (0.0, 5.0)


def test_enum_axis_from_literal():
    class _LitIn(msgspec.Struct):
        aspect: Annotated[Literal["1:1", "3:4", "16:9"], CompileAxis(classes="enum")] = "1:1"

    (axis,) = extract_payload_axes("owner", _LitIn)
    assert axis.class_names == ("1:1", "3:4", "16:9")
    assert axis.classify("3:4") == "3:4"
    assert axis.classify("9:16") is None  # outside the envelope


def test_axis_warm_must_satisfy_its_own_match():
    with pytest.raises(ValueError, match="does not satisfy"):
        AxisClass("cfg_off", match=lambda v: v == 0, warm=5.0)


def test_compile_without_text_axis_is_a_decoration_error():
    with pytest.raises(ValueError, match="text-sequence axis"):
        @endpoint(compile=Compile(shapes=((1024, 1024),), family="v2fam"))
        class Gen:
            def setup(self) -> None:
                pass

            def generate(self, ctx: RequestContext, p: _In) -> _Out:
                return _Out()


def test_compile_with_pinned_text_len_passes():
    @endpoint(compile=Compile(shapes=((1024, 1024),), family="v2fam", text_len=77))
    class Gen:
        def setup(self) -> None:
            pass

        def generate(self, ctx: RequestContext, p: _In) -> _Out:
            return _Out()

    (spec,) = extract_specs(Gen)
    assert spec.compile is not None and spec.compile.text_len == 77


def test_compile_with_dynamic_sequence_passes_and_batch_axis_supported():
    @endpoint(compile=Compile(
        shapes=((1024, 1024),), family="v2fam",
        dynamic=(DynamicDim("sequence", min=64, max=512),
                 DynamicDim("batch", min=2, max=16)),
    ))
    class Gen:
        def setup(self) -> None:
            pass

        def generate(self, ctx: RequestContext, p: _In) -> _Out:
            return _Out()

    (spec,) = extract_specs(Gen)
    graph = spec.compile_contract()
    assert graph is not None
    facts = graph.contract_facts()
    assert facts["dynamic"] == [
        {"dim": "sequence", "min": 64, "max": 512},
        {"dim": "batch", "min": 2, "max": 16},
    ]


def test_dynamic_dim_min_must_respect_01_specialization():
    # torch's 0/1 specialization is not overridable: min >= 2.
    with pytest.raises(ValueError, match="min must be >= 2"):
        DynamicDim("batch", min=1, max=8)


def test_contract_digest_changes_with_the_contract():
    def graph(**kw):
        @endpoint(compile=Compile(shapes=((1024, 1024),), family="v2fam", **kw))
        class Gen:
            def setup(self) -> None:
                pass

            def generate(self, ctx: RequestContext, p: _In) -> _Out:
                return _Out()

        (spec,) = extract_specs(Gen)
        graph = spec.compile_contract()
        assert graph is not None
        return graph.contract_digest()

    a = graph(text_len=77)
    b = graph(text_len=512)
    c = graph(text_len=0)
    assert len({a, b, c}) == 3  # pre-fix and post-fix compiled graphs key differently


def test_resources_v2_deleted_fields_raise():
    # `vram_gb` and `ram_gb` stay deleted; th#1867 deleted the VRAM markers.
    # pgw#1313 deleted the last two bespoke axes — `compute_capability` and
    # `ram_gb_hint` — into the one requirement vocabulary. No aliases.
    for kw in ({"vram_gb": 12}, {"ram_gb": 48}, {"min_compute_capability": 8.0},
               {"compute_capability": 8.9}, {"ram_gb_hint": 64}):
        with pytest.raises(TypeError):
            Resources(**kw)


def test_resources_v2_the_arch_floor_survives_as_a_requirement_term():
    # pgw#660 REVERSED this part of the v2 cut (Paul, 2026-07-26). The cut was
    # right that precision-per-card is the ladder's call and wrong that an
    # architecture floor is the same thing: scaled_mm below sm_89 is
    # incapability, not a slower rung, and the hub placed the fp8 producer on
    # sm_80 A100s for want of the declaration. See test_compute_capability_pgw660.
    req = Resources(requires="sm89+").requirement()
    assert req is not None
    terms = req.min_terms()
    assert terms is not None and terms.min_sm == 89
    assert Resources(requires="sm89+").gpu is True


def test_resources_v2_hint_and_gpu_count_imply_gpu():
    assert Resources(requires="sm89+").gpu is True
    assert Resources(gpu_count=2).gpu is True
    assert Resources().gpu is False
    with pytest.raises(ValueError):
        Resources(gpu_count=0)


def test_resolve_slot_neutral_defaults_without_catalog_metadata():
    slot: Any = Slot(str, default_checkpoint=Hub("acme/ckpt"))
    resolved = resolve_slot(
        "pipeline", slot, ref=slot.default_checkpoint,
        defaults_cls=_V2Defaults, family="v2-testfam",
    )
    assert resolved.defaults == _V2Defaults()


def test_resolve_slot_catalog_metadata_wins():
    slot: Any = Slot(str, default_checkpoint=Hub("acme/ckpt"))
    resolved = resolve_slot(
        "pipeline", slot, ref=slot.default_checkpoint,
        defaults_cls=_V2Defaults, family="v2-testfam",
        raw_metadata_json='{"steps": 8, "guidance": 1.0}',
    )
    assert resolved.defaults.steps == 8
    assert resolved.defaults.guidance == 1.0


def test_ctx_defaults_reads_root_slot():
    from gen_worker.api.slot import ResolvedSlot

    ctx: Any = RequestContext(
        "req-1",
        resolved_slots={"pipeline": ResolvedSlot(
            ref=Hub("acme/ckpt"), defaults=_V2Defaults(steps=12))},
    )
    assert ctx.defaults.steps == 12


def test_bad_ctx_annotation_is_a_walk_error():
    with pytest.raises(ValueError, match="GenerationDefaults"):
        @endpoint
        class Gen:
            def generate(self, ctx: "RequestContext[Any]", p: _In) -> _Out:  # noqa: F821
                return _Out()

        extract_specs(Gen)


def test_extra_handler_params_are_rejected():
    """pgw#647: A per-handler MODEL argument is still rejected, and for the v2 reason."""
    with pytest.raises(TypeError, match=r"per-handler MODEL args are rejected"):
        @endpoint(models={"pipeline": Hub("acme/ckpt")})
        class Gen:
            def setup(self, pipeline: str) -> None:
                self.pipeline = pipeline

            def generate(self, ctx: RequestContext, p: _In, pipeline: str) -> _Out:
                return _Out()


def test_a_family_instance_param_is_the_one_permitted_extra(tmp_path):
    """pgw#1332's carve-out, asserted beside the rule it carves out of."""
    from gen_worker.model.catalog import Sdxl

    @endpoint(families={"sdxl": Sdxl})
    class Gen:
        def generate(self, ctx: RequestContext, p: _In, sdxl: Sdxl) -> _Out:
            return _Out()

    declared = getattr(Gen, "__gen_worker_endpoint__").families
    assert {name: row.model for name, row in declared.items()} == {"sdxl": Sdxl}


def test_class_models_require_setup():
    with pytest.raises(ValueError, match="require a setup"):
        @endpoint(models={"pipeline": Hub("acme/ckpt")})
        class Gen:
            def generate(self, ctx: RequestContext, p: _In) -> _Out:
                return _Out()


class _FakeSchedulerA:
    def __init__(self, config=None, **overrides):
        self.config = dict(config or {"num_train_timesteps": 1000})
        self.config.update(overrides)
        self.step_index = 0

    @classmethod
    def from_config(cls, config, **overrides):
        return cls(dict(config), **overrides)


class _FakeUnet:
    pass


class _ViewPipeline:
    def __init__(self):
        self.unet = _FakeUnet()
        self.vae = _FakeUnet()
        self.scheduler = _FakeSchedulerA()
        self._internal_dict = {"scheduler": ("x", "FakeSchedulerA")}


def test_for_request_shares_modules_and_owns_scheduler():
    pipe = _ViewPipeline()
    pipe.scheduler.step_index = 17  # dirty state from a previous request
    view = for_request(pipe)
    assert view.unet is pipe.unet          # weights shared by reference
    assert view.vae is pipe.vae
    assert view.scheduler is not pipe.scheduler  # scheduler is view-private
    assert view.scheduler.step_index == 0        # fresh trajectory
    assert pipe.scheduler.step_index == 17       # instance untouched
    assert type(view) is type(pipe)


def test_for_request_applies_v_prediction_objective():
    pipe = _ViewPipeline()
    view = for_request(pipe, objective="v_prediction")
    assert view.scheduler.config["prediction_type"] == "v_prediction"
    assert view.scheduler.config["rescale_betas_zero_snr"] is True
    assert "prediction_type" not in pipe.scheduler.config


def test_clone_scheduler_unknown_sampler_is_typed():
    pipe = _ViewPipeline()
    with pytest.raises(UnknownSamplerError, match="known:"):
        clone_scheduler(pipe, sampler="not-a-sampler")


def test_for_request_sampler_swap_with_real_diffusers():
    diffusers = pytest.importorskip("diffusers")
    pipe = _ViewPipeline()
    pipe.scheduler = diffusers.EulerDiscreteScheduler()
    view = for_request(pipe, sampler="euler_a")
    assert type(view.scheduler).__name__ == "EulerAncestralDiscreteScheduler"
    assert type(pipe.scheduler).__name__ == "EulerDiscreteScheduler"


def test_pad_text_sequence_is_canonically_strided():
    torch = pytest.importorskip("torch")
    # dim 0 of size 1 — exactly the case where .contiguous() is a no-op and
    # a size-only pin is not a pin.
    embeds = torch.randn(1, 33, 8)[:, :, :]
    mask = torch.ones(1, 33, dtype=torch.long)
    out, out_mask = pad_text_sequence(embeds, 77, mask=mask)
    assert tuple(out.shape) == (1, 77, 8)
    assert out.stride() == (77 * 8, 8, 1)  # canonical row-major strides
    assert torch.equal(out[:, :33, :], embeds)
    assert out is not None and out_mask is not None
    assert torch.equal(out[:, 33:, :], torch.zeros(1, 44, 8))
    assert tuple(out_mask.shape) == (1, 77)
    assert int(out_mask.sum()) == 33


def test_pad_text_sequence_over_length_is_typed_refusal():
    torch = pytest.importorskip("torch")
    embeds = torch.randn(1, 100, 8)
    with pytest.raises(TextLengthExceededError):
        pad_text_sequence(embeds, 77)


# ============================================================================
# pgw#1294 — pgw#1294 ⇄ th#2049 (JOBS program, issue 2 of 8): ``@job`` —
#   run-once functions with a PORTABLE body, a merged ``JobContext``, and
#   liveness that is POSITION, not pulse.
# ============================================================================

class BakeIn(msgspec.Struct):
    rung: str = "w8a8"


class BakeOut(msgspec.Struct):
    rung: str
    ctx_class: str


@job(publishes=True, resources=Resources(vcpus=2))
@endpoint(kind="conversion", publishes=True, name="bake")
def bake_both(ctx: JobContext, spec: BakeIn) -> BakeOut:
    ctx.progress(position=1, total=2, phase="bake")
    ctx.metric({"cosine": 0.999}, step=1, total=2)
    ctx.progress(position=2, total=2, phase="bake")
    return BakeOut(rung=spec.rung, ctx_class=type(ctx).__mro__[1].__name__)


def test_one_body_carries_both_declarations() -> None:
    """The portability property, stated on the objects themselves."""
    job_spec = extract_job_spec(bake_both)
    (endpoint_spec,) = extract_specs(bake_both)
    assert job_spec is not None
    # Same function object under both harnesses — not a copy, not a wrapper.
    assert job_spec.method is bake_both
    assert endpoint_spec.method is bake_both
    assert job_spec.payload_type is endpoint_spec.payload_type is BakeIn
    assert job_spec.output_type is endpoint_spec.output_type is BakeOut
    assert job_spec.publishes and endpoint_spec.publishes


def test_job_context_is_a_superset_of_the_producer_endpoint_context() -> None:
    """pgw#1294: Promotion must be a redeploy, not a rewrite: every name a producer endpoint handler may use has..."""
    from gen_worker.request_context import JobContext as JC

    for name in (
        "mktemp", "checkpoint_dir", "resolve_dataset", "dataset_paths",
        "save_checkpoint", "open_checkpoint_stream", "cancelled",
        "call_endpoint", "progress", "metric", "training_metric", "log",
    ):
        assert hasattr(JC, name), name


PKG_SRC = """
    import msgspec
    from gen_worker import JobContext, Resources, endpoint, job

    class BakeIn(msgspec.Struct):
        rung: str = "w8a8"

    class BakeOut(msgspec.Struct):
        rung: str
        positions: int

    @job(publishes=True, resources=Resources(vcpus=2))
    @endpoint(kind="conversion", publishes=True, name="bake")
    def bake(ctx: JobContext, spec: BakeIn) -> BakeOut:
        ctx.progress(position=1, total=2, phase="bake")
        ctx.progress(position=2, total=2, phase="bake")
        return BakeOut(rung=spec.rung, positions=int(ctx.position("bake") or 0))
"""


@pytest.fixture()
def both_harness_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("GEN_WORKER_LOCAL_OUTPUT_DIR", str(tmp_path / "out"))
    pkg = tmp_path / "portable_job"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent(PKG_SRC))
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "portable-job"\nversion = "0.0.0"\n'
        '[tool.gen_worker]\nmain = "portable_job.main"\n'
    )
    return tmp_path


def test_the_same_body_runs_green_under_both_harnesses(
    both_harness_pkg: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """THE charter obligation (th#2049 constraint 1). Both CLIs, end to end."""
    import json

    from gen_worker.cli import main

    payload = both_harness_pkg / "payload.json"
    payload.write_text('{"rung": "w4a4"}')
    cfg = str(both_harness_pkg / "pyproject.toml")

    # (a) the ENDPOINT harness. `run` wraps the handler's return in its
    # {"event": "result", "value": ...} stdout envelope.
    assert main(["run", "--config", cfg, "--payload", '{"rung": "w4a4"}']) == 0
    envelope = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert envelope["event"] == "result"
    endpoint_out = envelope["value"]

    # (b) the JOB harness — same body, same payload, no hub either way
    assert main(
        ["job", "run", "bake", "--config", cfg, "--payload", str(payload)]
    ) == 0
    job_out = json.loads(capsys.readouterr().out)

    assert endpoint_out["rung"] == job_out["rung"] == "w4a4"
    # Both harnesses ran the monotonic-position surface, under the same name.
    assert endpoint_out["positions"] == job_out["positions"] == 2


def test_a_job_may_not_be_a_class() -> None:
    with pytest.raises(TypeError, match="module-level FUNCTION"):
        job(  # type: ignore[call-overload]
            resources=Resources()
        )(type("Trainer", (), {}))


def test_a_job_may_not_live_inside_a_class() -> None:
    with pytest.raises(TypeError, match="declared inside class"):
        class Holder:
            @job
            def train(self, ctx: JobContext, spec: BakeIn) -> BakeOut:
                raise AssertionError("never runs")


def test_a_job_may_not_be_a_generator() -> None:
    with pytest.raises(TypeError, match="must not be a generator"):
        @job
        def streamer(ctx: JobContext, spec: BakeIn) -> Any:
            yield BakeOut(rung="x", ctx_class="")


def test_a_job_takes_ctx_payload_and_declared_families_only() -> None:
    """pgw#1294: An UNDECLARED extra parameter is still refused."""
    with pytest.raises(TypeError, match=r"plus declared family instances"):
        @job
        def three(ctx: JobContext, spec: BakeIn, extra: int) -> BakeOut:
            raise AssertionError("never runs")


def test_a_job_declares_struct_payload_and_result() -> None:
    with pytest.raises(TypeError, match="must be annotated with a msgspec"):
        @job
        def loose(ctx: JobContext, spec: dict) -> BakeOut:  # type: ignore[type-arg]
            raise AssertionError("never runs")

    with pytest.raises(TypeError, match="return type must be a msgspec.Struct"):
        @job
        def loose_out(ctx: JobContext, spec: BakeIn) -> dict:  # type: ignore[type-arg]
            raise AssertionError("never runs")


def test_visibility_is_private_by_default_and_a_closed_vocabulary() -> None:
    @job
    def defaulted(ctx: JobContext, spec: BakeIn) -> BakeOut:
        raise AssertionError("never runs")

    spec = extract_job_spec(defaulted)
    assert spec is not None and spec.visibility == "private"
    with pytest.raises(ValueError, match="visibility must be one of"):
        job(visibility="world-readable")


def _ctx(*, publishes: bool, kind: str = "job") -> JobContext:
    return JobContext(
        request_id="r-1294", job_id="j-1294", publishes=publishes,
        execution_hints={"kind": kind},
    )


def test_undeclared_publish_is_refused_typed_under_both_harnesses(
    tmp_path: Path,
) -> None:
    """pgw#1294: One declaration surface, so one refusal — reached from the @job side and from the @endpoint sid..."""
    blob = tmp_path / "adapter.safetensors"
    blob.write_bytes(b"\x00" * 16)

    for kind in ("job", "inference"):
        ctx = _ctx(publishes=False, kind=kind)
        with pytest.raises(PublishNotDeclaredError) as caught:
            ctx.save_checkpoint("org/repo", blob)
        assert caught.value.surface == "save_checkpoint"
        assert "publishes=True" in str(caught.value)
        with pytest.raises(PublishNotDeclaredError):
            ctx.open_checkpoint_stream("org/repo")


def test_publish_flavors_refuses_undeclared_before_it_reads_anything() -> None:
    from gen_worker.convert.publish import publish_flavors

    with pytest.raises(PublishNotDeclaredError) as caught:
        publish_flavors(_ctx(publishes=False), [], destination_repo="org/repo")
    assert caught.value.surface == "publish_flavors"


def test_a_declared_publisher_gets_past_the_declaration_gate(
    tmp_path: Path,
) -> None:
    """pgw#1294: The same call with the declaration is no longer refused HERE — it goes on to the repo-scope/tra..."""
    blob = tmp_path / "adapter.safetensors"
    blob.write_bytes(b"\x00" * 16)
    ctx = _ctx(publishes=True)
    ctx.open_checkpoint_stream("org/repo")  # no PublishNotDeclaredError


def test_producer_kinds_still_publish_but_are_told_they_are_on_borrowed_time(
    tmp_path: Path,
) -> None:
    """TRANSITIONAL (th#2052 deletes it): kind still implies write authority for the un-migrated fleet, and ever..."""
    emitted: list[Dict[str, Any]] = []
    ctx = JobContext(
        request_id="r-legacy", job_id="j-legacy", publishes=False,
        execution_hints={"kind": "conversion"}, emitter=emitted.append,
    )
    ctx.open_checkpoint_stream("org/repo")  # admitted by kind, not declaration
    warnings = [e for e in emitted if e["type"] == "request.log"]
    assert warnings and "th#2052" in warnings[0]["payload"]["message"]


class PubIn(msgspec.Struct):
    pass


class PubOut(msgspec.Struct):
    declared: bool


@job(publishes=True, name="declared-publisher")
def declared_publisher(ctx: JobContext, spec: PubIn) -> PubOut:
    return PubOut(declared=ctx.publishes)


@job(name="silent-job")
def silent_job(ctx: JobContext, spec: PubIn) -> PubOut:
    return PubOut(declared=ctx.publishes)


def _run(fn: Any, ctx: JobContext) -> Any:
    spec = extract_job_spec(fn)
    assert spec is not None
    return execute_job(
        JobDispatch(job_name=spec.name, payload=msgspec.msgpack.encode(PubIn())),
        jobs={spec.name: spec}, ctx=ctx, reraise=True,
    )


def test_execute_job_stamps_the_declaration_so_a_dispatch_head_cannot_forget() -> None:
    """pgw#1294: The JobSpec is the ONE home for `publishes`; the context flag is a projection of it."""
    ctx = _ctx(publishes=False)          # head forgot to pass it
    outcome = _run(declared_publisher, ctx)
    assert msgspec.msgpack.decode(outcome.result)["declared"] is True
    assert ctx.publishes is True


def test_a_caller_may_not_grant_authority_the_release_never_declared() -> None:
    """pgw#1294: The opposite direction is a refusal, not a silent downgrade: the hub minted no write grant for ..."""
    with pytest.raises(ValueError, match="declaration is the release's"):
        _run(silent_job, _ctx(publishes=True))


def test_position_going_backwards_raises_rather_than_lying() -> None:
    ctx = _ctx(publishes=False)
    ctx.progress(position=10, total=100, phase="download")
    ctx.progress(position=10, total=100, phase="download")   # flat is legal
    ctx.progress(position=64, total=100, phase="download")
    with pytest.raises(NonMonotonicProgressError) as caught:
        ctx.progress(position=63, total=100, phase="download")
    assert caught.value.phase == "download"
    assert (caught.value.last, caught.value.attempted) == (64.0, 63.0)
    # A NEW phase restarts the count — that is how a job says "next stage".
    ctx.progress(position=0, total=100, phase="upload")
    assert ctx.position("download") == 64.0
    assert ctx.position("upload") == 0.0


def test_both_spellings_of_one_quantity_may_not_disagree() -> None:
    ctx = _ctx(publishes=False)
    with pytest.raises(ValueError, match="position= and step="):
        ctx.progress(position=1, step=2, phase="p")
    with pytest.raises(ValueError, match="phase= and stage="):
        ctx.progress(position=1, phase="a", stage="b")


def test_the_emitted_payload_is_the_shape_the_HUB_parses() -> None:
    """RECONCILED against th#2050's landed `forkJobProgress` + `runtimestore.ParseRequestProgressPayload`, which..."""
    emitted: list[Dict[str, Any]] = []
    ctx = JobContext(request_id="r", emitter=emitted.append)
    ctx.progress(position=4096.5, total=31_000_000, phase="download")
    payload = emitted[-1]["payload"]
    assert payload["step"] == 4096          # what the hub reads
    assert payload["stage"] == "download"   # what the hub reads
    assert payload["total"] == 31_000_000   # what the hub reads
    assert payload["position"] == 4096.5    # the exact value, beside it


def test_metric_emits_the_name_value_rows_the_hub_ingests() -> None:
    """th#2050's job-metric arm reads `name` and `value` off the payload and DROPS anything else, so one event p..."""
    emitted: list[Dict[str, Any]] = []
    ctx = JobContext(request_id="r", emitter=emitted.append)
    ctx.metric({"loss": 0.31, "cosine": 0.998}, step=120, total=2000)
    rows = [e["payload"] for e in emitted if e["type"] == "request.metric"]
    assert [(r["name"], r["value"]) for r in rows] == [
        ("cosine", 0.998), ("loss", 0.31),
    ]
    assert all(r["step"] == 120 and r["total"] == 2000 for r in rows)


def test_positions_ride_the_CTX_EVENT_channel_never_the_output_channel() -> None:
    """th#2050 landed `JobProgress` as a STREAMING OUTPUT chunk and put job liveness on the ctx-event envelope i..."""
    from gen_worker.executor import EVENT_CONTENT_TYPE

    # Verbatim from tensorhub `runtimestore.RequestEventContentType`.
    assert EVENT_CONTENT_TYPE == "application/x-request-event+json"

    # The output channel's content types, from executor._encode_chunk. None of
    # them may collide with the ctx-event channel, or an output chunk would be
    # parsed as a position (and vice versa).
    output_content_types = {
        "text/plain", "application/x-batch-item+msgpack", "application/json",
    }
    assert EVENT_CONTENT_TYPE not in output_content_types

    # And a job structurally has NO output-chunk path at all: a generator body
    # is refused at decoration, so there is nothing that could stream one.
    with pytest.raises(TypeError, match="must not be a generator"):
        @job
        def streams(ctx: JobContext, spec: BakeIn) -> Any:
            yield BakeOut(rung="x", ctx_class="")


def test_the_request_spelling_is_untouched() -> None:
    """pgw#1294: Portability cuts both ways: an existing endpoint body must keep working, byte for byte."""
    emitted: list[Dict[str, Any]] = []
    ctx = JobContext(request_id="r", emitter=emitted.append)
    ctx.progress(0.5, "denoise", step=5, total=20)
    payload = emitted[-1]["payload"]
    assert payload["progress"] == 0.5
    assert payload["stage"] == "denoise"
    assert payload["step"] == 5 and payload["total"] == 20
    # ...and the same call fed the position ledger the stall watch reads.
    assert ctx.position("denoise") == 5.0


def test_the_positional_job_form_is_refused_instead_of_reinterpreted() -> None:
    """pgw#1294: `ctx.progress(4096, 31_000_000, "download")` would have to guess which quantity the second argu..."""
    ctx = _ctx(publishes=False)
    # Called through getattr: these are DELIBERATE misuses, and a type checker
    # rejecting them statically is half the guarantee — the runtime refusal
    # below is the other half, for callers with no type checker.
    misuse = getattr(ctx, "progress")
    # Two positional args: the SDK's own refusal, naming the keyword form.
    with pytest.raises(TypeError, match="second positional argument"):
        misuse(4096, 31_000_000)
    # Three: Python refuses it before we get a chance to, which is the same
    # answer — the point is that nothing reinterprets the argument by type.
    with pytest.raises(TypeError):
        misuse(4096, 31_000_000, "download")


class SilentIn(msgspec.Struct):
    report: bool = False


class SilentOut(msgspec.Struct):
    moved: int


@job(name="transfer")
def transfer(ctx: JobContext, spec: SilentIn) -> SilentOut:
    """pgw#1294: A transfer loop."""
    moved = 0
    for _chunk in range(4):
        moved += 64
        if spec.report:
            ctx.progress(position=moved, total=256, phase="download")
    if not spec.report:
        await_progress(
            lambda: ctx.cancelled,
            lambda seen: bool(seen),
            what="the progress watch to cancel the stalled run",
            cadence=Cadence(),
        )
    return SilentOut(moved=moved)


_TRANSFER_JOBS = {"transfer": extract_job_spec(transfer)}


def _run_transfer(*, report: bool, budget_s: float) -> Any:
    ctx = _ctx(publishes=False)
    dispatch = JobDispatch(
        job_name="transfer",
        payload=msgspec.msgpack.encode(SilentIn(report=report)),
        phase_budget_s=budget_s,
    )
    return execute_job(dispatch, jobs=_TRANSFER_JOBS, ctx=ctx)  # type: ignore[arg-type]


def test_a_transfer_that_reports_nothing_fails_the_job() -> None:
    """pgw#1294: Silence for the whole budget is the fault, and the body RETURNING afterwards does not launder i..."""
    outcome = _run_transfer(report=False, budget_s=0.05)
    assert outcome.status == "failed"
    assert outcome.error_type == "JobProgressStalledError"


def test_the_same_loop_reporting_position_succeeds() -> None:
    """pgw#1294: The identical body, reporting, under the PRODUCTION default budget."""
    outcome = _run_transfer(report=True, budget_s=DEFAULT_PHASE_BUDGET_S)
    assert outcome.status == "succeeded"
    assert msgspec.msgpack.decode(outcome.result)["moved"] == 256


def test_a_stalled_context_is_cancelled_and_the_error_names_the_phase() -> None:
    ctx = _ctx(publishes=False)
    ctx.progress(position=64, total=256, phase="download")
    with ProgressWatch(ctx, budget_s=0.05, poll_s=0.01) as watch:
        # Waited on PROGRESS, never on a clock (pgw#795): the only success is
        # the verdict arriving, and a wait that never advances dies at the
        # harness floor naming what it last saw.
        await_progress(
            lambda: watch.stalled,
            lambda seen: seen is not None,
            what="the progress watch to judge the phase stalled",
            cadence=Cadence(),
            render=lambda seen: "no verdict yet" if seen is None else str(seen),
        )
    assert watch.stalled is not None
    assert watch.stalled.phase == "download"
    assert ctx.cancelled  # production-owned: the run is told, not just logged
    with pytest.raises(JobProgressStalledError):
        watch.check()


def test_run_once_is_stated_as_data_on_the_outcome() -> None:
    """pgw#1294: Nothing survives in-process between jobs by contract, so the outcome tells its driver to recycl..."""
    assert _run_transfer(report=True, budget_s=0.05).recycle_child is True


@pytest.fixture()
def manifest_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg = tmp_path / "manifest_job"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent("""
        import msgspec
        from gen_worker import JobContext, RequestContext, Resources, endpoint, job

        class In_(msgspec.Struct):
            steps: int = 10

        class Out_(msgspec.Struct):
            ok: bool

        @job(resources=Resources(vcpus=4), env=("HF_TOKEN",),
             resumable=True, publishes=True)
        def zebra_bake(ctx: JobContext, spec: In_) -> Out_:
            return Out_(ok=True)

        @job
        def alpha_plan(ctx: JobContext, spec: In_) -> Out_:
            return Out_(ok=True)

        @endpoint
        class Gen:
            def generate(self, ctx: RequestContext, p: In_) -> Out_:
                return Out_(ok=True)
    """))
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "manifest-job"\nversion = "0.0.0"\n'
        '[tool.gen_worker]\nmain = "manifest_job.main"\n'
    )
    return tmp_path


def test_jobs_ride_the_manifest_beside_functions(manifest_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_jobs

    jobs = discover_jobs(manifest_pkg, main_module="manifest_job.main")
    # Sorted by name: the manifest is a published artifact, so the block is
    # byte-stable across runs.
    assert [j["name"] for j in jobs] == ["alpha-plan", "zebra-bake"]
    assert jobs == discover_jobs(manifest_pkg, main_module="manifest_job.main")

    bake = jobs[1]
    assert bake["resources"]["vcpus"] == 4
    assert bake["env"] == ["HF_TOKEN"]
    assert bake["resumable"] is True
    assert bake["visibility"] == "private"
    assert bake["publishes"] is True
    assert bake["payload_schema_sha256"] and bake["output_schema_sha256"]
    # A POINTER into the release tarball, never a copy of the bytes
    # (RECONCILED to th#2049's landed correction 6: a `source` text field would
    # be a second copy that can only drift).
    assert bake["source_file"] == "manifest_job/main.py"
    assert "source" not in bake
    # A job declares no lanes, no compiled graph, no slots — deliberately.
    assert not {"execution_lanes", "compile", "slots"} & set(bake)

    assert jobs[0]["publishes"] is False   # never omitted; see below


def test_publishes_is_emitted_on_the_function_row_too(manifest_pkg: Path) -> None:
    """pgw#1294: Both row shapes, ALWAYS emitted: the hub mints a write grant off this, so 'absent' must mean 'w..."""
    from gen_worker.discovery.discover import discover_functions

    fns = discover_functions(manifest_pkg, main_module="manifest_job.main")
    assert [f["publishes"] for f in fns] == [False]


def test_the_full_manifest_carries_a_jobs_block_beside_functions(
    manifest_pkg: Path,
) -> None:
    """One package may carry BOTH; publish once, submit as needed."""
    from gen_worker.discovery.discover import discover_manifest

    manifest = discover_manifest(manifest_pkg)
    assert [j["name"] for j in manifest["jobs"]] == ["alpha-plan", "zebra-bake"]
    assert [f["name"] for f in manifest["functions"]] == ["generate"]
    assert manifest["functions"][0]["publishes"] is False


# ============================================================================
# pgw#1336 — pgw#1336 / pgw#1307 arm (8): the RunJob `compat-*` minter is
#   GONE.
# ============================================================================

JOB_ID = "11111111-2222-3333-4444-555555555555"


HUB_GOAL_ID = f"job-{JOB_ID}"


HUB_INTENT_ID = f"job-{JOB_ID}-0"


class In(msgspec.Struct):
    rung: str = "w8a8"


class Out(msgspec.Struct):
    rung: str


_SEEN: Dict[str, Any] = {}


@job(name="plan-h3-svdq", resources=Resources(vcpus=4))
def plan_h3_svdq(ctx: JobContext, payload: In) -> Out:
    _SEEN["request_id"] = ctx.request_id
    return Out(rung=payload.rung)


def _serve(ctx: Any, payload: In) -> Out:
    return Out(rung=payload.rung)


def _job_specs() -> List[Any]:
    return [extract_job_spec(plan_h3_svdq)]


def _executor(send: Any) -> Executor:
    return Executor(
        [EndpointSpec(
            name="generate", method=_serve, kind="inference",
            payload_type=In, output_mode="single",
        )],
        send,
        jobs=_job_specs(),
    )


class _Harness:
    """A real Executor with both tables populated and a bound registry."""

    def __init__(self) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self.ex = _executor(self._send)
        self.ex._process_exit = lambda code: None
        self.registry = IntentRegistry("release-1", ["generate"])
        self.ex.bind_intent_registry(self.registry)

    async def _send(self, msg: pb.WorkerMessage) -> None:
        self.sent.append(msg)

    def job_frame(self, name: str = "plan-h3-svdq", **kw: Any) -> pb.RunJob:
        """A JOB dispatch, stamped as `JobWire.Dispatch` stamps it hub-side."""
        kw.setdefault("intent_kind", pb.DESIRED_INTENT_KIND_RUN_JOB)
        kw.setdefault("intent_id", HUB_INTENT_ID)
        kw.setdefault("goal_id", HUB_GOAL_ID)
        return self.request_frame(name, **kw)

    def request_frame(self, name: str = "generate", **kw: Any) -> pb.RunJob:
        """A SERVED-REQUEST dispatch: no kind, no carrier — as the hub sends."""
        return pb.RunJob(
            request_id=kw.pop("request_id", JOB_ID),
            attempt=int(kw.pop("attempt", 0)),
            function_name=name,
            input_payload=msgspec.msgpack.encode(In()),
            **kw,
        )

    async def dispatch(self, run: pb.RunJob) -> Optional[pb.JobResult]:
        await self.ex.handle_run_job(run)
        record = self.ex.jobs.get((run.request_id, run.attempt))
        if record is not None and record.task is not None:
            await record.task
        results = self.results()
        return results[-1] if results else None

    def results(self) -> List[pb.JobResult]:
        return [m.job_result for m in self.sent
                if m.WhichOneof("msg") == "job_result"]

    def intent_ids(self) -> List[str]:
        return list(self.registry._intents)


@pytest.fixture(autouse=True)
def _clean() -> Any:
    _SEEN.clear()
    yield
    _SEEN.clear()


def test_a_job_dispatch_reports_against_the_hub_authored_carrier() -> None:
    """pgw#1336: The id the worker reports is the id the HUB authored — not a hash of (request_id, attempt) the ..."""

    async def scenario() -> Tuple[Optional[pb.JobResult], List[str], str]:
        h = _Harness()
        run = h.job_frame()
        result = await h.dispatch(run)
        return result, h.intent_ids(), h.registry._intents[HUB_INTENT_ID].goal_id

    result, intent_ids, goal_id = asyncio.run(scenario())
    assert result is not None and result.status == pb.JOB_STATUS_OK
    assert HUB_INTENT_ID in intent_ids
    assert goal_id == HUB_GOAL_ID
    # THE DELETION, asserted as an absence: nothing in this registry was
    # fabricated for the job. This is the row that reds if the minter comes
    # back, whatever else still passes.
    assert not [i for i in intent_ids if i.startswith("compat-")], intent_ids


def test_an_adopted_carrier_is_never_renamed_on_a_redelivery() -> None:
    """pgw#1336: The compat minter appended `-N` when it found the id taken, because the id was ITS OWN to choos..."""
    registry = IntentRegistry("release-1", [])
    first = registry.adopt_dispatch_intent(HUB_INTENT_ID, HUB_GOAL_ID)
    registry.transition(
        first,
        pb.LIFECYCLE_INTENT_STATUS_FAILED,
        pb.LIFECYCLE_INTENT_STAGE_FINALIZING,
    )
    again = registry.adopt_dispatch_intent(HUB_INTENT_ID, HUB_GOAL_ID)
    assert again == first == HUB_INTENT_ID
    assert list(registry._intents) == [HUB_INTENT_ID]
    # Replaced, not left terminal: a live obligation must be transitionable.
    state = registry._intents[HUB_INTENT_ID]
    assert state.status == pb.LIFECYCLE_INTENT_STATUS_ACCEPTED


def test_the_same_name_routes_differently_with_and_without_the_kind() -> None:
    """pgw#1336: THE DISTINGUISHABILITY ROW."""

    async def scenario() -> Tuple[Optional[pb.JobResult], Optional[pb.JobResult]]:
        declared = _Harness()
        as_job = await declared.dispatch(declared.job_frame("plan-h3-svdq"))
        undeclared = _Harness()
        as_request = await undeclared.dispatch(
            undeclared.request_frame("plan-h3-svdq"))
        return as_job, as_request

    as_job, as_request = asyncio.run(scenario())
    assert as_job is not None and as_job.status == pb.JOB_STATUS_OK
    # The SAME name, without the kind, is a served request — and this release
    # declares no endpoint by that name.
    assert as_request is not None
    assert as_request.status == pb.JOB_STATUS_INVALID
    # And the refusal SAYS which way it crossed, so a submitter is not told to
    # rename a name that is perfectly correct.
    assert "declared in this release as a @job" in as_request.safe_message
    assert "asked for an @endpoint" in as_request.safe_message


def test_an_endpoint_name_dispatched_as_a_job_is_refused_the_other_way() -> None:
    async def scenario() -> Optional[pb.JobResult]:
        h = _Harness()
        return await h.dispatch(h.job_frame("generate"))

    result = asyncio.run(scenario())
    assert result is not None and result.status == pb.JOB_STATUS_INVALID
    assert "declared in this release as an @endpoint" in result.safe_message
    assert "asked for a @job" in result.safe_message


def test_a_run_job_frame_with_no_carrier_cannot_be_papered_over() -> None:
    """pgw#1336: The old code could always invent an id, so a hub bug here was invisible."""
    registry = IntentRegistry("release-1", [])
    with pytest.raises(ValueError, match="hub-authored intent id"):
        registry.adopt_dispatch_intent("", HUB_GOAL_ID)
    with pytest.raises(ValueError, match="hub-authored intent id"):
        registry.adopt_dispatch_intent("   ", HUB_GOAL_ID)
    assert list(registry._intents) == []


def test_a_served_request_still_mints_its_worker_local_carrier() -> None:
    """pgw#1336: NOT a leftover."""

    async def scenario() -> List[str]:
        h = _Harness()
        await h.dispatch(h.request_frame("generate"))
        return h.intent_ids()

    intent_ids = asyncio.run(scenario())
    assert [i for i in intent_ids if i.startswith("compat-job-")], intent_ids


def test_ensure_intent_still_mints_a_carrier_for_uncommanded_work() -> None:
    """pgw#1307 arm (8), verbatim: this twin *"SURVIVES and arm (8) says so explicitly"*."""
    registry = IntentRegistry("release-1", ["generate"])
    intent_id = registry.ensure_intent(
        pb.DESIRED_INTENT_KIND_FUNCTION_READY, function_name="generate")
    assert intent_id.startswith("compat-"), intent_id
    assert registry.is_active(intent_id)
    # And it is REPORTABLE — the property the whole arm exists for.
    registry.transition(
        intent_id,
        pb.LIFECYCLE_INTENT_STATUS_RUNNING,
        pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
    )
    assert registry._intents[intent_id].status == pb.LIFECYCLE_INTENT_STATUS_RUNNING


def test_a_blockerless_waiting_report_still_carries_a_deadline() -> None:
    """`_WAITING_DEADLINE_FALLBACK_MS`, re-read against `phase_budget_s` and KEPT (pgw#1336)."""
    registry = IntentRegistry("release-1", [])
    intent_id = registry.adopt_dispatch_intent(HUB_INTENT_ID, HUB_GOAL_ID)
    registry.transition(
        intent_id,
        pb.LIFECYCLE_INTENT_STATUS_WAITING,
        pb.LIFECYCLE_INTENT_STAGE_WAIT_LOAD_LOCK,
        reason=pb.LIFECYCLE_WAIT_REASON_SINGLE_FLIGHT_OWNER,
    )
    state = registry._intents[intent_id]
    assert state.deadline_at_unix_ms > 0


def _spy_budget(monkeypatch: pytest.MonkeyPatch) -> List[JobDispatch]:
    seen: List[JobDispatch] = []

    def spy(dispatch: JobDispatch, **kw: Any) -> Any:
        seen.append(dispatch)
        return _real_execute_job(dispatch, **kw)

    monkeypatch.setattr("gen_worker.executor.execute_job", spy)
    return seen


def test_a_stated_phase_budget_replaces_the_wheels_compiled_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1336: One question — "is this job advancing?" — had two numbers: the hub's liveness sweep read the op..."""
    seen = _spy_budget(monkeypatch)

    async def scenario() -> None:
        h = _Harness()
        await h.dispatch(h.job_frame(phase_budget_s=90))

    asyncio.run(scenario())
    assert len(seen) == 1
    assert seen[0].phase_budget_s == 90.0
    assert seen[0].phase_budget_s != DEFAULT_PHASE_BUDGET_S


def test_no_stated_budget_keeps_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0 is "no instruction", not "no budget" — an unbounded phase is exactly what made pgw#1287's silent downlo..."""
    seen = _spy_budget(monkeypatch)

    async def scenario() -> None:
        h = _Harness()
        await h.dispatch(h.job_frame())

    asyncio.run(scenario())
    assert len(seen) == 1
    assert seen[0].phase_budget_s == DEFAULT_PHASE_BUDGET_S


# ============================================================================
# th#2069 — ``@job(emits_media=True)`` — the media sibling of ``publishes``.
# ============================================================================

class In_th2069(msgspec.Struct):
    pass


class Out_th2069(msgspec.Struct):
    declared: bool = False


@job(emits_media=True, name="declared-media")
def declared_media(ctx: JobContext, spec: In_th2069) -> Out_th2069:
    ctx.save_bytes("outputs/report.json", b"{}")
    return Out_th2069(declared=ctx.emits_media)


@job(name="silent-media")
def silent_media(ctx: JobContext, spec: In_th2069) -> Out_th2069:
    ctx.save_bytes("outputs/report.json", b"{}")
    return Out_th2069(declared=ctx.emits_media)


def _ctx_th2069(tmp_path: Path, *, emits_media: Any = None) -> JobContext:
    return JobContext(
        request_id="r-2069", job_id="j-2069",
        emits_media=emits_media,
        local_output_dir=str(tmp_path),
        execution_hints={"kind": "job"},
    )


def _run_th2069(fn: Any, ctx: JobContext) -> Any:
    spec = extract_job_spec(fn)
    assert spec is not None
    return execute_job(
        JobDispatch(job_name=spec.name, payload=msgspec.msgpack.encode(In_th2069())),
        jobs={spec.name: spec}, ctx=ctx, reraise=True,
    )


def test_the_declaration_reaches_the_spec_and_defaults_off() -> None:
    assert extract_job_spec(declared_media).emits_media is True  # type: ignore[union-attr]
    assert extract_job_spec(silent_media).emits_media is False  # type: ignore[union-attr]


def test_it_is_independent_of_publishes() -> None:
    """th#2069: An eval job writes media and NO repo; a quality matrix writes both."""

    @job(emits_media=True, publishes=False, name="eval-only")
    def eval_only(ctx: JobContext, spec: In_th2069) -> Out_th2069:
        return Out_th2069()

    @job(emits_media=True, publishes=True, name="matrix")
    def matrix(ctx: JobContext, spec: In_th2069) -> Out_th2069:
        return Out_th2069()

    assert (extract_job_spec(eval_only).emits_media,  # type: ignore[union-attr]
            extract_job_spec(eval_only).publishes) == (True, False)  # type: ignore[union-attr]
    assert (extract_job_spec(matrix).emits_media,  # type: ignore[union-attr]
            extract_job_spec(matrix).publishes) == (True, True)  # type: ignore[union-attr]


def test_the_manifest_row_carries_it_because_that_is_what_the_hub_reads() -> None:
    from gen_worker.discovery.discover import _job_entry

    row = _job_entry(extract_job_spec(declared_media), Path.cwd())
    assert row["emits_media"] is True
    assert _job_entry(extract_job_spec(silent_media), Path.cwd())["emits_media"] is False


def test_an_undeclared_job_is_refused_before_a_byte_moves(tmp_path: Path) -> None:
    with pytest.raises(MediaNotDeclaredError) as caught:
        _run_th2069(silent_media, _ctx_th2069(tmp_path))
    assert caught.value.surface == "save_bytes"
    assert "emits_media=True" in str(caught.value)
    assert not list(tmp_path.rglob("report.json"))


def test_a_declared_job_mints_the_media(tmp_path: Path) -> None:
    outcome = _run_th2069(declared_media, _ctx_th2069(tmp_path))
    assert msgspec.msgpack.decode(outcome.result)["declared"] is True
    assert [p.name for p in tmp_path.rglob("report.json")] == ["report.json"]


def test_save_file_is_fenced_too(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"\x00" * 8)
    ctx = _ctx_th2069(tmp_path / "out", emits_media=False)
    with pytest.raises(MediaNotDeclaredError) as caught:
        ctx.save_file("outputs/copy.bin", src)
    assert caught.value.surface == "save_file"


def test_an_endpoint_declares_nothing_and_writes_media(tmp_path: Path) -> None:
    """th#2069: The gate is JOBS-ONLY."""
    ctx = _ctx_th2069(tmp_path)
    assert ctx.emits_media is True
    ctx.save_bytes("outputs/image.webp", b"\x00" * 4)


def test_the_result_envelope_is_not_media(tmp_path: Path) -> None:
    """th#2069: Worker->orchestrator transport rides no media grant, so an undeclared job still returns its resu..."""
    ctx = _ctx_th2069(tmp_path, emits_media=False)
    ctx._save_result_envelope("results/r-2069.msgpack", b"\x00" * 4)


def test_execute_job_stamps_it_so_a_dispatch_head_cannot_forget(
    tmp_path: Path,
) -> None:
    ctx = _ctx_th2069(tmp_path)              # head passed nothing
    _run_th2069(declared_media, ctx)
    assert ctx.emits_media is True


def test_a_caller_may_not_grant_media_authority_the_release_never_declared(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="declaration is the release's"):
        _run_th2069(silent_media, _ctx_th2069(tmp_path, emits_media=True))
