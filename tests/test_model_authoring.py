"""The Model/Endpoint split (pgw#1382) — the model-authoring acceptance suite.

Integration, no mocks, no GPU: the main_v2-shaped fixture imports clean and
extracts statically; a fake-checkpoint load+serve drives the ONE merged
entrypoint end-to-end through a real Model instance under single-flight;
mutation-scope discipline holds (the scheduler-leak class of bug is the red
arm); the unload contract is drain-then-call, best-effort-never-correctness.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import msgspec
import pytest

from gen_worker import (
    Adapter,
    DYNAMIC,
    DistillationAdapter,
    LoadContext,
    Model,
    RequestContext,
    STATIC,
    Structural,
    entrypoint,
    lane,
)
from gen_worker.demand import GiB, MiB, const, per_mp_batch
from gen_worker.models import SDXL
from gen_worker.serving import (
    DefaultsError,
    DeployBinding,
    EndpointHost,
    ServeDispatchError,
    EntrypointDeclarationError,
    LaneDeclarationError,
    ModelDeclarationError,
    lane_handle,
    load_endpoint,
    model_declared_lanes,
    model_lanes,
    model_marks_compile,
    model_requires,
    model_shapes,
    model_structural,
    model_type,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_v2_endpoint"
RT_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_rt_endpoint"
LANE = "sdxl.diffusers-bf16@1"


def make_checkpoint(tmp_path: Path, **config: object) -> Path:
    root = tmp_path / "checkpoint"
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"seed": 7, "scheduler": {}, **config}))
    return root


def make_binding(tmp_path: Path, **kwargs: object) -> DeployBinding:
    kwargs.setdefault("checkpoint_ref", "ckpt:tiny@1")
    kwargs.setdefault("checkpoint_dir", make_checkpoint(tmp_path))
    kwargs.setdefault("model", "sdxl")
    kwargs.setdefault(
        "defaults",
        {"steps": {"default": 2, "lo": 1, "hi": 8}},
    )
    return DeployBinding(**kwargs)  # type: ignore[arg-type]


# --- static extraction: import clean, read the whole surface ----------------


def test_fixture_imports_clean_and_extracts_the_declared_surface() -> None:
    loaded = load_endpoint(FIXTURE_DIR)
    from serving_v2_fixture.main import ImageOutput, SdxlModel, TextToImageInput

    # One merged entrypoint (Paul's merge ruling): the deployment decides
    # turbo vs CFG; the caller sees one function.
    assert sorted(loaded.entrypoints) == ["generate"]
    spec = loaded.entrypoints["generate"]
    assert spec.payload_type is TextToImageInput
    assert spec.return_type is ImageOutput

    # Slots, in declaration order: model slot + optional adapter slot + the
    # request's adapter-list slot; the PARAM NAME is the slot name.
    assert [(s.name, s.kind, s.required) for s in spec.slots] == [
        ("model", "model", True),
        ("turbo", "adapter", False),
        ("loras", "adapters", False),
    ]
    # The annotation records the slot KIND (Paul's structural guard): the
    # turbo slot takes only distillation-marked adapters; the request list
    # takes any Adapter.
    assert spec.slots[1].annotation is DistillationAdapter
    assert spec.slots[2].annotation is Adapter
    assert spec.model_classes == (SdxlModel,)
    assert loaded.models == (SdxlModel,)

    # The class header is the single declaration source: model type via the
    # generic, lanes via the class kwarg — read, not executed.
    assert model_type(SdxlModel) is SDXL
    lanes = model_lanes(SdxlModel)
    assert [lane.contract for lane in lanes] == [
        "sdxl.diffusers-bf16@1", "cozy.sdxl-fp8-rowwise@1",
    ]
    assert len(lanes) == 2

    # pgw#1599: what each lane declares is its own DEMAND FORMULA — read
    # statically, so a deployment is sized without running author code. The
    # formulas DIFFER per lane and that is the point: the fp8 lane's weights
    # and activations are both smaller, so one per-model number would have
    # been wrong for one of the two.
    per_lane = {
        row.contract_id: row.request.coefficients()
        for row in model_declared_lanes(SdxlModel)
    }
    assert per_lane == {
        "sdxl.diffusers-bf16@1": {"const": MiB(96), "mp_batch": MiB(24)},
        "cozy.sdxl-fp8-rowwise@1": {"const": MiB(48), "mp_batch": MiB(12)},
    }
    # The placement row carries ONLY the floor DERIVED from the contract
    # dtype, and is ABSENT for a lane that derives none — both fixture
    # stand-ins declare float32, which has no capability floor, so the honest
    # answer is no row at all rather than an empty one (an empty row reads to
    # the resolver as "runs anywhere", which is th#1754's shape).
    assert model_requires(SdxlModel) == {}


def _sdxl_contract() -> Any:
    from gen_worker._vendor.tensorfs import contracts

    return contracts.SDXL_DIFFUSERS_BF16


def _engine_marks(ctx: Any) -> None:
    """A DELEGATED compile mark: real, and invisible to the AST reader."""

    ctx.compile(object())


def _lane() -> Any:
    """A minimal but REAL lane declaration for a specimen class."""

    return lane(request=const(GiB(1)))


def test_model_header_declarations_and_refusals() -> None:
    """pgw#1599's acceptance (a) and (b): every header either declares REAL
    lanes with a demand formula, or is REFUSED at class-definition time with
    a message naming what is missing. Five vocabularies die here."""

    with pytest.raises(ModelDeclarationError, match=r"Model\[SDXL\]"):
        type("Bare", (Model,), {})

    # --- lanes= is REQUIRED, real contracts only (Paul's ruling pair) ------

    with pytest.raises(ModelDeclarationError, match="lanes= is REQUIRED"):
        class NoLanes(Model[SDXL]):
            pass

    with pytest.raises(ModelDeclarationError, match="lanes= is EMPTY"):
        class EmptyLanes(Model[SDXL], lanes={}):
            pass

    # DELETED (1/5): `lanes=()` and the whole DerivedLane machinery. It was
    # the "I state no layout contract" spelling, and a derived lane names no
    # layout document — so it could answer neither checkpoint compatibility
    # nor lane selection, which is what a lane is FOR.
    with pytest.raises(ModelDeclarationError, match="tuple form is deleted"):
        class TupleLanes(Model[SDXL], lanes=()):  # type: ignore[arg-type]
            pass

    with pytest.raises(ModelDeclarationError, match="tuple form is deleted"):
        class TupleReal(
            Model[SDXL],
            lanes=(_sdxl_contract(),),  # type: ignore[arg-type]
        ):
            pass

    with pytest.raises(ModelDeclarationError, match="MAPPING of tensorfs"):
        class BadLanes(Model[SDXL], lanes=["not-a-mapping"]):  # type: ignore[arg-type]
            pass

    with pytest.raises(ModelDeclarationError, match="not a layout contract"):
        class StringLane(Model[SDXL], lanes={"sdxl.diffusers-bf16@1": _lane()}):
            pass

    # DELETED (2/5): the canonical-contract BORROW. An omitted `lanes=` used
    # to silently adopt `ModelType.canonical_contract`; the attribute is gone
    # from every model type, so the omission is a refusal (above) rather than
    # a layout claim the author never made.
    assert not hasattr(SDXL, "canonical_contract")

    # DELETED (3/5): the floor STRING. Paul, 2026-08-20: "there is no
    # required VRAM" — demand varies per request, so a lane declares a
    # FORMULA, not a number.
    with pytest.raises(ModelDeclarationError, match="machine-floor STRING"):
        class FloorString(Model[SDXL], lanes={_sdxl_contract(): "vram7g"}):
            pass

    with pytest.raises(ModelDeclarationError, match=r"lane\(request="):
        class NoSpec(Model[SDXL], lanes={_sdxl_contract(): None}):
            pass

    # DELETED (4/5): `eager_only=`. Compilation participation IS the presence
    # of `ctx.compile` marks — no keyword, and none accepted.
    with pytest.raises(ModelDeclarationError, match="eager_only.*is DELETED"):
        class EagerOnly(
            Model[SDXL],
            lanes={_sdxl_contract(): _lane()},
            eager_only="the fixture compiles nothing on purpose",
        ):
            pass

    # `requires=` (deleted earlier, pgw#1404) still refuses by name.
    with pytest.raises(ModelDeclarationError, match="requires.*is DELETED"):
        class StrayFloor(
            Model[SDXL],
            lanes={_sdxl_contract(): _lane()},
            requires={"sdxl.other@1": "vram8g"},
        ):
            pass

    # --- the GREEN header, and everything it makes readable ---------------

    class SdxlLike(
        Model[SDXL],
        lanes={_sdxl_contract(): lane(
            request=const(GiB(1.2)) + per_mp_batch(MiB(220)),
            resident=("vae",),
        )},
        structural={"timestep_dtype": Structural(
            field="scheduler",
            classes={"int64": "dpmpp_2m_karras", "float32": "euler"},
            measured="pgw#1572, CPU: set_timesteps(20) on each served scheduler",
        )},
        shapes={"aspect": STATIC},
    ):
        def load(self, ctx: object) -> None:
            self.unet = ctx.compile(object())  # type: ignore[attr-defined]

    assert model_type(SdxlLike) is SDXL
    (declared,) = model_declared_lanes(SdxlLike)
    assert declared.contract_id == "sdxl.diffusers-bf16@1"
    assert declared.dtype == "bfloat16"
    # DERIVED from the contract dtype, never hand-written — and per LANE,
    # which is why one hand-written floor could not serve a multi-lane class.
    assert declared.min_sm == 80
    assert declared.request.coefficients() == {
        "const": GiB(1.2), "mp_batch": MiB(220),
    }
    assert declared.resident == ("vae",)
    assert model_shapes(SdxlLike) == {"aspect": STATIC}
    assert [d.as_document(a) for a, d in model_structural(SdxlLike).items()] == [{
        "axis": "timestep_dtype",
        "declared": ["int64", "float32"],
        "from": "scheduler",
        "representatives": ["dpmpp_2m_karras", "euler"],
        "measured": "pgw#1572, CPU: set_timesteps(20) on each served scheduler",
    }]
    # The placement row carries the DERIVED floor and nothing else: the VRAM
    # half went with the strings, and what replaces it is COMPUTED from the
    # formula (pgw#1600), never annotated.
    assert {h: r.render() for h, r in model_requires(SdxlLike).items()} == {
        "sdxl.diffusers-bf16@1": "sm80+",
    }

    # A hand-written sm floor is still a refusal — two producers of one fact
    # is how they drift apart (Paul, 2026-08-18).
    with pytest.raises(ModelDeclarationError, match="lane\\(request="):
        class HandWrittenSm(Model[SDXL], lanes={_sdxl_contract(): "sm90+"}):
            pass

    # --- compilation participation is the MARK, and nothing else ----------

    assert model_marks_compile(SdxlLike) is True

    # An EAGER model under the new rules: real lanes like everyone else, zero
    # `ctx.compile` calls, no keyword anywhere. This is the whole declaration.
    class EagerModel(Model[SDXL], lanes={_sdxl_contract(): _lane()}):
        def load(self, ctx: object) -> None:
            """No ctx.compile(): torch.compile measured no win here."""

    assert model_marks_compile(EagerModel) is False
    assert model_lanes(EagerModel) == (_sdxl_contract(),)
    # ...and the words in that DOCSTRING are not a mark. A class that
    # documents "no ctx.compile here" is the best-behaved one on the fleet and
    # a substring check would refuse exactly it.

    # --- fork axes: declared, or refused --------------------------------

    # DELETED (5/5): derive's global DYNAMIC_AXES flag. The choice is per
    # MODEL now, and a compiling class that states none is refused rather
    # than defaulted.
    from gen_worker.release import derive as _derive
    assert not hasattr(_derive, "DYNAMIC_AXES")

    with pytest.raises(LaneDeclarationError, match="shapes= is REQUIRED"):
        class NoShapes(Model[SDXL], lanes={_sdxl_contract(): _lane()}):
            def load(self, ctx: object) -> None:
                ctx.compile(object())  # type: ignore[attr-defined]

    # CFG/batch is a PERMANENTLY STATIC shape fork (Paul, 2026-08-20) — not
    # declarable at all, in either direction.
    with pytest.raises(LaneDeclarationError, match="PERMANENTLY STATIC"):
        class BatchDynamic(
            Model[SDXL],
            lanes={_sdxl_contract(): _lane()},
            shapes={"aspect": STATIC, "batch": DYNAMIC},
        ):
            def load(self, ctx: object) -> None:
                ctx.compile(object())  # type: ignore[attr-defined]

    # BOTH shape declarations are expressible and NEITHER is presumed
    # (acceptance (d)): the same model, declared the other way.
    class DynamicAspect(
        Model[SDXL],
        lanes={_sdxl_contract(): _lane()},
        shapes={"aspect": DYNAMIC},
    ):
        def load(self, ctx: object) -> None:
            ctx.compile(object())  # type: ignore[attr-defined]

    assert model_shapes(DynamicAspect) == {"aspect": DYNAMIC}

    # A DELEGATED mark (`self.engine.compile_dit(ctx)`) is invisible to the
    # AST reader but produces real graphs, so shapes= is PERMITTED — never
    # refused — on a class the reader calls unmarked. Refusing would make a
    # correct endpoint undeclarable to buy a tidiness check.
    class DelegatedMark(
        Model[SDXL],
        lanes={_sdxl_contract(): _lane()},
        shapes={"aspect": STATIC},
    ):
        def load(self, ctx: object) -> None:
            _engine_marks(ctx)

    assert model_marks_compile(DelegatedMark) is False  # the reader's limit
    assert model_shapes(DelegatedMark) == {"aspect": STATIC}  # declared anyway

    # A structural axis with one variant class is not a fork; a declared fork
    # with no measurement behind it is a guess wearing a declaration.
    with pytest.raises(LaneDeclarationError, match="at least TWO variant"):
        class OneVariant(
            Model[SDXL],
            lanes={_sdxl_contract(): _lane()},
            structural={"t": Structural(
                field="scheduler", classes={"int64": "ddim"}, measured="m")},
        ):
            pass

    with pytest.raises(LaneDeclarationError, match="measured=. is MANDATORY"):
        class Unmeasured(
            Model[SDXL],
            lanes={_sdxl_contract(): _lane()},
            structural={"t": Structural(
                field="scheduler",
                classes={"a": "ddim", "b": "euler"},
                measured="   ",
            )},
        ):
            pass

    # --- the lane handle's wire shape (unchanged by this issue) ----------

    # The SHIPPED tensorfs Contract's attribute shape: `stamp` + `digest`,
    # and NO `contract` (tensorfs#111). Reading the bare 64-hex `digest` as
    # the handle put `f1455f56…` where `sdxl.diffusers-bf16@1` belonged and
    # made torchcg refuse the lane. This is the producer half of a cross-repo
    # wire agreement, so the shape is pinned here.
    class ShippedContract:
        __slots__ = ("digest", "dtype", "name", "stamp", "version")

        def __init__(self) -> None:
            self.name = "sdxl.diffusers-bf16"
            self.version = 1
            self.stamp = "sdxl.diffusers-bf16@1"
            self.digest = "f1455f56321d1f268772912c223170f015564ac0" + "0" * 24
            self.dtype = None

    assert lane_handle(ShippedContract()) == "sdxl.diffusers-bf16@1"

    # A digest-only object still yields a STAMP, never a bare hex string:
    # an anonymous custom contract's digest IS its stamp (tensorfs#112's third
    # Stamp arm), and that spelling carries the `sha256:` prefix.
    class AnonymousContract:
        __slots__ = ("digest", "dtype")

        def __init__(self) -> None:
            self.digest = "a" * 64
            self.dtype = None

    assert lane_handle(AnonymousContract()) == "sha256:" + "a" * 64

    # Cheap __init__: constructing a model does not load anything.
    assert not vars(EagerModel())


# Malformed-signature specimens: defined at module level (the contract is
# module-level functions), decorated inside the test so the refusal is
# observable rather than an import-time explosion.


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    ok: bool = True


class _M(Model[SDXL], lanes={_sdxl_contract(): _lane()}):
    pass


def _payload_first(payload: _In, model: _M, ctx: RequestContext) -> _Out:
    return _Out()


def _bad_payload(ctx: RequestContext, payload: int, model: _M) -> _Out:
    return _Out()


def _no_model(ctx: RequestContext, payload: _In, turbo: Adapter | None) -> _Out:
    return _Out()


def _bare_model(ctx: RequestContext, payload: _In, model: Model) -> _Out:
    return _Out()


def _optional_model(ctx: RequestContext, payload: _In, model: Optional[_M]) -> _Out:
    return _Out()


def _bad_list(ctx: RequestContext, payload: _In, model: _M,
              loras: list[int]) -> _Out:
    return _Out()


def _bad_return(ctx: RequestContext, payload: _In, model: _M) -> int:
    return 0


def _too_few(ctx: RequestContext) -> _Out:
    return _Out()


def _weightless(ctx: RequestContext, payload: _In) -> _Out:
    """pgw#1392: the shipped `(ctx, payload) -> Out` workflow-helper shape."""
    return _Out()


def _junk_slot(ctx: RequestContext, payload: _In, junk: int) -> _Out:
    return _Out()


def _kw_only(ctx: RequestContext, payload: _In, *, model: _M) -> _Out:
    return _Out()


class _Grouping:
    def method(self, ctx: RequestContext, payload: _In, model: _M) -> _Out:
        return _Out()


def test_malformed_entrypoint_signatures_refuse_typed() -> None:
    cases: list[tuple[Callable[..., Any], str]] = [
        (_payload_first, "ctx comes FIRST"),   # old payload-first order
        (_bad_payload, "payload"),
        (_bare_model, "bare Model base"),
        (_optional_model, r"`Adapter \| None`"),
        (_bad_list, r"`list\[Adapter\]`"),
        (_bad_return, "return type"),
        (_too_few, "takes 1 parameters"),
        (_kw_only, "keyword"),
        (_Grouping.method, "MODULE-LEVEL"),
    ]
    for fn, pattern in cases:
        with pytest.raises(EntrypointDeclarationError, match=pattern):
            entrypoint(fn)


def test_zero_model_slots_is_legal_pgw1392() -> None:
    """pgw#1392: model-less entrypoints are LEGAL (se#757 blocker C).

    `_no_model` used to be a refusal case here ("declares no model slot").
    Ten shipped production functions have that signature, so the floor
    dropped to zero. Junk slots did NOT become legal -- the loop above
    still pins every other refusal, `_junk_slot` included."""

    spec = getattr(entrypoint(_no_model), "__cozy_entrypoint__")
    assert [slot.kind for slot in spec.slots] == ["adapter"]
    assert spec.model_params == ()

    spec = getattr(entrypoint(_weightless), "__cozy_entrypoint__")
    assert spec.slots == ()
    assert spec.model_classes == ()

    with pytest.raises(EntrypointDeclarationError, match="must be a model slot"):
        entrypoint(_junk_slot)


# --- the ctx split ----------------------------------------------------------


def test_load_context_defaults_decode_matrix(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)

    def ctx_for(defaults: dict) -> LoadContext:
        return LoadContext(
            binding=DeployBinding(
                checkpoint_ref="ckpt:x@1", checkpoint_dir=checkpoint,
                model="sdxl", defaults=defaults,
            ),
            model_type=SDXL,
        )

    # Row absent -> the platform fallbacks (zero-arg struct = trace fixture).
    fallback = ctx_for({}).defaults()
    assert fallback == SDXL.Defaults()
    assert fallback.cfg is True and fallback.steps.default == 28

    # Partial row -> field-level overlay; untouched fields keep fallbacks.
    # (No scheduler field: checkpoints carry no scheduler metadata — the
    # tree IS their choice; scheduler demands are ADAPTER metadata.)
    decoded = ctx_for(
        {"cfg": False, "step_distilled": True, "timesteps": [8, 6, 4, 2]}
    ).defaults()
    assert decoded.cfg is False
    assert decoded.step_distilled is True
    assert decoded.timesteps == (8, 6, 4, 2)
    assert decoded.guidance == SDXL.Defaults().guidance

    # Ill-typed value -> typed refusal naming the checkpoint, never coercion.
    with pytest.raises(DefaultsError, match="ckpt:x@1"):
        ctx_for({"cfg": "definitely"}).defaults()


def test_request_context_facts() -> None:
    bare: RequestContext[Any] = RequestContext("req-1")
    with pytest.raises(RuntimeError, match="no deploy binding"):
        _ = bare.checkpoint_ref
    # There is deliberately NO trace flag on ctx (Paul ruling): author code
    # is trace-oblivious by construction.
    assert not hasattr(bare, "is_trace")

    bound: RequestContext[Any] = RequestContext(
        "req-2",
        binding=DeployBinding(checkpoint_ref="ckpt:y@2", checkpoint_dir=Path(".")),
    )
    assert bound.checkpoint_ref == "ckpt:y@2"
    # The salvaged base surface rides along (clamp records caller-visibly).
    assert bound.clamp("x", 5.0, hi=2.0) == 2.0
    assert bound.adjustments and bound.adjustments[0]["field"] == "x"
    # ctx.warn: the caller-visible warning channel — same delivery path as
    # clamp notes (the adjustment ledger), accumulated per-request.
    bound.warn("first")
    bound.warn("second")
    assert bound.warnings == ("first", "second")
    assert len(bound.adjustments) == 3  # clamp row + two warn rows, one ledger


# --- end-to-end: fake-checkpoint load + serve, mutation scopes, turbo -------


@pytest.fixture()
def host(tmp_path: Path) -> EndpointHost:
    loaded = load_endpoint(FIXTURE_DIR)
    booted = EndpointHost(
        loaded, make_binding(tmp_path), lane_contract=LANE,
        output_dir=tmp_path / "outputs",
    )
    booted.setup()
    return booted


def fixture_model(host: EndpointHost) -> Any:
    (instance,) = host.instances.values()
    return instance.model


def test_scheduler_and_adapter_scopes_restore_after_a_turbo_request(
    host: EndpointHost, tmp_path: Path
) -> None:
    """THE leak bug as a test: the pre-split file swapped
    ``self.pipe.scheduler`` for a turbo request and never restored it, so one
    turbo request poisoned every later CFG request on the persistent
    instance. Model-owned scopes make the restore structural."""
    from diffusers import EulerDiscreteScheduler

    model = fixture_model(host)
    baseline = model.pipe.scheduler
    assert isinstance(baseline, EulerDiscreteScheduler)

    binding = host.binding
    host.rebind(
        DeployBinding(
            checkpoint_ref=binding.checkpoint_ref,
            checkpoint_dir=binding.checkpoint_dir,
            model=binding.model,
            defaults=binding.defaults,
            adapter=DistillationAdapter(
                name="lcm-lora", path=tmp_path / "lora",
                defaults=SDXL.Lora.Defaults(scheduler="lcm"),
                ref="cozy/lcm-lora@1",
            ),
        )
    )
    out = host.dispatch("generate", {"prompt": "turbo"}, request_id="turbo-1")
    assert [used.ref for used in out.loras] == ["cozy/lcm-lora@1"]

    # Leave it as you found it: the SAME scheduler object is back, and the
    # adapter is unloaded — configuration equals the post-load baseline.
    assert model.pipe.scheduler is baseline
    assert model.pipe.loaded_loras == []

    # The next CFG request runs on the baseline scheduler (the leaked
    # LCM/trailing scheduler is the unrepresentable state).
    host.rebind(binding)
    out = host.dispatch("generate", {"prompt": "cfg"}, request_id="cfg-1")
    assert out.model == binding.checkpoint_ref and out.loras == []
    assert model.pipe.scheduler is baseline


def test_scopes_restore_even_when_the_request_raises(
    host: EndpointHost, tmp_path: Path
) -> None:
    model = fixture_model(host)
    baseline = model.pipe.scheduler
    binding = host.binding
    host.rebind(
        DeployBinding(
            checkpoint_ref=binding.checkpoint_ref,
            checkpoint_dir=binding.checkpoint_dir,
            model=binding.model,
            defaults=binding.defaults,
            adapter=DistillationAdapter(
                name="x", path=tmp_path / "lora", defaults=SDXL.Lora.Defaults(),
            ),
        )
    )
    # The pipeline raises INSIDE both scopes (adapter loaded, scheduler
    # swapped): the `finally` restores on the way out.
    with pytest.raises(RuntimeError, match="exploded mid-request"):
        host.dispatch("generate", {"prompt": "explode"}, request_id="r")
    assert model.pipe.scheduler is baseline
    assert model.pipe.loaded_loras == []


def test_stacking_on_a_step_distilled_checkpoint_warns_and_ignores(
    tmp_path: Path,
) -> None:
    """Paul's either-or ruling, warn-shaped: a step-distillation adapter on a
    step-distilled checkpoint is IGNORED caller-visibly — the checkpoint
    serves as deployed, never a fried render and never an aborted request."""
    loaded = load_endpoint(FIXTURE_DIR)
    stacked = EndpointHost(
        loaded,
        DeployBinding(
            checkpoint_ref="ckpt:distilled@1",
            checkpoint_dir=make_checkpoint(tmp_path),
            model="sdxl",
            defaults={"cfg": False, "step_distilled": True,
                      "steps": {"default": 4, "lo": 1, "hi": 8}},
            adapter=DistillationAdapter(
                name="x", path=tmp_path / "lora", defaults=SDXL.Lora.Defaults(),
                ref="cozy/x@1",
            ),
        ),
        lane_contract=LANE,
        output_dir=tmp_path / "outputs",
    )
    stacked.setup()
    ctx = stacked.make_context("r")
    out = stacked.dispatch("generate", {"prompt": "x"}, request_id="r", ctx=ctx)
    assert out.model == "ckpt:distilled@1" and out.loras == []
    assert [w for w in ctx.warnings if "already step-distilled" in w]


def test_distilled_checkpoint_serves_turbo_without_an_adapter(
    host: EndpointHost, tmp_path: Path
) -> None:
    binding = host.binding
    fresh = EndpointHost(
        host.loaded,
        DeployBinding(
            checkpoint_ref="ckpt:distilled@1",
            checkpoint_dir=binding.checkpoint_dir,
            model="sdxl",
            defaults={
                "cfg": False, "step_distilled": True,
                "timesteps": [8, 6, 4, 2],
                "steps": {"default": 4, "lo": 1, "hi": 8},
            },
        ),
        lane_contract=LANE,
        output_dir=tmp_path / "outputs-distilled",
    )
    fresh.setup()
    out = fresh.dispatch("generate", {"prompt": "fast"}, request_id="r")
    assert out.model == "ckpt:distilled@1" and out.loras == []


# --- the concurrency + lifecycle contract (runtime fixture) -----------------


@pytest.fixture()
def rt_host(tmp_path: Path) -> Iterator[EndpointHost]:
    loaded = load_endpoint(RT_FIXTURE_DIR)
    import serving_rt_fixture.main as rt

    rt.reset()
    booted = EndpointHost(loaded, make_binding(tmp_path, defaults={}))
    booted.setup()
    yield booted
    rt.RELEASE.set()  # never leave a held request stuck on teardown


def test_single_flight_per_model_instance(rt_host: EndpointHost) -> None:
    import serving_rt_fixture.main as rt

    results: list = []
    a = threading.Thread(
        target=lambda: results.append(
            rt_host.dispatch("run", {"value": 1, "hold": True}, request_id="a")),
        daemon=True,
    )
    a.start()
    assert rt.ENTERED.acquire(timeout=30)  # A is inside the model

    b = threading.Thread(
        target=lambda: results.append(
            rt_host.dispatch("run", {"value": 2, "hold": True}, request_id="b")),
        daemon=True,
    )
    b.start()
    # Single-flight: B must NOT enter the model while A holds the admission.
    # (Negative probe: in a broken build B enters immediately and this
    # acquire succeeds — the red arm.)
    assert not rt.ENTERED.acquire(timeout=0.4)

    rt.RELEASE.set()
    a.join(timeout=30)
    b.join(timeout=30)
    assert not a.is_alive() and not b.is_alive()
    assert rt.HIGH_WATER == 1
    assert sorted(r.value for r in results) == [2, 4]


def test_eviction_drains_in_flight_requests_before_unload(
    rt_host: EndpointHost,
) -> None:
    import serving_rt_fixture.main as rt
    from serving_rt_fixture.main import SlowModel

    request = threading.Thread(
        target=lambda: rt_host.dispatch(
            "run", {"value": 3, "hold": True}, request_id="held"),
        daemon=True,
    )
    request.start()
    assert rt.ENTERED.acquire(timeout=30)

    evictor = threading.Thread(target=lambda: rt_host.evict(SlowModel), daemon=True)
    evictor.start()
    rt.RELEASE.set()
    request.join(timeout=30)
    evictor.join(timeout=30)
    assert not request.is_alive() and not evictor.is_alive()

    # Drain-then-call: the request finished BEFORE unload ran.
    assert rt.ORDER.index("request_done") < rt.ORDER.index("unload:SlowModel")
    assert SlowModel not in rt_host.instances


def test_failing_unload_is_logged_and_eviction_proceeds(
    rt_host: EndpointHost, caplog: pytest.LogCaptureFixture
) -> None:
    import serving_rt_fixture.main as rt
    from serving_rt_fixture.main import BrokenUnloadModel

    out = rt_host.dispatch("broken", {"value": 5}, request_id="r")
    assert out.value == 10
    with caplog.at_level(logging.ERROR):
        rt_host.evict(BrokenUnloadModel)
    # Best-effort, never correctness: the exception is logged, the instance
    # is gone — a failing unload cannot pin residency.
    assert BrokenUnloadModel not in rt_host.instances
    assert "unload:BrokenUnloadModel" in rt.ORDER
    assert any("eviction proceeds" in r.message for r in caplog.records)


def test_multi_model_entrypoint_fills_slots_in_declaration_order(
    rt_host: EndpointHost,
) -> None:
    out = rt_host.dispatch("pair", {"value": 9}, request_id="r")
    assert out.value == 9
    assert out.served_by == "SlowModel+OtherModel"


def test_per_request_loras_ride_the_list_slot_and_restore(
    host: EndpointHost, tmp_path: Path
) -> None:
    """Paul's per-request LoRA-list promotion: `loras: list[Adapter]` is the
    request's picks; scales apply via the plural model scope; full restore."""
    model = fixture_model(host)
    picks = [
        Adapter(name="style-ink", path=tmp_path / "ink",
                defaults=SDXL.Lora.Defaults(), scale=0.7, ref="me/style-ink@3"),
        Adapter(name="char-fox", path=tmp_path / "fox",
                defaults=SDXL.Lora.Defaults(), scale=1.2, ref="me/char-fox@1"),
    ]

    out = host.dispatch(
        "generate", {"prompt": "styled"}, request_id="r", loras=picks)
    # Structured output evidence: pinned refs + applied scales, in
    # application order — fields, never string grammar (Paul/ie#731).
    assert out.model == "ckpt:tiny@1"
    assert [(used.ref, used.scale) for used in out.loras] == [
        ("me/style-ink@3", 0.7), ("me/char-fox@1", 1.2),
    ]
    # The envelope scales were applied through the plural scope...
    assert model.pipe.adapter_history == [
        [("style-ink", 0.7), ("char-fox", 1.2)],
    ]
    # ...and nothing stays loaded or active after the request (full restore).
    assert model.pipe.loaded_loras == []
    assert model.pipe.active_adapters == []


def test_scheduler_precedence_and_the_distillation_slot_kind_guard(
    host: EndpointHost, tmp_path: Path
) -> None:
    model = fixture_model(host)
    baseline = model.pipe.scheduler

    # Layer 1: the request's pick swaps the scheduler for the call, restored.
    out = host.dispatch(
        "generate", {"prompt": "x", "scheduler": "lcm"}, request_id="r1")
    assert out.model == "ckpt:tiny@1"
    assert model.pipe.scheduler is baseline  # restored after the request

    # Unsupported REQUEST scheduler: typed 400 at the API boundary.
    with pytest.raises(msgspec.ValidationError):
        host.dispatch(
            "generate", {"prompt": "x", "scheduler": "not-a-scheduler"},
            request_id="r2")

    # Layer 2: the applied distillation adapter's DEMAND — and a demand this
    # endpoint does not serve warns and falls through to the checkpoint's.
    binding = host.binding
    demanding = EndpointHost(
        host.loaded,
        DeployBinding(
            checkpoint_ref=binding.checkpoint_ref,
            checkpoint_dir=binding.checkpoint_dir,
            model=binding.model,
            defaults=binding.defaults,
            adapter=DistillationAdapter(
                name="odd", path=tmp_path / "lora",
                defaults=SDXL.Lora.Defaults(scheduler="unipc"),
                ref="cozy/odd@1",
            ),
        ),
        lane_contract=LANE,
        output_dir=tmp_path / "outputs-sched",
    )
    demanding.setup()
    ctx = demanding.make_context("r3")
    demanding.dispatch("generate", {"prompt": "x"}, request_id="r3", ctx=ctx)
    assert [w for w in ctx.warnings if "does not serve" in w]
    # Relay check: the adapter's decoded overlay is SDXL.Lora.Defaults,
    # reached through the DistillationAdapter subclass.
    assert isinstance(demanding.binding.adapter, DistillationAdapter)
    assert isinstance(demanding.binding.adapter.defaults, SDXL.Lora.Defaults)

    # THE SLOT-KIND GUARD: a plain (style) Adapter bound where the
    # entrypoint declares a DistillationAdapter slot is a typed refusal
    # BEFORE author code runs — takeover power is typed, not positional.
    misdeployed = EndpointHost(
        host.loaded,
        DeployBinding(
            checkpoint_ref=binding.checkpoint_ref,
            checkpoint_dir=binding.checkpoint_dir,
            model=binding.model,
            defaults=binding.defaults,
            adapter=Adapter(
                name="style", path=tmp_path / "style",
                defaults=SDXL.Lora.Defaults(), ref="me/style@1",
            ),
        ),
        lane_contract=LANE,
        output_dir=tmp_path / "outputs-guard",
    )
    misdeployed.setup()
    with pytest.raises(ServeDispatchError, match="not distillation-marked"):
        misdeployed.dispatch("generate", {"prompt": "x"}, request_id="r4")
