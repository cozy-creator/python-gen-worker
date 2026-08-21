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
#: The deploy's lane pin, in the WIRE spelling `"<topology>+<quant>"`
#: (pgw#1621) — what a `DeployBinding` carries and what `loader.lane()`
#: matches on.
LANE = "sdxl.diffusers@1+plain.bf16@1"


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


def test_fixture_imports_clean_and_extracts_the_declared_surface() -> None:
    loaded = load_endpoint(FIXTURE_DIR)
    from serving_v2_fixture.main import ImageOutput, SdxlModel, TextToImageInput

    assert sorted(loaded.entrypoints) == ["generate"]
    spec = loaded.entrypoints["generate"]
    assert spec.payload_type is TextToImageInput
    assert spec.return_type is ImageOutput

    assert [(s.name, s.kind, s.required) for s in spec.slots] == [
        ("model", "model", True),
        ("turbo", "adapter", False),
        ("loras", "adapters", False),
    ]
    assert spec.slots[1].annotation is DistillationAdapter
    assert spec.slots[2].annotation is Adapter
    assert spec.model_classes == (SdxlModel,)
    assert loaded.models == (SdxlModel,)

    assert model_type(SdxlModel) is SDXL
    lanes = model_lanes(SdxlModel)
    assert [row.render() for row in lanes] == [
        "sdxl.diffusers@1+plain.bf16@1",
        "sdxl.diffusers@1+cozy.fp8-rowwise@1",
    ]
    assert len(lanes) == 2

    per_lane = {
        row.contract_id: row.request.coefficients()
        for row in model_declared_lanes(SdxlModel)
    }
    assert per_lane == {
        "sdxl.diffusers@1+plain.bf16@1": {"const": MiB(96), "mp_batch": MiB(24)},
        "sdxl.diffusers@1+cozy.fp8-rowwise@1": {
            "const": MiB(48), "mp_batch": MiB(12),
        },
    }
    # The placement row carries ONLY the floor DERIVED from the lane's QUANT
    # RULE (`capability_floor_sm`, read off the ratified document), and is
    # ABSENT for a lane whose rule states none.
    #
    # This assertion CHANGED with pgw#1621 and the change is a fact, not a
    # relaxation: the fixture's lanes used to be `LaneRef(handle,
    # dtype=torch.float32)` stand-ins — a handle plus a dtype the fixture
    # PICKED — so both derived 0 and the honest answer was no row at all. A v2
    # lane cannot pick its dtype: it names a ratified rule and the rule states
    # 80 for bf16 and 89 for fp8-rowwise. The "no row rather than an empty
    # row" property (an empty row reads to the resolver as "runs anywhere",
    # th#1754's shape) is asserted on `plain.f32@1` in the release fixtures,
    # which is the one ratified rule whose floor really is 0.
    assert {h: r.render() for h, r in model_requires(SdxlModel).items()} == {
        "sdxl.diffusers@1+plain.bf16@1": "sm80+",
        "sdxl.diffusers@1+cozy.fp8-rowwise@1": "sm89+",
    }


#: pgw#1621: a lane is the `(topology, quant)` STAMP PAIR, both halves
#: ratified documents in the vendored `spec/v2` corpus. The v1 `Contract`
#: OBJECT this helper used to import is deleted with the v1 vocabulary.
_SDXL_BF16 = ("sdxl.diffusers@1", "plain.bf16@1")
_SDXL_BF16_ID = "sdxl.diffusers@1+plain.bf16@1"


def _sdxl_contract() -> Any:
    return _SDXL_BF16


def _engine_marks(ctx: Any) -> None:

    ctx.compile(object())


def _lane() -> Any:

    return lane(request=const(GiB(1)))


def test_model_header_declarations_and_refusals() -> None:

    with pytest.raises(ModelDeclarationError, match=r"Model\[SDXL\]"):
        type("Bare", (Model,), {})

    with pytest.raises(ModelDeclarationError, match="lanes= is REQUIRED"):
        class NoLanes(Model[SDXL]):
            pass

    with pytest.raises(ModelDeclarationError, match="lanes= is EMPTY"):
        class EmptyLanes(Model[SDXL], lanes={}):
            pass

    with pytest.raises(ModelDeclarationError, match="tuple form is deleted"):
        class TupleLanes(Model[SDXL], lanes=()):  # type: ignore[arg-type]
            pass

    with pytest.raises(ModelDeclarationError, match="tuple form is deleted"):
        class TupleReal(
            Model[SDXL],
            lanes=(_sdxl_contract(),),  # type: ignore[arg-type]
        ):
            pass

    with pytest.raises(ModelDeclarationError, match="MAPPING of"):
        class BadLanes(Model[SDXL], lanes=["not-a-mapping"]):  # type: ignore[arg-type]
            pass

    # A BARE handle is not a lane: a lane is named by a PAIR, and the refusal
    # says so rather than guessing which axis was meant. An OLD v1 spelling
    # gets the display-name hint and still has to be rewritten as the pair —
    # there is no alias resolution, deliberately (a spelling that resolves is
    # a spelling that spreads, and the hub's bridge for the un-re-keyed fleet
    # exists to be deleted).
    with pytest.raises(ModelDeclarationError, match="a lane is named by a"):
        class StringLane(Model[SDXL], lanes={"sdxl.diffusers-bf16@1": _lane()}):
            pass

    with pytest.raises(ModelDeclarationError,
                       match="used to name 'sdxl.diffusers@1\\+plain.bf16@1'"):
        class OldSpelling(Model[SDXL], lanes={"sdxl.diffusers-bf16@1": _lane()}):
            pass

    # Both halves are REQUIRED and both are checked against the corpus.
    with pytest.raises(ModelDeclarationError,
                       match="topology 'nope.nothing@1' is not in the vendored"):
        class UnknownTopology(
            Model[SDXL], lanes={("nope.nothing@1", "plain.bf16@1"): _lane()}
        ):
            pass

    with pytest.raises(ModelDeclarationError,
                       match="quant 'nope.q9@1' is not in the vendored"):
        class UnknownRule(
            Model[SDXL], lanes={("sdxl.diffusers@1", "nope.q9@1"): _lane()}
        ):
            pass

    assert not hasattr(SDXL, "canonical_contract")

    with pytest.raises(ModelDeclarationError, match="machine-floor STRING"):
        class FloorString(Model[SDXL], lanes={_sdxl_contract(): "vram7g"}):
            pass

    with pytest.raises(ModelDeclarationError, match=r"lane\(request="):
        class NoSpec(Model[SDXL], lanes={_sdxl_contract(): None}):
            pass

    with pytest.raises(ModelDeclarationError, match="eager_only.*is DELETED"):
        class EagerOnly(
            Model[SDXL],
            lanes={_sdxl_contract(): _lane()},
            eager_only="the fixture compiles nothing on purpose",
        ):
            pass

    with pytest.raises(ModelDeclarationError, match="requires.*is DELETED"):
        class StrayFloor(
            Model[SDXL],
            lanes={_sdxl_contract(): _lane()},
            requires={"sdxl.other@1": "vram8g"},
        ):
            pass

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
    assert declared.contract_id == _SDXL_BF16_ID
    assert (declared.topology, declared.quant) == _SDXL_BF16
    assert declared.dtype == "bfloat16"
    # The v1 spelling survives as a DISPLAY name — carried for refusal
    # messages, never parsed and gating nothing.
    assert declared.display_name == "sdxl.diffusers-bf16@1"
    # DERIVED from the lane's QUANT RULE, never hand-written — and per LANE,
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
    assert {h: r.render() for h, r in model_requires(SdxlLike).items()} == {
        _SDXL_BF16_ID: "sm80+",
    }

    with pytest.raises(ModelDeclarationError, match="lane\\(request="):
        class HandWrittenSm(Model[SDXL], lanes={_sdxl_contract(): "sm90+"}):
            pass

    assert model_marks_compile(SdxlLike) is True

    class EagerModel(Model[SDXL], lanes={_sdxl_contract(): _lane()}):
        def load(self, ctx: object) -> None:
            """No ctx.compile(): torch.compile measured no win here."""

    assert model_marks_compile(EagerModel) is False
    # `model_lanes` answers the PARSED pair (a `LayoutId`), not the author's
    # tuple: the stamp is read once, at class definition, and every consumer
    # shares that read rather than re-parsing the header's spelling.
    assert [row.render() for row in model_lanes(EagerModel)] == [_SDXL_BF16_ID]

    from gen_worker.release import derive as _derive
    assert not hasattr(_derive, "DYNAMIC_AXES")

    with pytest.raises(LaneDeclarationError, match="shapes= is REQUIRED"):
        class NoShapes(Model[SDXL], lanes={_sdxl_contract(): _lane()}):
            def load(self, ctx: object) -> None:
                ctx.compile(object())  # type: ignore[attr-defined]

    with pytest.raises(LaneDeclarationError, match="PERMANENTLY STATIC"):
        class BatchDynamic(
            Model[SDXL],
            lanes={_sdxl_contract(): _lane()},
            shapes={"aspect": STATIC, "batch": DYNAMIC},
        ):
            def load(self, ctx: object) -> None:
                ctx.compile(object())  # type: ignore[attr-defined]

    class DynamicAspect(
        Model[SDXL],
        lanes={_sdxl_contract(): _lane()},
        shapes={"aspect": DYNAMIC},
    ):
        def load(self, ctx: object) -> None:
            ctx.compile(object())  # type: ignore[attr-defined]

    assert model_shapes(DynamicAspect) == {"aspect": DYNAMIC}

    class DelegatedMark(
        Model[SDXL],
        lanes={_sdxl_contract(): _lane()},
        shapes={"aspect": STATIC},
    ):
        def load(self, ctx: object) -> None:
            _engine_marks(ctx)

    assert model_marks_compile(DelegatedMark) is False
    assert model_shapes(DelegatedMark) == {"aspect": STATIC}

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

    # --- the lane handle's wire shape ------------------------------------

    # ONE rendering, `"<topology>+<quant>"` — th#1809's spelling, shared with
    # the hub's `tensorfs.LayoutID.String` and with the derived-artifact CAS
    # address, so a drift here is a fork rather than a cosmetic bug.
    #
    # pgw#1621 deleted the four-fallback ladder this used to be, along with the
    # ambiguity that needed it: a v1 `Contract` spelled its handle four ways,
    # one of which was a BARE 64-hex `digest` that read as a handle and was not
    # one — it once put `f1455f56…` where `sdxl.diffusers-bf16@1` belonged and
    # made torchcg refuse the lane. A pair has exactly one rendering, so every
    # spelling that reaches this repo answers the same string.
    from gen_worker.models.tensor_layout_contract import LayoutId

    (row,) = model_declared_lanes(SdxlLike)
    for spelling in (
        row,                                    # the DeclaredLane
        row.layout,                             # the LayoutId
        _SDXL_BF16,                             # the author's two-tuple
        _SDXL_BF16_ID,                          # the wire string, read back
        LayoutId(topology="sdxl.diffusers@1", quant="plain.bf16@1"),
    ):
        assert lane_handle(spelling) == _SDXL_BF16_ID

    # A 64-hex digest is NOT a lane handle and can no longer masquerade as
    # one: it is not a pair, so it refuses by name where it used to render.
    with pytest.raises(ModelDeclarationError):
        lane_handle("a" * 64)

    assert not vars(EagerModel())


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
        (_payload_first, "ctx comes FIRST"),
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

    spec = getattr(entrypoint(_no_model), "__cozy_entrypoint__")
    assert [slot.kind for slot in spec.slots] == ["adapter"]
    assert spec.model_params == ()

    spec = getattr(entrypoint(_weightless), "__cozy_entrypoint__")
    assert spec.slots == ()
    assert spec.model_classes == ()

    with pytest.raises(EntrypointDeclarationError, match="must be a model slot"):
        entrypoint(_junk_slot)


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

    fallback = ctx_for({}).defaults()
    assert fallback == SDXL.Defaults()
    assert fallback.cfg is True and fallback.steps.default == 28

    decoded = ctx_for(
        {"cfg": False, "step_distilled": True, "timesteps": [8, 6, 4, 2]}
    ).defaults()
    assert decoded.cfg is False
    assert decoded.step_distilled is True
    assert decoded.timesteps == (8, 6, 4, 2)
    assert decoded.guidance == SDXL.Defaults().guidance

    with pytest.raises(DefaultsError, match="ckpt:x@1"):
        ctx_for({"cfg": "definitely"}).defaults()


def test_request_context_facts() -> None:
    bare: RequestContext[Any] = RequestContext("req-1")
    with pytest.raises(RuntimeError, match="no deploy binding"):
        _ = bare.checkpoint_ref
    assert not hasattr(bare, "is_trace")

    bound: RequestContext[Any] = RequestContext(
        "req-2",
        binding=DeployBinding(checkpoint_ref="ckpt:y@2", checkpoint_dir=Path(".")),
    )
    assert bound.checkpoint_ref == "ckpt:y@2"
    assert bound.clamp("x", 5.0, hi=2.0) == 2.0
    assert bound.adjustments and bound.adjustments[0]["field"] == "x"
    bound.warn("first")
    bound.warn("second")
    assert bound.warnings == ("first", "second")
    assert len(bound.adjustments) == 3


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
    """THE leak bug as a test: the pre-split file swapped ``self.pipe.scheduler`` for a turbo request and never restored it, so one turbo request poisoned every later CFG request on the persistent instance."""
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

    assert model.pipe.scheduler is baseline
    assert model.pipe.loaded_loras == []

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
    with pytest.raises(RuntimeError, match="exploded mid-request"):
        host.dispatch("generate", {"prompt": "explode"}, request_id="r")
    assert model.pipe.scheduler is baseline
    assert model.pipe.loaded_loras == []


def test_stacking_on_a_step_distilled_checkpoint_warns_and_ignores(
    tmp_path: Path,
) -> None:
    """Paul's either-or ruling, warn-shaped: a step-distillation adapter on a step-distilled checkpoint is IGNORED caller-visibly — the checkpoint serves as deployed, never a fried render and never an abo..."""
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


@pytest.fixture()
def rt_host(tmp_path: Path) -> Iterator[EndpointHost]:
    loaded = load_endpoint(RT_FIXTURE_DIR)
    import serving_rt_fixture.main as rt

    rt.reset()
    booted = EndpointHost(loaded, make_binding(tmp_path, defaults={}))
    booted.setup()
    yield booted
    rt.RELEASE.set()


def test_single_flight_per_model_instance(rt_host: EndpointHost) -> None:
    import serving_rt_fixture.main as rt

    results: list = []
    a = threading.Thread(
        target=lambda: results.append(
            rt_host.dispatch("run", {"value": 1, "hold": True}, request_id="a")),
        daemon=True,
    )
    a.start()
    assert rt.ENTERED.acquire(timeout=30)

    b = threading.Thread(
        target=lambda: results.append(
            rt_host.dispatch("run", {"value": 2, "hold": True}, request_id="b")),
        daemon=True,
    )
    b.start()
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
    """Paul's per-request LoRA-list promotion: `loras: list[Adapter]` is the request's picks; scales apply via the plural model scope; full restore."""
    model = fixture_model(host)
    picks = [
        Adapter(name="style-ink", path=tmp_path / "ink",
                defaults=SDXL.Lora.Defaults(), scale=0.7, ref="me/style-ink@3"),
        Adapter(name="char-fox", path=tmp_path / "fox",
                defaults=SDXL.Lora.Defaults(), scale=1.2, ref="me/char-fox@1"),
    ]

    out = host.dispatch(
        "generate", {"prompt": "styled"}, request_id="r", loras=picks)
    assert out.model == "ckpt:tiny@1"
    assert [(used.ref, used.scale) for used in out.loras] == [
        ("me/style-ink@3", 0.7), ("me/char-fox@1", 1.2),
    ]
    assert model.pipe.adapter_history == [
        [("style-ink", 0.7), ("char-fox", 1.2)],
    ]
    assert model.pipe.loaded_loras == []
    assert model.pipe.active_adapters == []


def test_scheduler_precedence_and_the_distillation_slot_kind_guard(
    host: EndpointHost, tmp_path: Path
) -> None:
    model = fixture_model(host)
    baseline = model.pipe.scheduler

    out = host.dispatch(
        "generate", {"prompt": "x", "scheduler": "lcm"}, request_id="r1")
    assert out.model == "ckpt:tiny@1"
    assert model.pipe.scheduler is baseline

    with pytest.raises(msgspec.ValidationError):
        host.dispatch(
            "generate", {"prompt": "x", "scheduler": "not-a-scheduler"},
            request_id="r2")

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
    assert isinstance(demanding.binding.adapter, DistillationAdapter)
    assert isinstance(demanding.binding.adapter.defaults, SDXL.Lora.Defaults)

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
