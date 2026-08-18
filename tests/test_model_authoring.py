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
    DistillationAdapter,
    LoadContext,
    Model,
    RequestContext,
    entrypoint,
)
from gen_worker.models import SDXL
from gen_worker.serving import (
    DefaultsError,
    DeployBinding,
    EndpointHost,
    ServeDispatchError,
    EntrypointDeclarationError,
    ModelDeclarationError,
    lane_handle,
    load_endpoint,
    model_lanes,
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
    kwargs.setdefault(
        "defaults",
        {"steps": {"default": 2, "lo": 1, "hi": 8, "field": "num_inference_steps"}},
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


def test_model_header_declarations_and_refusals() -> None:
    with pytest.raises(ModelDeclarationError, match=r"Model\[SDXL\]"):
        type("Bare", (Model,), {})

    with pytest.raises(ModelDeclarationError, match="lanes= must be a tuple"):
        class BadLanes(Model[SDXL], lanes=["not-a-tuple"]):  # type: ignore[arg-type]
            pass

    with pytest.raises(ModelDeclarationError, match="not a layout contract"):
        class StringLane(Model[SDXL], lanes=("sdxl.diffusers-bf16@1",)):
            pass

    class EagerPermanent(Model[SDXL], lanes=()):
        pass

    assert model_lanes(EagerPermanent) == ()
    assert model_type(EagerPermanent) is SDXL

    class OmittedLanes(Model[SDXL]):
        pass

    # Omitted lanes = the model type's canonical contract object (pgw#1377),
    # which satisfies the lane protocol: handle + load dtype.
    (canonical,) = model_lanes(OmittedLanes)
    assert canonical is SDXL.canonical_contract
    assert lane_handle(canonical) == "sdxl.diffusers-bf16@1"

    # Cheap __init__: constructing a model does not load anything.
    instance = EagerPermanent()
    assert not vars(instance)


# Malformed-signature specimens: defined at module level (the contract is
# module-level functions), decorated inside the test so the refusal is
# observable rather than an import-time explosion.


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    ok: bool = True


class _M(Model[SDXL], lanes=()):
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


def _too_few(ctx: RequestContext, payload: _In) -> _Out:
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
        (_no_model, "declares no model slot"),
        (_bare_model, "bare Model base"),
        (_optional_model, r"`Adapter \| None`"),
        (_bad_list, r"`list\[Adapter\]`"),
        (_bad_return, "return type"),
        (_too_few, "takes 2 parameters"),
        (_kw_only, "keyword"),
        (_Grouping.method, "MODULE-LEVEL"),
    ]
    for fn, pattern in cases:
        with pytest.raises(EntrypointDeclarationError, match=pattern):
            entrypoint(fn)


# --- the ctx split ----------------------------------------------------------


def test_load_context_defaults_decode_matrix(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)

    def ctx_for(defaults: dict) -> LoadContext:
        return LoadContext(
            binding=DeployBinding(
                checkpoint_ref="ckpt:x@1", checkpoint_dir=checkpoint,
                defaults=defaults,
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
            defaults=binding.defaults,
            adapter=DistillationAdapter(
                name="odd", path=tmp_path / "lora",
                defaults=SDXL.Lora.Defaults(scheduler="heun"),
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
