"""pgw#654: the objective/distilled vocabulary split + per-function
declarations — ``@worker_function`` surface, discovery manifest emission,
Slot resolution's ``.objective``/``.distilled`` backstop, view-construction
scheduler math, the class-union compile contract (pgw#647 gap #1/#6), the
multi-slot root rule (gap #7), the lora_state_dict lint (gap #9), the
derived warm plan, and the converter's objective scheduler stamp. Real
discovery/registry/convert code, no mocking (test_variant_th1004 idiom).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import msgspec
import pytest


@pytest.fixture()
def tmp_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def _write(pkg: Path, main_src: str) -> None:
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent(main_src))


# ---------------------------------------------------------------------------
# decorator surface + discovery manifest emission
# ---------------------------------------------------------------------------


def test_worker_function_objectives_round_trip_into_manifest(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_obj_method"
    _write(pkg, """
        import msgspec
        from gen_worker import RequestContext, endpoint, worker_function

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint
        class Gen:
            @worker_function(objectives=("epsilon", "v_prediction"), distilled=False)
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="base")

            @worker_function(objectives=("epsilon",))
            def generate_turbo(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="turbo")

            def caption(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="undeclared")
    """)

    fns = {f["name"]: f for f in discover_functions(tmp_pkg, main_module="ep_obj_method.main")}
    assert fns["generate"]["objectives"] == ["epsilon", "v_prediction"]
    assert fns["generate"]["distilled"] is False
    assert fns["generate-turbo"]["objectives"] == ["epsilon"]
    # distilled=None (serves either) is omitted from the manifest.
    assert "distilled" not in fns["generate-turbo"]
    # Undeclared handler: unrestricted — both keys omitted.
    assert "objectives" not in fns["caption"]
    assert "distilled" not in fns["caption"]


def test_worker_function_applies_to_bare_endpoint_functions(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_obj_fn"
    _write(pkg, """
        import msgspec
        from gen_worker import RequestContext, endpoint, worker_function

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint
        @worker_function(objectives=("flow",), distilled=True)
        def render(ctx: RequestContext, data: In_) -> Out_:
            return Out_(y="base")
    """)

    fns = {f["name"]: f for f in discover_functions(tmp_pkg, main_module="ep_obj_fn.main")}
    assert fns["render"]["objectives"] == ["flow"]
    assert fns["render"]["distilled"] is True


# Module scope: typing.get_type_hints resolves handler annotations from the
# declaring module's globals, so spec-time structs cannot be test-local.
from gen_worker import RequestContext, endpoint, worker_function  # noqa: E402


class RIn(msgspec.Struct):
    prompt: str = ""


class ROut(msgspec.Struct):
    y: str


def test_unknown_objective_rejected_at_decoration() -> None:
    with pytest.raises(ValueError, match="unknown objective"):
        @worker_function(objectives=("standard",))
        def render(ctx: RequestContext, data: RIn) -> ROut:  # pragma: no cover
            return ROut(y="x")


def test_class_regimes_dict_is_a_decoration_time_error() -> None:
    # The v1 class-level name-keyed dict is DELETED, not aliased: the kwarg
    # no longer exists, so passing it is a TypeError by construction.
    with pytest.raises(TypeError, match="regimes"):
        @endpoint(regimes={"generate": ("epsilon",)})  # type: ignore[call-arg]
        class Gen:
            def generate(self, ctx: RequestContext, data: RIn) -> ROut:
                return ROut(y="x")


def test_warmup_payload_dict_is_a_decoration_time_error() -> None:
    with pytest.raises(TypeError, match="DELETED"):
        @endpoint(warmup={"generate": {"prompt": "warmup"}})  # type: ignore[arg-type]
        class Gen:
            def generate(self, ctx: RequestContext, data: RIn) -> ROut:
                return ROut(y="x")


def test_warm_override_requires_reason() -> None:
    with pytest.raises(ValueError, match="warm_reason"):
        worker_function(warm={"frames": 8})


def test_double_worker_function_rejected() -> None:
    with pytest.raises(ValueError, match="already carries"):
        @worker_function(objectives=("epsilon",))
        @worker_function(distilled=True)
        def render(ctx: RequestContext, data: RIn) -> ROut:  # pragma: no cover
            return ROut(y="x")


# ---------------------------------------------------------------------------
# Slot resolution: .objective/.distilled + the executor-level backstop
# ---------------------------------------------------------------------------


def test_resolved_slot_carries_objective_facts() -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import Slot, resolve_slot
    from _example_family import ExampleDefaults

    resolved = resolve_slot(
        "pipeline", Slot(object), ref=HF("acme/gonzalomo-xl"),
        defaults_cls=ExampleDefaults, objective="flow", distilled=True,
        distilled_status="classified",
    )
    assert resolved.objective == "flow"
    assert resolved.distilled is True
    assert resolved.distilled_status == "classified"
    assert isinstance(resolved.defaults, ExampleDefaults)


def test_resolve_slot_defaults_to_unstamped() -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import Slot, resolve_slot
    from _example_family import ExampleDefaults

    resolved = resolve_slot(
        "pipeline", Slot(object), ref=HF("acme/plain-xl"), defaults_cls=ExampleDefaults,
    )
    assert resolved.objective == ""
    assert resolved.distilled is False
    assert resolved.distilled_status == ""


def test_objective_mismatch_raises_typed_backstop_error() -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import ObjectiveMismatchError, Slot, resolve_slot
    from _example_family import ExampleDefaults

    with pytest.raises(ObjectiveMismatchError, match="flow"):
        resolve_slot(
            "pipeline", Slot(object), ref=HF("acme/z-image"),
            defaults_cls=ExampleDefaults,
            objective="flow", allowed_objectives=("epsilon", "v_prediction"),
        )


def test_distilled_mismatch_raises_typed_backstop_error() -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import ObjectiveMismatchError, Slot, resolve_slot
    from _example_family import ExampleDefaults

    with pytest.raises(ObjectiveMismatchError, match="distilled"):
        resolve_slot(
            "pipeline", Slot(object), ref=HF("acme/dmd2-merge"),
            defaults_cls=ExampleDefaults,
            objective="epsilon", distilled=True,
            distilled_status="classified",
            allowed_distilled=False,
        )


@pytest.mark.parametrize("status", ["unclassified", "inconclusive"])
def test_unknown_distillation_evidence_fails_closed(status: str) -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import ObjectiveMismatchError, Slot, resolve_slot
    from _example_family import ExampleDefaults

    with pytest.raises(ObjectiveMismatchError, match=status):
        resolve_slot(
            "pipeline", Slot(object), ref=HF("acme/plain-xl"),
            defaults_cls=ExampleDefaults,
            distilled=False, distilled_status=status, allowed_distilled=False,
        )


def test_explicit_false_distillation_evidence_is_independent_of_objective() -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import Slot, resolve_slot
    from _example_family import ExampleDefaults

    resolved = resolve_slot(
        "pipeline", Slot(object), ref=HF("acme/plain-xl"),
        defaults_cls=ExampleDefaults,
        objective="", distilled=False,
        distilled_status="classified",
        allowed_distilled=False,
    )
    assert resolved.objective == ""
    assert resolved.distilled is False
    assert resolved.distilled_status == "classified"


def test_old_sender_unstamped_checkpoint_preserves_legacy_backstop() -> None:
    # Empty status is only an older sender. Its fully unstamped binding keeps
    # the rolling-upgrade behavior that predated the additive evidence field.
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import Slot, resolve_slot
    from _example_family import ExampleDefaults

    resolved = resolve_slot(
        "pipeline", Slot(object), ref=HF("acme/plain-xl"),
        defaults_cls=ExampleDefaults,
        allowed_objectives=("epsilon",), allowed_distilled=False,
    )
    assert resolved.objective == ""


def test_unknown_distillation_status_is_rejected() -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import ObjectiveMismatchError, Slot, resolve_slot
    from _example_family import ExampleDefaults

    with pytest.raises(ObjectiveMismatchError, match="unknown distilled_status"):
        resolve_slot(
            "pipeline", Slot(object), ref=HF("acme/plain-xl"),
            defaults_cls=ExampleDefaults,
            distilled_status="guessed",
        )


def test_wire_distillation_status_reaches_the_slot_backstop() -> None:
    from gen_worker import Slot
    from gen_worker.api.binding import HF
    from gen_worker.pb import worker_scheduler_pb2 as pb
    from gen_worker.registry import extract_specs
    from gen_worker.warmup import resolved_slots_kwargs

    @endpoint(
        models={
            "pipeline": Slot(object, default_checkpoint=HF("acme/plain-xl")),
        }
    )
    class Gen:
        def setup(self, pipeline: object) -> None:
            self.pipeline = pipeline

        @worker_function(distilled=False)
        def render(self, ctx: RequestContext, data: RIn) -> ROut:
            return ROut(y="ok")

    spec = extract_specs(Gen)[0]
    from gen_worker import dispatch

    slots = {
        "pipeline": dispatch.SlotOrder(
            ref="acme/plain-xl",
            distilled=False,
            distilled_status="unclassified",
        )
    }

    result = resolved_slots_kwargs(spec, slots)

    assert "pipeline" not in result["resolved_slots"]
    assert "unclassified" in result["slot_errors"]["pipeline"]
    assert pb.ModelBinding.DISTILLED_STATUS_FIELD_NUMBER == 8


def test_stamped_facts_within_contract_resolve_cleanly() -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import Slot, resolve_slot
    from _example_family import ExampleDefaults

    resolved = resolve_slot(
        "pipeline", Slot(object), ref=HF("acme/gonzalomo-xl"),
        defaults_cls=ExampleDefaults,
        objective="epsilon", distilled=True,
        allowed_objectives=("epsilon",), allowed_distilled=None,
    )
    assert resolved.objective == "epsilon"
    assert resolved.distilled is True


# ---------------------------------------------------------------------------
# hub_client: resolve_repo parses objective/distilled (mirrors model_family)
# ---------------------------------------------------------------------------


def _fake_resolve(monkeypatch: pytest.MonkeyPatch, body: dict) -> "object":
    import requests

    from gen_worker.models.hub_client import resolve_repo
    from gen_worker.models.refs import TensorhubRef

    class _FakeResp:
        status_code = 200

        def json(self) -> dict:
            return {
                "snapshot_digest": "abc123",
                "files": [{"path": "model.safetensors",
                           "digest": "sha256:" + "b" * 64,
                           "url": "http://x/f", "size_bytes": 10}],
                **body,
            }

    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp())
    ref = TensorhubRef(owner="acme", repo="gonzalomo-xl", tag="prod")
    return resolve_repo(ref, base_url="https://hub.example")


def test_resolve_repo_parses_objective_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = _fake_resolve(monkeypatch, {
        "model_family": "z-image", "objective": "flow", "distilled": True,
        "distilled_status": "classified",
    })
    assert resolved.objective == "flow"
    assert resolved.distilled is True
    assert resolved.distilled_status == "classified"


def test_resolve_repo_defaults_unstamped_on_older_hubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _fake_resolve(monkeypatch, {})
    assert resolved.objective == ""
    assert resolved.distilled is False
    assert resolved.distilled_status == ""


def test_resolve_repo_rejects_unknown_distillation_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker.models.hub_client import HubResolveError

    with pytest.raises(HubResolveError, match="unknown distilled_status"):
        _fake_resolve(monkeypatch, {"distilled_status": "guessed"})


# ---------------------------------------------------------------------------
# view: objective scheduler math at view construction
# ---------------------------------------------------------------------------


class _StubScheduler:
    """Duck-typed non-diffusers scheduler (deepcopy fallback path)."""

    def __init__(self) -> None:
        self.prediction_type = "epsilon"


class _StubPipe:
    def __init__(self, scheduler: object) -> None:
        self.scheduler = scheduler


def test_clone_scheduler_v_prediction_sets_pred_type_and_zero_snr() -> None:
    from gen_worker.view import clone_scheduler

    fresh = clone_scheduler(_StubPipe(_StubScheduler()), objective="v_prediction")
    assert fresh.prediction_type == "v_prediction"
    assert fresh.rescale_betas_zero_snr is True


def test_clone_scheduler_unstamped_applies_nothing() -> None:
    from gen_worker.view import clone_scheduler

    base = _StubScheduler()
    fresh = clone_scheduler(_StubPipe(base), objective="")
    assert fresh is not base  # always a private per-request object
    assert fresh.prediction_type == "epsilon"
    assert not hasattr(fresh, "rescale_betas_zero_snr")


def test_flow_objective_refuses_a_diffusion_sampler() -> None:
    pytest.importorskip("diffusers")
    from gen_worker.view import clone_scheduler

    with pytest.raises(ValueError, match="flow"):
        clone_scheduler(_StubPipe(_StubScheduler()), sampler="euler_a", objective="flow")


# ---------------------------------------------------------------------------
# The flow gate reads CAPABILITY, not the class NAME.
#
# `WAN_SCHEDULER_CONFIG` is verbatim
# `Wan-AI/Wan2.2-TI2V-5B-Diffusers@b8fff731:scheduler/scheduler_config.json` —
# the official Wan solver, which our mirrors ship byte-identically. It is a
# UniPC, so a `cls.__name__.startswith("FlowMatch")` test refused every
# `objective="flow"` request against it (three wan-2.2 functions, zero renders
# ever).
# ---------------------------------------------------------------------------

WAN_SCHEDULER_CONFIG = {
    "_class_name": "UniPCMultistepScheduler",
    "beta_end": 0.02, "beta_schedule": "linear", "beta_start": 0.0001,
    "disable_corrector": [], "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero", "flow_shift": 5.0, "lower_order_final": True,
    "num_train_timesteps": 1000, "predict_x0": True,
    "prediction_type": "flow_prediction", "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0, "solver_order": 2, "solver_p": None,
    "solver_type": "bh2", "steps_offset": 0, "thresholding": False,
    "time_shift_type": "exponential", "timestep_spacing": "linspace",
    "trained_betas": None, "use_beta_sigmas": False,
    "use_dynamic_shifting": False, "use_exponential_sigmas": False,
    "use_flow_sigmas": True, "use_karras_sigmas": False,
}


def _wan_pipe():
    from diffusers import UniPCMultistepScheduler

    return _StubPipe(UniPCMultistepScheduler.from_config(WAN_SCHEDULER_CONFIG))


def test_flow_objective_accepts_a_config_declared_unipc() -> None:
    """The official Wan solver SERVES: flow by config, not by class name."""
    pytest.importorskip("diffusers")
    from diffusers import UniPCMultistepScheduler
    from gen_worker.view import clone_scheduler

    fresh = clone_scheduler(_wan_pipe(), objective="flow")
    assert isinstance(fresh, UniPCMultistepScheduler)
    assert fresh.config.use_flow_sigmas is True
    assert fresh.config.prediction_type == "flow_prediction"


def test_flow_objective_still_refuses_a_diffusion_sampler_on_that_tree() -> None:
    """The other direction, on the SAME flow checkpoint: euler_a integrates no
    flow sigmas, so the gate must still refuse it."""
    pytest.importorskip("diffusers")
    from gen_worker.view import clone_scheduler

    with pytest.raises(ValueError, match="flow"):
        clone_scheduler(_wan_pipe(), sampler="euler_a", objective="flow")


def test_flow_objective_refuses_a_flow_capable_class_left_undeclared() -> None:
    """UniPC is flow-CAPABLE but not flow by construction: on a checkpoint whose
    config never declares it, plain UniPC is a diffusion sampler and is refused
    — the capability test is not a blanket pass for the class."""
    pytest.importorskip("diffusers")
    from diffusers import UniPCMultistepScheduler
    from gen_worker.view import clone_scheduler

    diffusion = dict(WAN_SCHEDULER_CONFIG,
                     prediction_type="epsilon", use_flow_sigmas=False)
    pipe = _StubPipe(UniPCMultistepScheduler.from_config(diffusion))
    with pytest.raises(ValueError, match="flow"):
        clone_scheduler(pipe, objective="flow")


def test_flow_capable_is_not_satisfied_by_prediction_type_alone() -> None:
    """Every diffusers scheduler ACCEPTS `prediction_type`, so a declaration
    carried onto a diffusion class must not read as flow."""
    pytest.importorskip("diffusers")
    from diffusers import (EulerAncestralDiscreteScheduler,
                           FlowMatchEulerDiscreteScheduler,
                           UniPCMultistepScheduler)
    from gen_worker.view import flow_capable

    declared = {"prediction_type": "flow_prediction", "use_flow_sigmas": True}
    assert flow_capable(EulerAncestralDiscreteScheduler, declared) is False
    assert flow_capable(UniPCMultistepScheduler, declared) is True
    assert flow_capable(FlowMatchEulerDiscreteScheduler, {}) is True


def test_declared_flow_shift_reaches_the_field_the_class_honours() -> None:
    """ie#535's second half: `{"shift": …}` on a UniPC mirror was DROPPED by
    from_config — a published shift ignored rather than applied."""
    pytest.importorskip("diffusers")
    from diffusers import FlowMatchEulerDiscreteScheduler
    from gen_worker.view import clone_scheduler

    fresh = clone_scheduler(_wan_pipe(), objective="flow",
                            config_overrides={"shift": 3.0})
    assert fresh.config.flow_shift == 3.0

    # and the reverse spelling, on a class whose field is `shift`
    fm = _StubPipe(FlowMatchEulerDiscreteScheduler())
    back = clone_scheduler(fm, objective="flow",
                           config_overrides={"flow_shift": 7.0})
    assert back.config.shift == 7.0


def test_sampler_table_defines_euler_trailing_and_dpmpp_completely() -> None:
    # The SDK table is the ONE definition of each
    # named sampler; endpoints delete their private maps.
    from gen_worker.view import SAMPLERS

    cls_name, extra = SAMPLERS["euler_trailing"]
    assert cls_name == "EulerDiscreteScheduler"
    assert extra == {"timestep_spacing": "trailing"}
    for key in ("dpmpp_2m", "dpmpp_2m_karras", "dpmpp_2m_sde", "dpmpp_2m_sde_karras"):
        _cls, extra = SAMPLERS[key]
        assert extra["solver_order"] == 2
        assert extra["final_sigmas_type"] == "zero"


def test_sampler_table_defines_ddim_trailing() -> None:
    """th#1174: the hub already RECOGNIZES `ddim_trailing` in both sdxl
    schemas; the SDK owed the DEFINITION (Hyper-SD's published recipe is DDIM
    with trailing timestep spacing), resolvable against real diffusers.

    the companion assertion — that the sdxl family enum exports
    the key so a catalog row can name it — is an SDXL fact and moved to the
    sdxl endpoint's own tests along with `SdxlScheduler`."""
    pytest.importorskip("diffusers")

    import diffusers

    from gen_worker.view import SAMPLERS, clone_scheduler

    cls_name, extra = SAMPLERS["ddim_trailing"]
    assert cls_name == "DDIMScheduler"
    assert extra == {"timestep_spacing": "trailing"}

    base = diffusers.EulerDiscreteScheduler()
    fresh = clone_scheduler(_StubPipe(base), sampler="ddim_trailing")
    assert isinstance(fresh, diffusers.DDIMScheduler)
    assert fresh.config["timestep_spacing"] == "trailing"


# ---------------------------------------------------------------------------
# convert: objective-correct scheduler config in produced snapshots
# ---------------------------------------------------------------------------


def _diffusers_source(tmp_path: Path) -> "object":
    from gen_worker.convert.ingest import IngestedSource

    src = tmp_path / "source"
    (src / "scheduler").mkdir(parents=True)
    (src / "scheduler" / "config.json").write_text(json.dumps({
        "_class_name": "EulerDiscreteScheduler", "prediction_type": "epsilon",
    }))
    (src / "unet").mkdir()
    (src / "unet" / "model.safetensors").write_bytes(b"\x00" * 8)
    (src / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionXLPipeline"}))
    return IngestedSource(
        provider="huggingface", source_ref="acme/gonzalomo-xl", source_revision="sha1",
        dir=src, layout="multi-file", model_family="sdxl", model_family_variant="",
        attrs={"dtype": "fp16"},
    )


def test_distilled_fact_writes_trailing_timestep_spacing(tmp_path: Path) -> None:
    from gen_worker.convert.clone import OutputSpec, build_flavor_tree

    source = _diffusers_source(tmp_path)
    spec = OutputSpec(dtype="fp16", file_layout="multi-file", file_type="safetensors")
    tree, _attrs = build_flavor_tree(
        source, spec, tmp_path / "flavor-distilled", distilled=True)

    cfg = json.loads((tree / "scheduler" / "config.json").read_text())
    assert cfg["timestep_spacing"] == "trailing"
    assert cfg["prediction_type"] == "epsilon"  # objective untouched


def test_v_prediction_objective_writes_zero_snr_v_pred(tmp_path: Path) -> None:
    from gen_worker.convert.clone import OutputSpec, build_flavor_tree

    source = _diffusers_source(tmp_path)
    spec = OutputSpec(dtype="fp16", file_layout="multi-file", file_type="safetensors")
    tree, _attrs = build_flavor_tree(
        source, spec, tmp_path / "flavor-vpred", objective="v_prediction")

    cfg = json.loads((tree / "scheduler" / "config.json").read_text())
    assert cfg["prediction_type"] == "v_prediction"
    assert cfg["rescale_betas_zero_snr"] is True


def test_epsilon_unstamped_leaves_scheduler_config_untouched(tmp_path: Path) -> None:
    from gen_worker.convert.clone import OutputSpec, build_flavor_tree

    source = _diffusers_source(tmp_path)
    spec = OutputSpec(dtype="fp16", file_layout="multi-file", file_type="safetensors")
    tree, _attrs = build_flavor_tree(source, spec, tmp_path / "flavor-plain")

    cfg = json.loads((tree / "scheduler" / "config.json").read_text())
    assert cfg == {"_class_name": "EulerDiscreteScheduler", "prediction_type": "epsilon"}


def test_apply_objective_scheduler_config_noop_without_scheduler_dir(tmp_path: Path) -> None:
    from gen_worker.convert.writer import apply_objective_scheduler_config

    out_dir = tmp_path / "singlefile-out"
    out_dir.mkdir()
    apply_objective_scheduler_config(out_dir, "epsilon", distilled=True)  # must not raise
    assert not (out_dir / "scheduler").exists()
