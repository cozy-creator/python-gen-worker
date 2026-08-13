"""A `components.*` deploy override must not make the deploy-bound LoRA
unattachable — the lane's RESIDENCY handle is not the PIPELINE.

Without the fix, a pipeline slot with one overridden component refuses every
request with "model slot does not support LoRA adapters" while the same release
with no override attaches normally. It is an OBJECT IDENTITY defect in the
executor, not an adapter-code defect:

  * sharing is automatic by content address, so every component of a
    Slot-declared pipeline slot enters the share plan;
  * an OVERRIDDEN component is popped OUT of that plan (its bytes differ from
    the base's), so the lane acquires a non-empty EXCLUSIVE module set;
  * `_register_lane` books `nn.ModuleDict(exclusive)` as the residency entry —
    correct, and deliberate: LRU demote/promote must move only lane-owned
    weights, never the shared encoder;
  * but `_adapter_target` read `residency.obj(ref)` as if it were the
    pipeline. `branch_targets` finds no denoiser on a ModuleDict, so
    `_split_adapters` never runs, the whole adapter stays on the peft side,
    and `isinstance(pipe, LoraCapablePipeline)` is False -> the refusal above,
    on EVERY request, on the eager lane AND after `compiled_armed=true`.

With no override, `exclusive` is empty and `_register_lane` returns the pipe
itself, which is why the defect is invisible until a component is overridden.
The fix keeps BOTH facts and stops conflating them: the record owns the
pipeline identity (`slot_pipelines`), residency owns the movement handle.

Real path throughout: a real diffusers pipeline carrying a real (tiny)
transformer, the real `Residency`, the real `_register_lane` /
`_slot_pipeline` executor functions, and the real `AdapterResidency.activate`
with a real denoiser-key adapter grammar.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from gen_worker import compile_cache  # noqa: E402
from gen_worker.api.binding import ModelRef, wire_ref  # noqa: E402
from gen_worker.api.errors import ValidationError  # noqa: E402
from gen_worker.executor import Executor, _ClassRecord, _InjectionResult  # noqa: E402
from gen_worker.models import residency as residency_mod  # noqa: E402
from gen_worker.utils.lora import AdapterResidency, PreparedAdapter  # noqa: E402

#: A real RANK_BUCKETS value — the branch lane refuses anything else, and the
#: curated distillation adapters are rank 64 (sdxl declares lora_bucket=64).
_RANK = 16
_SLOT = "pipeline"


# ---------------------------------------------------------------------------
# Fixtures: a real pipeline whose vae is the OVERRIDDEN (exclusive) component.
# ---------------------------------------------------------------------------


def _tiny_pipe() -> Any:
    """A real diffusers pipeline: branch-capable denoiser + a real VAE (the
    component a th#980 `components.vae` override substitutes)."""
    from diffusers import (
        AutoencoderKLWan,
        UniPCMultistepScheduler,
        WanPipeline,
        WanTransformer3DModel,
    )

    torch.manual_seed(0)
    transformer = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8,
        in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=32,
        num_layers=1, cross_attn_norm=True,
        qk_norm="rms_norm_across_heads", rope_max_seq_len=32,
    )
    vae = AutoencoderKLWan(
        base_dim=4, z_dim=4, dim_mult=[1], num_res_blocks=1,
        temperal_downsample=[False],
    )
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
    return WanPipeline(
        tokenizer=None, text_encoder=None, vae=vae, scheduler=scheduler,
        transformer=transformer,
    )


def _denoiser_adapter() -> PreparedAdapter:
    """A UNet/transformer-only distillation overlay in diffusers key grammar —
    the shape of `sdxl-lightning-4step-lora`: no text-encoder half at all."""
    sd: Dict[str, Any] = {}
    for kind, shape in (("lora_A", (_RANK, 16)), ("lora_B", (16, _RANK))):
        sd[f"transformer.blocks.0.attn1.to_q.{kind}.weight"] = (
            torch.randn(*shape) * 0.02)
    return PreparedAdapter(
        slot=_SLOT, ref="tensorhub/sdxl-lightning-4step-lora:prod",
        cache_key="lightning@deadbeef", name="lightning",
        weight=1.0, state_dict=sd,
    )


class _StoreStub:
    """Only what `_register_lane` touches on the store: the REAL residency
    registry plus the load-identity sink."""

    def __init__(self) -> None:
        self.residency = residency_mod.Residency()
        self.identities: List[Any] = []

    def activate_load_identity(self, ref: str, identity: Any) -> None:
        self.identities.append((ref, identity))


class _ExecStub:
    """A minimal `self` carrying the REAL executor functions under test — the
    surrounding store is stubbed, the logic never is."""

    _slot_pipeline = Executor._slot_pipeline
    _adapter_target = Executor._adapter_target
    _register_execution_lane = Executor._register_execution_lane

    def __init__(self) -> None:
        self.store = _StoreStub()
        self._classes: Dict[Any, _ClassRecord] = {}

    def _arm_lane_residency_gate(self, pipe: Any, ref: str, spec: Any = None) -> bool:
        return False


_BINDING = ModelRef(
    source="tensorhub", path="tensorhub/wai-illustrious",
    tag="prod")
#: The one ref spelling the executor books under — derived, never hand-typed.
_REF = wire_ref(_BINDING)


class _SpecStub:
    def __init__(self) -> None:
        self.instance_key = "sdxl:SDXLFamily"
        self.models = {_SLOT: _BINDING}


def _key(component: str, digest: str) -> Any:
    return residency_mod.LoadedComponentKey.for_component(
        content_digest=digest, component=component, dtype="bf16",
        label=f"{_REF}/{component}")


def _register(pipe: Any, *, override: bool) -> _ExecStub:
    """Book the lane exactly as setup injection does. `override=True` is the
    the overridden component is popped out of the share plan, so
    it becomes the lane's EXCLUSIVE module."""
    ex = _ExecStub()
    shared = {"transformer": _key("transformer", "d-transformer")}
    if not override:
        shared["vae"] = _key("vae", "d-vae")
    injected: Dict[str, Any] = {"vae": pipe.vae} if override else {}
    result = _InjectionResult(kwargs={}, loaded={})
    result.slot_pipelines[_SLOT] = pipe          # the pgw#678 fix
    execution_lane_obj, _bytes = ex._register_execution_lane(
        _SLOT, _REF, pipe, shared, injected, 0, result, ("", 0))
    ex.store.residency.track_ram(_REF, execution_lane_obj)
    rec = _ClassRecord(cls=type(pipe), specs=[])  # type: ignore[call-arg]
    rec.slot_pipelines = dict(result.slot_pipelines)
    ex._classes["sdxl:SDXLFamily"] = rec
    return ex


# ---------------------------------------------------------------------------
# 1. The defect, reproduced through the real objects.
# ---------------------------------------------------------------------------


def test_component_override_makes_the_residency_handle_a_moduledict() -> None:
    """The mechanism, pinned: WITH an override the lane's residency entry is an
    `nn.ModuleDict` of exclusive modules; WITHOUT one it is the pipeline. Both
    are correct as MOVEMENT handles — only reading one as the pipeline is not."""
    pipe = _tiny_pipe()
    with_override = _register(pipe, override=True)
    assert isinstance(
        with_override.store.residency.obj(_REF), torch.nn.ModuleDict)

    no_override = _register(_tiny_pipe(), override=False)
    assert no_override.store.residency.obj(_REF) is not None
    assert not isinstance(
        no_override.store.residency.obj(_REF), torch.nn.ModuleDict)


def test_the_execution_lane_handle_reproduces_the_live_refusal_verbatim() -> None:
    """RED, byte-for-byte: handing the residency handle to the real
    `AdapterResidency.activate` — which is what `_adapter_target` did — raises
    the exact live message, for a UNet-only adapter on a branch-capable
    pipeline that supports LoRA perfectly well."""
    pipe = _tiny_pipe()
    ex = _register(pipe, override=True)
    handle = ex.store.residency.obj(_REF)
    adapters = AdapterResidency()
    with pytest.raises(ValidationError) as excinfo:
        adapters.activate(_REF, handle, [_denoiser_adapter()], "r-red")
    assert "model slot does not support LoRA adapters" in str(excinfo.value)
    assert "load_lora_weights/set_adapters/unload_lora_weights" in str(
        excinfo.value)


# ---------------------------------------------------------------------------
# 2. The fix: the record owns the pipeline identity.
# ---------------------------------------------------------------------------


def test_slot_pipeline_returns_the_pipeline_not_the_execution_lane_handle() -> None:
    """`_slot_pipeline` (what `_adapter_target`, the explicit-deactivation
    sweep and the OOM offload rung all read now) returns the pipeline even
    when residency books a ModuleDict."""
    pipe = _tiny_pipe()
    ex = _register(pipe, override=True)
    spec = _SpecStub()
    assert ex._slot_pipeline(spec, _SLOT) is pipe
    assert ex._adapter_target(spec, _SLOT) is pipe


def test_override_and_deploy_bound_lora_compose() -> None:
    """The contradiction the issue names is gone: the th#1174 vae binding and
    the th#1135 deploy adapter coexist. The adapter lands on the BRANCH (the
    denoiser half never touches peft) and the override survives it."""
    pipe = _tiny_pipe()
    ex = _register(pipe, override=True)
    spec = _SpecStub()
    compile_cache.apply_lora_execution_lane(pipe, _RANK)
    overridden_vae = pipe.vae

    target = ex._adapter_target(spec, _SLOT)
    adapters = AdapterResidency()
    adapters.activate(_REF, target, [_denoiser_adapter()], "r-green")

    probe = pipe.transformer.get_submodule("blocks.0.attn1.to_q")
    lora_a = getattr(probe, "lora_a", None)
    assert lora_a is not None, "the denoiser half did not reach the branch"
    assert float(lora_a.abs().sum()) > 0.0, "branch buffers still zeroed"
    # The overridden component is untouched by the attach.
    assert pipe.vae is overridden_vae

    adapters.deactivate(_REF, target, "r-green")


def test_tenant_loaded_slot_still_refuses_typed() -> None:
    """A slot with no worker-constructed pipeline keeps the honest refusal —
    the fix widens object identity, not the contract."""
    ex = _ExecStub()
    ex._classes["sdxl:SDXLFamily"] = _ClassRecord(
        cls=object, specs=[])  # type: ignore[call-arg]
    spec = _SpecStub()
    with pytest.raises(ValidationError, match="no worker-managed pipeline"):
        ex._adapter_target(spec, _SLOT)
