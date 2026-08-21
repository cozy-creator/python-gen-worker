"""gw#679: a DUAL-EXPERT MoE denoiser is a SET of branch targets."""

from __future__ import annotations

from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from gen_worker.api.errors import RefCompatibilitySurprise  # noqa: E402
from gen_worker.models import w8a8_lora  # noqa: E402
from gen_worker.utils.lora import AdapterResidency, PreparedAdapter  # noqa: E402

_RANK = 8
_PROBE = "blocks.0.attn1.to_q"


def _tiny_expert(seed: int) -> Any:
    from diffusers import WanTransformer3DModel

    torch.manual_seed(seed)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8,
        in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=32,
        num_layers=1, cross_attn_norm=True,
        qk_norm="rms_norm_across_heads", rope_max_seq_len=32,
    )


def _wan_pipe(*, moe: bool = True) -> Any:
    from diffusers import UniPCMultistepScheduler, WanPipeline

    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
    return WanPipeline(
        tokenizer=None, text_encoder=None, vae=None, scheduler=scheduler,
        transformer=_tiny_expert(0),
        transformer_2=_tiny_expert(1) if moe else None,
        boundary_ratio=0.9 if moe else None,
    )


def _half(expert: Any, prefix: str, seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    sd: Dict[str, Any] = {}
    for path, mod in w8a8_lora.branch_modules(expert).items():
        sd[f"{prefix}{path}.lora_A.weight"] = (
            torch.randn(_RANK, mod.in_features) * 0.05)
        sd[f"{prefix}{path}.lora_B.weight"] = (
            torch.randn(mod.out_features, _RANK) * 0.05)
    return sd


def _raw_lightx2v_half(seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    sd: Dict[str, Any] = {}
    for o in ("q", "k", "v", "o"):
        base = f"diffusion_model.blocks.0.self_attn.{o}"
        sd[f"{base}.lora_A.weight"] = torch.randn(_RANK, 16) * 0.05
        sd[f"{base}.lora_B.weight"] = torch.randn(16, _RANK) * 0.05
    return sd


def _prepared(sd: Dict[str, Any], ref: str, weight: float = 1.0) -> PreparedAdapter:
    return PreparedAdapter(
        slot="pipeline", ref=ref, cache_key=ref,
        name=f"gw-{abs(hash(ref)) % 10**8}", weight=weight, state_dict=sd,
    )


def _delta(expert: Any, path: str = _PROBE) -> Any:
    mod = expert.get_submodule(path)
    return mod.lora_b.detach() @ mod.lora_a.detach()


def _expected_delta(sd: Dict[str, Any], prefix: str, weight: float,
                    path: str = _PROBE) -> Any:
    a = sd[f"{prefix}{path}.lora_A.weight"]
    b = sd[f"{prefix}{path}.lora_B.weight"]
    return (b @ a) * weight


def _branch_addend(expert: Any, x: Any, path: str = _PROBE) -> Any:
    import torch.nn.functional as F

    mod = expert.get_submodule(path)
    with torch.no_grad():
        return mod(x) - F.linear(x, mod.weight, mod.bias)


def test_each_distillation_half_lands_on_its_own_expert() -> None:
    pipe = _wan_pipe()
    assert list(w8a8_lora.branch_targets(pipe)) == ["transformer", "transformer_2"]

    high = _half(pipe.transformer, "transformer.", seed=11)
    low = _half(pipe.transformer_2, "transformer_2.", seed=22)
    residency = AdapterResidency()
    residency.activate(
        "wan22", pipe,
        [_prepared(high, "hub/lightning-250928-high@1", weight=2.0),
         _prepared(low, "hub/lightning-250928-low@1", weight=1.0)],
        request_id="moe1",
    )

    assert w8a8_lora.branches_active(pipe.transformer)
    assert w8a8_lora.branches_active(pipe.transformer_2), (
        "the low-noise expert must be adapted — serving it undistilled on a "
        "distilled ladder is the gw#679 silent defect")

    want_high = _expected_delta(high, "transformer.", 2.0)
    want_low = _expected_delta(low, "transformer_2.", 1.0)
    got_high, got_low = _delta(pipe.transformer), _delta(pipe.transformer_2)

    assert torch.allclose(got_high, want_high, atol=1e-5)
    assert torch.allclose(got_low, want_low, atol=1e-5)
    assert not torch.allclose(got_high, want_high + want_low, atol=1e-5)
    assert not torch.allclose(got_high, got_low, atol=1e-5)

    x = torch.randn(2, 16)
    assert torch.allclose(_branch_addend(pipe.transformer, x),
                          x @ want_high.t(), atol=1e-4)
    assert torch.allclose(_branch_addend(pipe.transformer_2, x),
                          x @ want_low.t(), atol=1e-4)

    base_high = _branch_addend(pipe.transformer, x)
    residency.deactivate("wan22", pipe, request_id="moe1")
    assert not w8a8_lora.branches_active(pipe.transformer)
    assert not w8a8_lora.branches_active(pipe.transformer_2)
    assert float(_branch_addend(pipe.transformer, x).abs().sum()) == 0.0
    assert float(_branch_addend(pipe.transformer_2, x).abs().sum()) == 0.0
    assert float(base_high.abs().sum()) > 0


def test_one_half_alone_leaves_the_other_expert_bare_not_double_adapted() -> None:
    """A single-expert adapter is routed, not smeared: the named expert gets it and the sibling keeps exactly branchless behavior."""
    pipe = _wan_pipe()
    low = _half(pipe.transformer_2, "transformer_2.", seed=33)
    AdapterResidency().activate(
        "wan22", pipe, [_prepared(low, "hub/lightning-250928-low@2")],
        request_id="moe2")

    assert w8a8_lora.branches_active(pipe.transformer_2)
    assert not w8a8_lora.branches_active(pipe.transformer)
    x = torch.randn(2, 16)
    assert float(_branch_addend(pipe.transformer, x).abs().sum()) == 0.0
    assert torch.allclose(
        _delta(pipe.transformer_2), _expected_delta(low, "transformer_2.", 1.0),
        atol=1e-5)


def test_unmirrored_per_expert_file_is_refused_not_silently_high_noise() -> None:
    """The exact live shape of the defect: both 250928 halves ship the same `diffusion_model.` key names, and diffusers' converter puts both on the `transformer.` prefix."""
    pipe = _wan_pipe()
    residency = AdapterResidency()
    with pytest.raises(RefCompatibilitySurprise) as err:
        residency.activate(
            "wan22", pipe,
            [_prepared(_raw_lightx2v_half(44), "hub/lightning-250928-high@3"),
             _prepared(_raw_lightx2v_half(55), "hub/lightning-250928-low@3")],
            request_id="moe3",
        )
    assert "component" in str(err.value)
    assert not w8a8_lora.branches_active(pipe.transformer)
    assert not w8a8_lora.branches_active(pipe.transformer_2)


def test_declared_component_the_pipeline_lacks_is_refused() -> None:
    """A `transformer_2.` half bound to a DENSE pipeline names a component that does not exist — unroutable, and nothing may attach."""
    pipe = _wan_pipe(moe=False)
    assert list(w8a8_lora.branch_targets(pipe)) == ["transformer"]
    sd = _half(pipe.transformer, "transformer_2.", seed=66)
    with pytest.raises(RefCompatibilitySurprise, match="does not carry"):
        AdapterResidency().activate(
            "wan22-dense", pipe, [_prepared(sd, "hub/lightning-250928-low@4")],
            request_id="moe4")
    assert not w8a8_lora.branches_active(pipe.transformer)


def test_a_broken_second_half_attaches_nothing_at_all() -> None:
    """Fail-closed under a MAPPING error, not just a routing one: a good high half next to a shape-broken low half must leave the pipeline bare."""
    pipe = _wan_pipe()
    high = _half(pipe.transformer, "transformer.", seed=77)
    low = _half(pipe.transformer_2, "transformer_2.", seed=88)
    low[f"transformer_2.{_PROBE}.lora_A.weight"] = torch.randn(_RANK, 999)

    with pytest.raises(RefCompatibilitySurprise):
        AdapterResidency().activate(
            "wan22", pipe,
            [_prepared(high, "hub/lightning-250928-high@5"),
             _prepared(low, "hub/lightning-250928-low@5")],
            request_id="moe5",
        )
    assert not w8a8_lora.branches_active(pipe.transformer)
    assert not w8a8_lora.branches_active(pipe.transformer_2)
    x = torch.randn(2, 16)
    assert float(_branch_addend(pipe.transformer, x).abs().sum()) == 0.0


def test_lora_bucket_container_is_armed_on_both_experts() -> None:
    pipe = _wan_pipe()
    assert w8a8_lora.apply_lora_execution_lane(pipe, 32)
    assert w8a8_lora.branch_bucket(pipe.transformer) == 32
    assert w8a8_lora.branch_bucket(pipe.transformer_2) == 32
    assert w8a8_lora.pipeline_branch_bucket(pipe) == 32
    assert pipe._cozy_weight_lane == "lora32"

    pipe._cozy_compile = object()
    residency = AdapterResidency()
    high = _half(pipe.transformer, "transformer.", seed=99)
    low = _half(pipe.transformer_2, "transformer_2.", seed=100)
    for attempt in ("attach", "swap", "reattach"):
        residency.activate(
            "wan22", pipe,
            [_prepared(high, "hub/lightning-250928-high@6"),
             _prepared(low, "hub/lightning-250928-low@6")],
            request_id=attempt,
        )
        assert w8a8_lora.branch_bucket(pipe.transformer) == 32
        assert w8a8_lora.branch_bucket(pipe.transformer_2) == 32
        assert pipe._cozy_weight_lane == "lora32"
        for expert in (pipe.transformer, pipe.transformer_2):
            slots = [m for m in w8a8_lora.branch_modules(expert).values()
                     if getattr(m, "lora_a", None) is not None]
            assert len(slots) == len(w8a8_lora.branch_modules(expert))
        assert torch.allclose(
            _delta(pipe.transformer_2),
            _expected_delta(low, "transformer_2.", 1.0), atol=1e-5)
        residency.deactivate("wan22", pipe, request_id=attempt)
        assert w8a8_lora.branch_bucket(pipe.transformer_2) == 32
        assert float(_delta(pipe.transformer_2).abs().sum()) == 0.0

    residency.detach("wan22", pipe)
    assert w8a8_lora.pipeline_branch_bucket(pipe) == 0
    assert getattr(pipe.transformer_2, "lora_a", None) is None
    assert pipe._cozy_weight_lane == ""


def test_a_half_armed_compiled_pipeline_refuses_instead_of_no_opping() -> None:
    """If only one expert carries the container (a family that declared just `transformer` as a compile target), the sibling's half would be copied into nothing."""
    pipe = _wan_pipe()
    w8a8_lora.enable_lora_branches(pipe.transformer, 32)
    pipe._cozy_compile = object()
    high = _half(pipe.transformer, "transformer.", seed=111)
    low = _half(pipe.transformer_2, "transformer_2.", seed=222)
    with pytest.raises(RefCompatibilitySurprise, match="branch container"):
        AdapterResidency().activate(
            "wan22", pipe,
            [_prepared(high, "hub/high@7"), _prepared(low, "hub/low@7")],
            request_id="moe7")
    assert not w8a8_lora.branches_active(pipe.transformer)
    assert float(_delta(pipe.transformer).abs().sum()) == 0.0


def test_compiled_pipeline_refuses_a_set_that_needs_a_wider_bucket() -> None:
    """The no-recompile-at-swap-time rule is enforced over the SET (the sum of concatenated ranks on either expert)."""
    pipe = _wan_pipe()
    w8a8_lora.apply_lora_execution_lane(pipe, 16)
    pipe._cozy_compile = object()
    wide: Dict[str, Any] = {}
    for path, mod in w8a8_lora.branch_modules(pipe.transformer_2).items():
        wide[f"transformer_2.{path}.lora_A.weight"] = torch.randn(24, mod.in_features)
        wide[f"transformer_2.{path}.lora_B.weight"] = torch.randn(mod.out_features, 24)
    from gen_worker.api.errors import ValidationError

    with pytest.raises(ValidationError, match="rank bucket"):
        AdapterResidency().activate(
            "wan22", pipe, [_prepared(wide, "hub/wide")], request_id="moe6")


def test_single_denoiser_pipeline_takes_the_component_prefixed_grammar() -> None:
    """One denoiser => its own component prefix routes to it, and the whole request behaves exactly as the LTX/qwen lanes do today."""
    pipe = _wan_pipe(moe=False)
    sd = _half(pipe.transformer, "transformer.", seed=123)
    residency = AdapterResidency()
    residency.activate(
        "dense", pipe, [_prepared(sd, "hub/style@dotted")], request_id="dense1")
    assert w8a8_lora.branches_active(pipe.transformer)
    assert torch.allclose(
        _delta(pipe.transformer), _expected_delta(sd, "transformer.", 1.0),
        atol=1e-5)

    x = torch.randn(2, 16)
    residency.deactivate("dense", pipe, request_id="dense1")
    assert float(_branch_addend(pipe.transformer, x).abs().sum()) == 0.0


def test_single_denoiser_pipeline_still_takes_an_UNDECLARED_adapter() -> None:
    """The declaration requirement is a MULTI-expert rule only: the raw lightx2v grammar (which the diffusers converter rewrites onto `transformer.`) keeps working unchanged on a dense pipeline — no mirro..."""
    pipe = _wan_pipe(moe=False)
    raw = _raw_lightx2v_half(seed=321)
    residency = AdapterResidency()
    residency.activate(
        "dense", pipe, [_prepared(raw, "hub/lightx2v@dense")],
        request_id="dense2")
    assert w8a8_lora.branches_active(pipe.transformer)
    x = torch.randn(2, 16)
    assert float(_branch_addend(pipe.transformer, x).abs().sum()) > 0
    residency.deactivate("dense", pipe, request_id="dense2")
    assert float(_branch_addend(pipe.transformer, x).abs().sum()) == 0.0


def test_text_encoder_only_overlay_owes_no_component_declaration() -> None:
    """The declaration is owed by adapters with a DENOISER half."""
    from gen_worker.utils.lora import _split_adapters

    pipe = _wan_pipe()
    sd = {
        "text_encoder.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 16),
        "text_encoder.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(16, 4),
    }
    peft, branch = _split_adapters(
        pipe, [_prepared(sd, "hub/te-only@8")],
        ("transformer", "transformer_2"))
    assert branch == {}
    assert [a.ref for a in peft] == ["hub/te-only@8"]


def test_route_is_a_pure_partition_of_the_denoiser_half() -> None:
    """The routing primitive itself, on both topologies."""
    both = {"transformer.a.lora_A.weight": 1, "transformer_2.a.lora_A.weight": 2}
    dense = {"transformer.a.lora_A.weight": 1, "lora_unet_a.lora_down.weight": 2}
    assert w8a8_lora.route_denoiser_keys(dense, ("transformer",)) == {
        "transformer": dense}
    assert w8a8_lora.route_denoiser_keys(
        both, ("transformer", "transformer_2")) == {
            "transformer": {"transformer.a.lora_A.weight": 1},
            "transformer_2": {"transformer_2.a.lora_A.weight": 2},
        }
    assert w8a8_lora.route_denoiser_keys({}, ("transformer",)) == {}
    with pytest.raises(RefCompatibilitySurprise, match="ambiguous"):
        w8a8_lora.route_denoiser_keys(
            {"a.lora_A.weight": 1}, ("transformer", "transformer_2"))
    with pytest.raises(RefCompatibilitySurprise, match="does not carry"):
        w8a8_lora.route_denoiser_keys(both, ("transformer",))
