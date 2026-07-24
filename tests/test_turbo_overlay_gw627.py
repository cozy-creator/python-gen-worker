"""gw#627: curated turbo LoRA onto the fp8 w8a8 lane — the live sdxl fatal.

The live failure (th#1037 addendum, request 16205a14, L4, lane
fp8-w8a8-dynamic+compiled): sdxl generate-turbo pushed its Lightning
adapter through raw diffusers/peft ``load_lora_weights`` onto a UNet whose
Linears are ``_Fp8ScaledLinear`` (pertensor) — peft rejects the module
class fatally. The sanctioned route is the gw#547/558 additive branch via
``AdapterResidency`` — which until gw#627 could not carry the 49 CONV LoRA
pairs every curated sdxl distill adapter ships.

These tests run the REAL codepath end-to-end on CPU: a real (tiny)
diffusers ``UNet2DConditionModel`` with real ``_Fp8ScaledLinear`` pertensor
swaps, an adapter in the exact live key grammar (kohya-flat diffusers
naming: ``lora_unet_*.lora_down.weight`` / ``.lora_up.weight`` /
``.alpha``, conv pairs 4-d), driven through
``AdapterResidency.activate/deactivate`` with the compiled marker set —
asserting the branch carries the adapter and peft is never consulted for
the denoiser half. Forward numerics for the conv branch run on the plain
lane (CPU has no ``torch._scaled_mm``; the Fp8ScaledLinear branch addend is
the pre-existing gw#547 forward, unchanged here).
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from gen_worker.api.errors import RefCompatibilitySurprise  # noqa: E402
from gen_worker.models import w8a8_lora  # noqa: E402
from gen_worker.models.w8a8 import fp8_scaled_linear_class  # noqa: E402
from gen_worker.utils.lora import AdapterResidency, PreparedAdapter  # noqa: E402


def _tiny_unet() -> Any:
    from diffusers import UNet2DConditionModel

    torch.manual_seed(0)
    return UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        block_out_channels=(32, 64),
        layers_per_block=1,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=8,
        norm_num_groups=8,
    )


def _swap_scaled(unet: Any, path: str) -> Any:
    """Replace one plain Linear with a REAL _Fp8ScaledLinear (pertensor —
    the live L4 mode) carrying the quantized form of its weights."""
    cls = fp8_scaled_linear_class()
    parent_path, _, leaf = path.rpartition(".")
    parent = unet.get_submodule(parent_path) if parent_path else unet
    old = getattr(parent, leaf)
    new = cls(
        old.in_features, old.out_features, bias=old.bias is not None,
        compute_dtype=torch.float32, static_input_scale=False,
        gemm_mode="pertensor",
    )
    new.weight = old.weight.detach().to(torch.float8_e4m3fn)
    new.weight_scale = torch.ones(old.out_features, 1, dtype=torch.float32)
    if old.bias is not None:
        new.bias = torch.nn.Parameter(
            old.bias.detach().clone(), requires_grad=False)
    setattr(parent, leaf, new)
    return new


class _PeftRecorder:
    """Stub diffusers-pipeline LoRA surface: every call is recorded; the
    denoiser half must never arrive here."""

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class _Pipe:
    """Minimal pipeline holding the real unet (the branch machinery and the
    normalize/split path key off ``pipe.unet`` and the class, nothing
    else)."""

    def __init__(self, unet: Any) -> None:
        self.unet = unet
        self.load_lora_weights = _PeftRecorder()
        self.set_adapters = _PeftRecorder()
        self.unload_lora_weights = _PeftRecorder()
        self.disable_lora = _PeftRecorder()


# Live key grammar: kohya-flat diffusers naming, exactly what
# sdxl_lightning_4step_lora.safetensors ships (rank 64 there; small here).
_RANK = 8


def _turbo_style_adapter(unet: Any) -> Dict[str, Any]:
    torch.manual_seed(1)
    sd: Dict[str, Any] = {}

    def pair(flat: str, a: Any, b: Any) -> None:
        sd[f"lora_unet_{flat}.lora_down.weight"] = a
        sd[f"lora_unet_{flat}.lora_up.weight"] = b
        sd[f"lora_unet_{flat}.alpha"] = torch.tensor(float(_RANK))

    # Linear pair onto the (quantized) attention projection.
    q = unet.get_submodule(
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q")
    pair(
        "down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q",
        torch.randn(_RANK, q.in_features) * 0.05,
        torch.randn(q.out_features, _RANK) * 0.05,
    )
    # Conv pairs — the gw#627 gap: downsampler conv + resnet conv, 4-d
    # halves at the base conv's kernel size (the 49-pair class every
    # curated sdxl distill adapter carries).
    ds = unet.get_submodule("down_blocks.0.downsamplers.0.conv")
    pair(
        "down_blocks_0_downsamplers_0_conv",
        torch.randn(_RANK, ds.in_channels, *ds.kernel_size) * 0.05,
        torch.randn(ds.out_channels, _RANK, 1, 1) * 0.05,
    )
    r1 = unet.get_submodule("down_blocks.0.resnets.0.conv1")
    pair(
        "down_blocks_0_resnets_0_conv1",
        torch.randn(_RANK, r1.in_channels, *r1.kernel_size) * 0.05,
        torch.randn(r1.out_channels, _RANK, 1, 1) * 0.05,
    )
    return sd


def _prepared(unet: Any, name: str = "turbo-lightning-4step") -> PreparedAdapter:
    return PreparedAdapter(
        slot="pipeline",
        ref="turbo/lightning-4step",
        cache_key="turbo/lightning-4step",
        name=name,
        weight=1.0,
        state_dict=_turbo_style_adapter(unet),
    )


def test_turbo_adapter_rides_the_branch_on_a_w8a8_pertensor_unet() -> None:
    """The exact live shape: compiled w8a8 pipeline, quantized attention
    Linears, conv-bearing curated adapter — activate must land the whole
    adapter on the additive branch, never on peft."""
    unet = _tiny_unet()
    scaled = _swap_scaled(
        unet, "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q")
    unet._cozy_w8a8_mode = "pertensor"
    pipe = _Pipe(unet)
    # The arming path (Compile.lora_bucket / apply_lora_lane) pre-enables
    # canonical zeroed branches; the compiled marker forbids resizes.
    w8a8_lora.enable_lora_branches(unet, 16)
    pipe._cozy_compile = object()

    residency = AdapterResidency()
    residency.activate("pipeline", pipe, [_prepared(unet)], request_id="t1")

    assert not pipe.load_lora_weights.calls, "denoiser half must never hit peft"
    assert not pipe.set_adapters.calls
    assert w8a8_lora.branches_active(unet)
    assert w8a8_lora.branch_bucket(unet) == 16  # no resize under compile
    # Quantized Linear carries the pair in its registered buffers.
    assert float(scaled.lora_b.abs().sum()) > 0
    # Conv branches carry theirs as __dict__ attrs (4-d).
    ds = unet.get_submodule("down_blocks.0.downsamplers.0.conv")
    assert ds.lora_a.dim() == 4 and float(ds.lora_b.abs().sum()) > 0
    # Uncovered modules keep canonical zeroed slots.
    r2 = unet.get_submodule("down_blocks.0.resnets.0.conv2")
    assert getattr(r2, "lora_a", None) is not None
    assert float(r2.lora_b.abs().sum()) == 0

    # Deactivate zeroes B (graph stays); re-activate repopulates — the
    # th#1036 "second request contributes nothing" class must stay dead.
    residency.deactivate("pipeline", pipe, request_id="t1")
    assert float(scaled.lora_b.abs().sum()) == 0
    assert float(ds.lora_b.abs().sum()) == 0
    residency.activate("pipeline", pipe, [_prepared(unet)], request_id="t2")
    assert float(scaled.lora_b.abs().sum()) > 0
    assert float(ds.lora_b.abs().sum()) > 0


def test_conv_branch_changes_output_and_clears_bit_exact() -> None:
    """Plain-lane numerics on CPU: the conv branch addend is real, and a
    cleared branch restores the exact baseline output."""
    unet = _tiny_unet().eval()
    pipe = _Pipe(unet)
    sample = torch.randn(1, 4, 16, 16)
    t = torch.tensor([3])
    ehs = torch.randn(1, 4, 32)

    def run() -> Any:
        with torch.no_grad():
            return unet(sample, t, encoder_hidden_states=ehs).sample

    baseline = run()
    residency = AdapterResidency()
    residency.activate("pipeline", pipe, [_prepared(unet)], request_id="n1")
    overlaid = run()
    assert not torch.equal(baseline, overlaid), "adapter must be visible"
    residency.deactivate("pipeline", pipe, request_id="n1")
    assert torch.equal(baseline, run()), "cleared branch must be bit-exact"


def test_conv_pair_shape_mismatch_is_a_typed_refusal() -> None:
    unet = _tiny_unet()
    sd = {
        "lora_unet_down_blocks_0_downsamplers_0_conv.lora_down.weight":
            torch.randn(_RANK, 999, 3, 3),
        "lora_unet_down_blocks_0_downsamplers_0_conv.lora_up.weight":
            torch.randn(32, _RANK, 1, 1),
    }
    with pytest.raises(RefCompatibilitySurprise):
        w8a8_lora.map_adapter(sd, unet, ref="turbo/bad")


def test_grouped_conv_is_refused_not_silently_wrong() -> None:
    import torch.nn as nn

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(8, 8, 3, padding=1, groups=2)

    m = _M()
    sd = {
        "conv.lora_down.weight": torch.randn(4, 8, 3, 3),
        "conv.lora_up.weight": torch.randn(8, 4, 1, 1),
    }
    with pytest.raises(RefCompatibilitySurprise, match="grouped conv"):
        w8a8_lora.map_adapter(sd, m, ref="turbo/grouped")
