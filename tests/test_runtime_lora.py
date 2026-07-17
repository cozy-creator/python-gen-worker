"""Lane-general runtime LoRA branches (gw#558).

The gw#547 additive branch generalizes beyond Fp8ScaledLinear: plain
``nn.Linear`` denoisers (bf16-resident lane) and layerwise-cast denoisers
(the fp8-storage lane, where peft module wrapping breaks — ie#374) carry the
same ``y += B(A @ x)`` branch through an instance-forward wrap, with
bit-exact removal. CPU-safe throughout.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from gen_worker.api.errors import RefCompatibilitySurprise  # noqa: E402
from gen_worker.models import w8a8_lora  # noqa: E402
from gen_worker.models.w8a8_lora import (  # noqa: E402
    apply_branch_adapters,
    branch_lane,
    branch_modules,
    branch_target,
    clear_branch_adapters,
    map_adapter,
    normalize_adapter_state_dict,
    stamp_lane,
)
from gen_worker.utils import lora as lora_util  # noqa: E402


@pytest.fixture(scope="module")
def bf16_unet() -> Any:
    from diffusers import UNet2DModel

    torch.manual_seed(11)
    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    ).to(torch.bfloat16).eval()
    return unet


def _fresh(unet: Any) -> Any:
    # Branch state is per-test: drop any leftover branches.
    w8a8_lora.disable_lora_branches(unet)
    return unet


def _sample() -> tuple[Any, Any]:
    torch.manual_seed(3)
    return torch.randn(1, 3, 8, 8, dtype=torch.bfloat16), torch.tensor([4])


def _adapter_for(unet: Any, n: int = 2, rank: int = 4,
                 dotted: bool = True) -> Dict[str, Any]:
    torch.manual_seed(5)
    sd: Dict[str, Any] = {}
    for path, mod in sorted(branch_modules(unet).items())[:n]:
        base = f"unet.{path}" if dotted else "lora_unet_" + path.replace(".", "_")
        sd[f"{base}.lora_down.weight"] = torch.randn(rank, mod.in_features)
        sd[f"{base}.lora_up.weight"] = torch.randn(mod.out_features, rank)
        sd[f"{base}.alpha"] = torch.tensor(float(rank))
    return sd


# ---------------------------------------------------------------------------
# Plain bf16 lane
# ---------------------------------------------------------------------------


def test_bf16_linear_branch_applies_and_removes_bit_exact(bf16_unet: Any) -> None:
    unet = _fresh(bf16_unet)
    x, t = _sample()
    with torch.no_grad():
        base = unet(x, t).sample.clone()

    sd = _adapter_for(unet)
    apply_branch_adapters(unet, [(sd, 1.0, "t/bf16")])
    assert w8a8_lora.branches_active(unet)
    with torch.no_grad():
        with_branch = unet(x, t).sample.clone()
    assert not torch.equal(with_branch, base)

    # Sparse (eager) clear drops the branch tensors -> the wrapped forward
    # is a pure pass-through: bit-exact restore.
    clear_branch_adapters(unet)
    with torch.no_grad():
        restored = unet(x, t).sample
    assert torch.equal(restored, base)


def test_bf16_branch_math_matches_manual_addend(bf16_unet: Any) -> None:
    unet = _fresh(bf16_unet)
    path, mod = sorted(branch_modules(unet).items())[0]
    rank = 4
    torch.manual_seed(7)
    a = torch.randn(rank, mod.in_features)
    b = torch.randn(mod.out_features, rank)
    sd = {
        f"unet.{path}.lora_down.weight": a,
        f"unet.{path}.lora_up.weight": b,
        f"unet.{path}.alpha": torch.tensor(float(rank)),
    }
    apply_branch_adapters(unet, [(sd, 0.5, "t/one")])
    x = torch.randn(3, mod.in_features, dtype=torch.bfloat16)
    with torch.no_grad():
        y = mod(x)
        expected = mod._cozy_lora_orig_forward(x) + 0.5 * (
            (x @ a.to(torch.bfloat16).t()) @ b.to(torch.bfloat16).t()
        )
    assert torch.allclose(y.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_bf16_lane_stamp_composes_on_plain_base(bf16_unet: Any) -> None:
    unet = _fresh(bf16_unet)
    pipe = SimpleNamespace(unet=unet)
    assert branch_target(pipe) is unet
    assert branch_lane(unet) == ""
    apply_branch_adapters(unet, [(_adapter_for(unet), 1.0, "t/l")])
    stamp_lane(pipe, unet)
    assert pipe._cozy_weight_lane == "lora16-sparse"
    clear_branch_adapters(unet)
    stamp_lane(pipe, unet)
    assert pipe._cozy_weight_lane == ""


# ---------------------------------------------------------------------------
# fp8-storage layerwise-cast lane (the ie#374 hook-fight experiment)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cast_pipe() -> Any:
    from diffusers import UNet2DModel

    from gen_worker.models.loading import apply_fp8_storage

    torch.manual_seed(11)
    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    ).to(torch.bfloat16).eval()
    pipe = SimpleNamespace(unet=unet)
    assert apply_fp8_storage(pipe, compute_dtype=torch.bfloat16) is True
    assert unet._cozy_fp8_storage_applied is True
    return pipe


def test_cast_lane_branch_composes_with_hooks_and_removes_bit_exact(
    cast_pipe: Any,
) -> None:
    """The gw#558 verdict experiment: the additive branch must compose with
    diffusers layerwise-cast hooks (fp8 weights at rest, per-module bf16
    upcast) — the exact regime where peft module wrapping breaks (ie#374)."""
    unet = _fresh(cast_pipe.unet)
    assert branch_lane(unet) == "fp8-hooks"
    assert any(p.dtype == torch.float8_e4m3fn for p in unet.parameters())

    x, t = _sample()
    with torch.no_grad():
        base = unet(x, t).sample.clone()

    sd = _adapter_for(unet, dotted=False)  # kohya-flat resolves too
    apply_branch_adapters(unet, [(sd, 1.0, "t/cast")])
    with torch.no_grad():
        y1 = unet(x, t).sample.clone()
        y2 = unet(x, t).sample.clone()
    assert not torch.equal(y1, base)
    # Deterministic across calls AND the branch tensors were never
    # round-tripped through the fp8 cast (they live outside the hooks'
    # reach in the module __dict__).
    assert torch.equal(y1, y2)
    for mod in branch_modules(unet).values():
        a = mod.__dict__.get("lora_a")
        if a is not None:
            assert a.dtype == torch.bfloat16

    clear_branch_adapters(unet)
    with torch.no_grad():
        restored = unet(x, t).sample
    assert torch.equal(restored, base)


def test_cast_lane_stamp(cast_pipe: Any) -> None:
    unet = _fresh(cast_pipe.unet)
    pipe = cast_pipe
    for attr in ("_cozy_lora_base_lane", "_cozy_weight_lane"):
        if hasattr(pipe, attr):
            delattr(pipe, attr)
    apply_branch_adapters(unet, [(_adapter_for(unet), 1.0, "t/c")])
    stamp_lane(pipe, unet)
    assert pipe._cozy_weight_lane == "fp8-hooks-lora16-sparse"
    clear_branch_adapters(unet)
    stamp_lane(pipe, unet)
    assert pipe._cozy_weight_lane == "fp8-hooks"


# ---------------------------------------------------------------------------
# lora_state_dict normalization (te#81 zero-drift pattern)
# ---------------------------------------------------------------------------


class _ConverterPipe:
    """Pipeline class exposing a lora_state_dict converter (the LTX2/sdxl
    shape): renames vendor keys to diffusers format, returns network_alphas
    separately."""

    def __init__(self, unet: Any) -> None:
        self.unet = unet

    @classmethod
    def lora_state_dict(cls, sd: Dict[str, Any]) -> tuple:
        converted = {
            k.replace("vendor_prefix.", "unet."): v
            for k, v in sd.items() if not k.endswith(".alpha")
        }
        alphas = {
            k.replace("vendor_prefix.", "unet.")[: -len(".alpha")]: float(v)
            for k, v in sd.items() if k.endswith(".alpha")
        }
        return converted, alphas


def test_normalize_via_pipeline_class_converter(bf16_unet: Any) -> None:
    unet = _fresh(bf16_unet)
    pipe = _ConverterPipe(unet)
    path, mod = sorted(branch_modules(unet).items())[0]
    rank = 4
    raw = {
        f"vendor_prefix.{path}.lora_down.weight": torch.randn(rank, mod.in_features),
        f"vendor_prefix.{path}.lora_up.weight": torch.randn(mod.out_features, rank),
        f"vendor_prefix.{path}.alpha": torch.tensor(2.0),
    }
    out = normalize_adapter_state_dict(pipe, raw, ref="t/n")
    assert f"unet.{path}.lora_down.weight" in out
    assert f"unet.{path}.alpha" in out  # network_alphas folded back in
    mapped = map_adapter(
        {k: v for k, v in out.items()}, unet, ref="t/n",
    )
    # alpha_scale = alpha/rank = 0.5
    assert mapped[path][2] == pytest.approx(0.5)


def test_normalize_without_converter_passes_through(bf16_unet: Any) -> None:
    pipe = SimpleNamespace(unet=bf16_unet)
    sd = {"unet.x.lora_down.weight": torch.zeros(1, 1)}
    assert normalize_adapter_state_dict(pipe, sd, ref="t/p") is sd


# ---------------------------------------------------------------------------
# AdapterResidency routing (typed errors + fallback)
# ---------------------------------------------------------------------------


def _prepared(sd: Dict[str, Any], ref: str, weight: float = 1.0) -> Any:
    return lora_util.PreparedAdapter(
        slot="pipeline", ref=ref, cache_key=f"{ref}@{abs(hash(ref)):x}",
        name=lora_util.adapter_name(ref), weight=weight, state_dict=sd,
    )


class _PeftRecordingPipe:
    def __init__(self, unet: Any) -> None:
        self.unet = unet
        self.loaded: list = []
        self.active: list = []

    def load_lora_weights(self, sd: Any, adapter_name: str = "") -> None:
        self.loaded.append((dict(sd), adapter_name))

    def set_adapters(self, names: Any, adapter_weights: Any = None) -> None:
        self.active = list(names)

    def unload_lora_weights(self) -> None:
        self.loaded.clear()

    def disable_lora(self) -> None:
        self.active = []

    def enable_lora(self) -> None:
        pass

    def delete_adapters(self, name: str) -> None:
        pass


def test_te_keys_on_cast_te_fail_typed(bf16_unet: Any) -> None:
    unet = _fresh(bf16_unet)
    te = torch.nn.Linear(4, 4)
    te._cozy_fp8_storage_applied = True  # type: ignore[attr-defined]
    pipe = _PeftRecordingPipe(unet)
    pipe.text_encoder = te  # type: ignore[attr-defined]
    sd = _adapter_for(unet)
    sd["text_encoder.layers.0.q_proj.lora_A.weight"] = torch.randn(4, 4)
    sd["text_encoder.layers.0.q_proj.lora_B.weight"] = torch.randn(4, 4)
    res = lora_util.AdapterResidency()
    with pytest.raises(RefCompatibilitySurprise, match="fp8\\+te lane"):
        res.activate("m", pipe, [_prepared(sd, "t/te-cast")])
    assert not w8a8_lora.branches_active(unet)  # rollback left nothing active


def test_unmappable_plain_lane_adapter_falls_back_to_peft(bf16_unet: Any) -> None:
    """Conv-targeting adapters (LoCon-class) cannot ride the Linear branch;
    on the PLAIN lane with a peft-capable pipeline they fall back to the
    whole-adapter peft path instead of failing (capability preserved)."""
    unet = _fresh(bf16_unet)
    pipe = _PeftRecordingPipe(unet)
    sd = {
        "unet.conv_in.lora_down.weight": torch.randn(4, 3, 3, 3),
        "unet.conv_in.lora_up.weight": torch.randn(32, 4, 1, 1),
    }
    res = lora_util.AdapterResidency()
    res.activate("m", pipe, [_prepared(sd, "t/conv")], request_id="r1")
    assert len(pipe.loaded) == 1
    assert set(pipe.loaded[0][0]) == set(sd)  # WHOLE adapter went to peft
    assert not w8a8_lora.branches_active(unet)


def test_bf16_residency_branch_roundtrip_restores_output(bf16_unet: Any) -> None:
    unet = _fresh(bf16_unet)

    class _BarePipe:
        def __init__(self, d: Any) -> None:
            self.unet = d

    pipe = _BarePipe(unet)
    x, t = _sample()
    with torch.no_grad():
        base = unet(x, t).sample.clone()
    res = lora_util.AdapterResidency()
    res.activate("m", pipe, [_prepared(_adapter_for(unet), "t/rt")])
    assert w8a8_lora.branches_active(unet)
    with torch.no_grad():
        changed = unet(x, t).sample
    assert not torch.equal(changed, base)
    res.deactivate("m", pipe)
    with torch.no_grad():
        restored = unet(x, t).sample
    assert torch.equal(restored, base)
    assert pipe._cozy_weight_lane == ""  # type: ignore[attr-defined]
