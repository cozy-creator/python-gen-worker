"""sm_89 W8A8 inference lane (gw#564): per-tensor fp8 GEMM + per-channel
epilogue rescale.

CPU lane: dispatch selection matrix (SKU x call-ok x micro-benchmark gate),
rowwise-vs-pertensor output equivalence against the exact dequant reference
(torch._scaled_mm patched), bias-after-rescale placement, LoRA additive-branch
composition on the epilogue lane, and loader/swap gemm_mode threading.
GPU lane (sm_89+): pertensor numerics vs the dequant reference and
rowwise parity on real kernels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

from gen_worker.models import w8a8, w8a8_lora
from gen_worker.models.w8a8 import (
    W8a8Artifact,
    fp8_scaled_linear_class,
    load_w8a8_denoiser,
    swap_w8a8_linears,
)


def _cuda_sm89() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 89


# ---------------------------------------------------------------------------
# Dispatch selection matrix — _choose_gemm_mode is pure given the two
# injectable gates (kernel-call probe + micro-benchmark).
# ---------------------------------------------------------------------------


def _wire(monkeypatch: pytest.MonkeyPatch, ok: Dict[str, bool],
          wins: Dict[str, bool]) -> List[str]:
    benched: List[str] = []
    monkeypatch.setattr(w8a8, "_gemm_call_ok", lambda m: ok.get(m, False))

    def _prof(mode: str) -> bool:
        benched.append(mode)
        return wins.get(mode, False)

    monkeypatch.setattr(w8a8, "_gemm_profitable", _prof)
    return benched


def test_sm90_prefers_rowwise(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, {"rowwise": True, "pertensor": True},
          {"rowwise": True, "pertensor": True})
    assert w8a8._choose_gemm_mode(90) == "rowwise"
    assert w8a8._choose_gemm_mode(100) == "rowwise"
    assert w8a8._choose_gemm_mode(120) == "rowwise"


def test_sm89_prefers_pertensor(monkeypatch: pytest.MonkeyPatch) -> None:
    benched = _wire(monkeypatch, {"rowwise": True, "pertensor": True},
                    {"rowwise": False, "pertensor": True})
    assert w8a8._choose_gemm_mode(89) == "pertensor"
    assert benched == ["pertensor"]  # rowwise never even benched


def test_probe_pass_is_not_profitable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ie#498 lesson: every kernel CALL succeeds but nothing wins the
    micro-benchmark — the host must take the dequant lane, not a slow GEMM."""
    benched = _wire(monkeypatch, {"rowwise": True, "pertensor": True},
                    {"rowwise": False, "pertensor": False})
    assert w8a8._choose_gemm_mode(89) == ""
    assert benched == ["pertensor", "rowwise"]  # both candidates got a chance


def test_bench_decides_not_the_sm_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """The non-preferred candidate arms when the preferred one is broken —
    a future stack that flips the kernel story needs no code change."""
    _wire(monkeypatch, {"rowwise": False, "pertensor": True},
          {"pertensor": True})
    assert w8a8._choose_gemm_mode(90) == "pertensor"
    _wire(monkeypatch, {"rowwise": True, "pertensor": False},
          {"rowwise": True})
    assert w8a8._choose_gemm_mode(89) == "rowwise"


def test_old_silicon_never_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(mode: str) -> bool:
        raise AssertionError("probe must not run below W8A8_MIN_SM")

    monkeypatch.setattr(w8a8, "_gemm_call_ok", _boom)
    assert w8a8._choose_gemm_mode(86) == ""
    assert w8a8._choose_gemm_mode(75) == ""


def test_bench_failure_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(w8a8, "_gemm_call_ok", lambda m: True)

    def _prof(mode: str) -> bool:
        if mode == "pertensor":
            raise RuntimeError("cuda OOM mid-bench")
        return True

    monkeypatch.setattr(w8a8, "_gemm_profitable", _prof)
    assert w8a8._choose_gemm_mode(89) == "rowwise"


def test_gate_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    pairs = {"win": (1.0, 2.0), "marginal": (1.0, 1.05), "exact": (1.0, 1.10)}
    for name, (fp8_ms, bf16_ms) in pairs.items():
        monkeypatch.setattr(w8a8, "_bench_gemm_pair",
                            lambda m, p=(fp8_ms, bf16_ms): p)
        assert w8a8._gemm_profitable("pertensor") is (name != "marginal"), name


def test_invalid_gemm_mode_rejected() -> None:
    lin_cls = fp8_scaled_linear_class()
    with pytest.raises(ValueError, match="gemm_mode"):
        lin_cls(16, 16, bias=False, compute_dtype=torch.bfloat16,
                static_input_scale=False, gemm_mode="colwise")


# ---------------------------------------------------------------------------
# Epilogue correctness — CPU, torch._scaled_mm patched with the exact
# dequant reference. Recording fake: asserts the scale SHAPES each branch
# hands the GEMM, and that the pertensor branch never fuses bias.
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, a: Any, b: Any, *, scale_a: Any, scale_b: Any,
                 bias: Any = None, out_dtype: Any = None) -> Any:
        self.calls.append({
            "scale_a_shape": tuple(scale_a.shape),
            "scale_b_shape": tuple(scale_b.shape),
            "scale_b": scale_b.detach().clone(),
            "bias": bias,
        })
        y = (a.float() * scale_a) @ (b.float() * scale_b)
        if bias is not None:
            y = y + bias.float()
        return y.to(out_dtype or torch.float32)


def _quantized_module(gemm_mode: str, K: int = 32, N: int = 48, *,
                      bias: bool = True, seed: int = 11) -> Any:
    torch.manual_seed(seed)
    lin_cls = fp8_scaled_linear_class()
    mod = lin_cls(K, N, bias=bias, compute_dtype=torch.bfloat16,
                  static_input_scale=False, gemm_mode=gemm_mode)
    w = torch.randn(N, K)
    scale = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    sd = {
        "weight": (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn),
        "weight_scale": scale,
    }
    if bias:
        sd["bias"] = torch.randn(N, dtype=torch.bfloat16)
    mod.load_state_dict(sd, assign=True)
    return mod


def _row_uniform_amax_input(M: int, K: int) -> Any:
    """Activations whose per-row amax equals the global amax — the per-row
    and per-tensor dynamic scales coincide, isolating the epilogue math."""
    x = torch.randn(M, K, dtype=torch.bfloat16).clamp(-4, 4)
    x[:, 0] = 8.0
    return x


def test_pertensor_matches_rowwise_on_uniform_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder()
    monkeypatch.setattr(torch, "_scaled_mm", rec)
    M, K, N = 5, 32, 48
    row = _quantized_module("rowwise", K, N)
    per = _quantized_module("pertensor", K, N)
    per.load_state_dict(row.state_dict(), assign=True)
    x = _row_uniform_amax_input(M, K)

    y_row = row(x)
    y_per = per(x)
    assert y_row.shape == y_per.shape == (M, N)
    assert torch.allclose(y_per.float(), y_row.float(), atol=2e-2, rtol=2e-2)

    # dispatch shapes: rowwise hands vectors to the GEMM, pertensor scalars
    assert rec.calls[0]["scale_a_shape"] == (M, 1)
    assert rec.calls[0]["scale_b_shape"] == (1, N)
    assert rec.calls[0]["bias"] is not None
    assert rec.calls[1]["scale_a_shape"] == (1, 1)
    assert rec.calls[1]["scale_b_shape"] == (1, 1)
    assert torch.equal(rec.calls[1]["scale_b"], torch.ones(1, 1))
    assert rec.calls[1]["bias"] is None  # bias joins AFTER the rescale


def test_pertensor_matches_dequant_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """General activations (per-row != per-tensor scale): the epilogue lane
    must still land on F.linear(x, dequant(W), b) within activation-quant
    error."""
    monkeypatch.setattr(torch, "_scaled_mm", _Recorder())
    K, N = 32, 48
    per = _quantized_module("pertensor", K, N)
    x = torch.randn(7, K, dtype=torch.bfloat16)
    y = per(x)
    w_deq = (per.weight.float() * per.weight_scale).to(torch.bfloat16)
    ref = torch.nn.functional.linear(x, w_deq, per.bias)
    rel = ((y - ref).abs().max() / ref.abs().max()).item()
    assert rel < 0.08  # per-tensor dynamic activation quant error only


def test_bias_rides_after_the_epilogue_rescale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With weight_scale == 2 everywhere, a bias fused INSIDE the GEMM would
    come out doubled — the reference comparison pins the ordering."""
    monkeypatch.setattr(torch, "_scaled_mm", _Recorder())
    K, N = 16, 16
    lin_cls = fp8_scaled_linear_class()
    mod = lin_cls(K, N, bias=True, compute_dtype=torch.bfloat16,
                  static_input_scale=False, gemm_mode="pertensor")
    w = torch.full((N, K), 4.0)
    mod.load_state_dict({
        "weight": (w / 2.0).to(torch.float8_e4m3fn),
        "weight_scale": torch.full((N, 1), 2.0),
        "bias": torch.full((N,), 3.0, dtype=torch.bfloat16),
    }, assign=True)
    x = torch.ones(2, K, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(
        x, w.to(torch.bfloat16), torch.full((N,), 3.0, dtype=torch.bfloat16))
    assert torch.allclose(mod(x).float(), ref.float(), atol=2e-2, rtol=2e-2)


def test_static_input_scale_feeds_pertensor_gemm_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder()
    monkeypatch.setattr(torch, "_scaled_mm", rec)
    K, N = 16, 16
    lin_cls = fp8_scaled_linear_class()
    mod = lin_cls(K, N, bias=False, compute_dtype=torch.bfloat16,
                  static_input_scale=True, gemm_mode="pertensor")
    mod.load_state_dict({
        "weight": torch.randn(N, K).clamp(-4, 4).to(torch.float8_e4m3fn),
        "weight_scale": torch.full((N, 1), 0.5),
        "input_scale": torch.full((1, 1), 0.25),
    }, assign=True)
    mod(torch.randn(3, K, dtype=torch.bfloat16))
    assert rec.calls[0]["scale_a_shape"] == (1, 1)  # [1,1] static, no expand


# ---------------------------------------------------------------------------
# LoRA composability (gw#558/gw#547): the additive branch rides identically
# on the epilogue lane — orthogonal to the GEMM scaling mode.
# ---------------------------------------------------------------------------


def test_pertensor_carries_additive_lora_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "_scaled_mm", _Recorder())
    K, N, rank = 32, 48, 4
    mod = _quantized_module("pertensor", K, N, seed=7)
    x = torch.randn(3, K, dtype=torch.bfloat16)
    base = mod(x)

    a = torch.randn(rank, K, dtype=torch.bfloat16)
    b = torch.randn(N, rank, dtype=torch.bfloat16)
    mod.lora_a, mod.lora_b = a, b
    with_branch = mod(x)
    addend = (x.reshape(-1, K) @ a.t()) @ b.t()
    assert torch.allclose(with_branch, base + addend, atol=2e-2, rtol=2e-2)

    # branch removal restores the branchless epilogue path bit-exactly
    mod.lora_a = mod.lora_b = None
    assert torch.equal(mod(x), base)


def test_execution_contract_records_activation_granularity() -> None:
    """gw#564: the activation-scale granularity is a graph property — the
    contract must distinguish the per-tensor epilogue lane from rowwise."""
    import torch.nn as nn

    from gen_worker import compile_cache as cc

    class _Cfg:
        shapes = ((64, 64),)
        targets = ("unet",)
        guidance_scales = ()

    contracts = {}
    for mode in ("rowwise", "pertensor"):
        mod = _quantized_module(mode, 16, 16, bias=False)

        class _Denoiser(nn.Module):
            def __init__(self, m) -> None:
                super().__init__()
                self.proj = m

        class _P:
            _cozy_weight_lane = "w8a8"

        pipe = _P()
        pipe.unet = _Denoiser(mod)
        _sig, wc = cc.execution_contract(pipe, _Cfg())
        contracts[mode] = wc
    assert contracts["rowwise"]["activation_scaling"] == ["dynamic-per-row"]
    assert contracts["pertensor"]["activation_scaling"] == ["dynamic-per-tensor"]
    # cross-lane adoption is structurally impossible: the manifests differ
    assert contracts["rowwise"] != contracts["pertensor"]


def test_branch_lane_covers_both_gemm_modes() -> None:
    class _M:
        pass

    for mode in ("rowwise", "pertensor"):
        m = _M()
        m._cozy_w8a8_mode = mode
        assert w8a8_lora.branch_lane(m) == "w8a8"
    m = _M()
    m._cozy_w8a8_mode = "dequant"
    assert w8a8_lora.branch_lane(m) == ""


# ---------------------------------------------------------------------------
# Loader / swap threading: the mode chosen once at load lands on every
# constructed module, and the lane stamp stays "w8a8" for both GEMM branches
# (cells are per-SKU keyed — one lane token, per-SKU graphs).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def w8a8_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

    from gen_worker.models.w8a8 import quantize_tree_w8a8

    root = tmp_path_factory.mktemp("w8a8-sm89") / "src"
    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    )
    DDPMPipeline(unet=unet, scheduler=DDPMScheduler()).save_pretrained(str(root))
    return quantize_tree_w8a8(root, root.parent / "w8a8")


def test_loader_threads_pertensor_onto_every_module(w8a8_tree: Path) -> None:
    from gen_worker.models.w8a8 import detect_w8a8_artifact

    art = detect_w8a8_artifact(w8a8_tree)
    assert art is not None
    model = load_w8a8_denoiser(w8a8_tree, art, mode="pertensor")
    assert model._cozy_w8a8_mode == "pertensor"
    lin_cls = fp8_scaled_linear_class()
    swapped = [m for m in model.modules() if isinstance(m, lin_cls)]
    assert swapped
    assert all(m.gemm_mode == "pertensor" for m in swapped)
    assert w8a8_lora.branch_lane(model) == "w8a8"


def test_pipeline_lane_stamp_is_w8a8_on_pertensor(
    w8a8_tree: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import DDPMPipeline

    from gen_worker.models.loading import load_from_pretrained, pipeline_weight_lane

    monkeypatch.setattr(w8a8, "w8a8_gemm_mode", lambda: "pertensor")
    pipe = load_from_pretrained(DDPMPipeline, w8a8_tree)
    assert pipe._cozy_weight_lane == "w8a8"
    assert pipeline_weight_lane(pipe) == "w8a8"
    assert pipe.unet._cozy_w8a8_mode == "pertensor"


def test_swap_threads_gemm_mode(tmp_path: Path) -> None:
    import torch.nn as nn
    from safetensors.torch import save_file

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(16, 16)

    w = torch.randn(16, 16)
    scale = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    shard = tmp_path / "model.safetensors"
    save_file({
        "proj.weight": (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn),
        "proj.weight_scale": scale.reshape(-1),
    }, str(shard))
    art = W8a8Artifact(component="", files=(shard,),
                       quantized=("proj",), static_input_scales=False)
    model = _Tiny()
    assert swap_w8a8_linears(model, art, gemm_mode="pertensor") == 1
    assert model.proj.gemm_mode == "pertensor"
    assert model.proj.weight.dtype == torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# GPU lane (sm_89+): real kernels. Tiny tensors only.
# ---------------------------------------------------------------------------

gpu = pytest.mark.skipif(not _cuda_sm89(), reason="needs CUDA sm_89+ (fp8 tensor cores)")


@gpu
def test_pertensor_gemm_call_ok_on_real_silicon() -> None:
    assert w8a8._gemm_call_ok("pertensor") is True


@gpu
def test_pertensor_matches_dequant_reference_on_gpu() -> None:
    torch.manual_seed(0)
    lin_cls = fp8_scaled_linear_class()
    M, K, N = 64, 128, 96
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    scale = (w.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    wq = (w.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)

    mod = lin_cls(K, N, bias=True, compute_dtype=torch.bfloat16,
                  static_input_scale=False, gemm_mode="pertensor")
    mod.load_state_dict(
        {"weight": wq, "weight_scale": scale, "bias": bias}, assign=True)
    y = mod(x)
    ref = torch.nn.functional.linear(x, (wq.float() * scale).to(torch.bfloat16), bias)
    rel = ((y - ref).abs().max() / ref.abs().max()).item()
    assert rel < 0.08  # per-tensor dynamic activation quant error only


@gpu
def test_rowwise_and_pertensor_agree_on_gpu() -> None:
    """Same weights, same input: the two dispatch branches are the same math
    up to activation-quant granularity + one bf16 epilogue rounding."""
    torch.manual_seed(1)
    lin_cls = fp8_scaled_linear_class()
    M, K, N = 32, 64, 48
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    scale = (w.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    wq = (w.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    sd = {"weight": wq, "weight_scale": scale}

    mods = {}
    for mode in ("rowwise", "pertensor"):
        m = lin_cls(K, N, bias=False, compute_dtype=torch.bfloat16,
                    static_input_scale=False, gemm_mode=mode)
        m.load_state_dict(sd, assign=True)
        mods[mode] = m
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    y_row = mods["rowwise"](x)
    y_per = mods["pertensor"](x)
    rel = ((y_row - y_per).abs().max() / y_row.abs().max()).item()
    assert rel < 0.08
