"""Runtime LoRA on the w8a8 lane (gw#547) — branch buffers, key mapping,
rank-concat + bucket padding, residency routing, and lane parity against a
REAL tiny diffusers pipeline (no network, no mocks of our own code).

CPU lane: module assembly, buffer copy semantics, addend math, and the
AdapterResidency w8a8 split (forced scaled_mm module build — forward never
runs). GPU lane (sm_89+): forward parity with an explicit reference addend,
exact zero-slot equality, swap latency, and no-recompile-at-swap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

from gen_worker.api.errors import RefCompatibilitySurprise, ValidationError
from gen_worker.models import w8a8, w8a8_lora
from gen_worker.models.w8a8 import (
    detect_w8a8_artifact,
    fp8_scaled_linear_class,
    load_w8a8_denoiser,
    load_w8a8_pipeline,
    quantize_tree_w8a8,
)
from gen_worker.models.w8a8_lora import (
    apply_branch_adapters,
    branch_bucket,
    branch_target,
    clear_branch_adapters,
    disable_lora_branches,
    enable_lora_branches,
    map_adapter,
    quantized_modules,
    rank_bucket,
    split_state_dict,
)
from gen_worker.utils import lora as lora_util


def _cuda_sm89() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 89


@pytest.fixture(scope="module")
def tiny_ddpm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

    root = tmp_path_factory.mktemp("w8a8l") / "src"
    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    )
    DDPMPipeline(unet=unet, scheduler=DDPMScheduler()).save_pretrained(str(root))
    return root


@pytest.fixture(scope="module")
def w8a8_tree(tiny_ddpm: Path) -> Path:
    return quantize_tree_w8a8(tiny_ddpm, tiny_ddpm.parent / "w8a8")


@pytest.fixture()
def denoiser(w8a8_tree: Path) -> Any:
    """Fresh scaled_mm-lane denoiser per test (CPU: modules assemble, only
    forward needs the GPU kernel)."""
    art = detect_w8a8_artifact(w8a8_tree)
    assert art is not None
    return load_w8a8_denoiser(w8a8_tree, art, mode="scaled_mm")


def _kohya_sd(paths: Dict[str, Any], rank: int, alpha: float,
              prefix: str = "lora_unet_") -> Dict[str, Any]:
    """Synthetic kohya-format adapter over real module shapes."""
    sd: Dict[str, Any] = {}
    for path, mod in paths.items():
        flat = prefix + path.replace(".", "_")
        sd[f"{flat}.lora_down.weight"] = torch.randn(rank, mod.in_features)
        sd[f"{flat}.lora_up.weight"] = torch.randn(mod.out_features, rank)
        sd[f"{flat}.alpha"] = torch.tensor(float(alpha))
    return sd


def _pick(denoiser: Any, n: int) -> Dict[str, Any]:
    mods = quantized_modules(denoiser)
    return dict(list(sorted(mods.items()))[:n])


# ---------------------------------------------------------------------------
# CPU lane
# ---------------------------------------------------------------------------


def test_rank_bucket_ladder() -> None:
    assert rank_bucket(1) == 16
    assert rank_bucket(16) == 16
    assert rank_bucket(17) == 32
    assert rank_bucket(64) == 64
    assert rank_bucket(128) == 128
    with pytest.raises(RefCompatibilitySurprise):
        rank_bucket(129)


def test_enable_allocates_uniform_zeroed_branches(denoiser: Any) -> None:
    mods = quantized_modules(denoiser)
    assert mods, "tiny unet must have quantized Linears"
    enable_lora_branches(denoiser, 16)
    assert branch_bucket(denoiser) == 16
    for mod in mods.values():
        assert mod.lora_a.shape == (16, mod.in_features)
        assert mod.lora_b.shape == (mod.out_features, 16)
        assert mod.lora_a.abs().sum() == 0 and mod.lora_b.abs().sum() == 0
        # non-persistent: swaps never leak into state_dict round-trips
        assert "lora_a" not in mod.state_dict()
    # idempotent at the same bucket: no reallocation (same traced graph)
    ids = [id(m.lora_a) for m in mods.values()]
    enable_lora_branches(denoiser, 16)
    assert ids == [id(m.lora_a) for m in quantized_modules(denoiser).values()]


def test_apply_kohya_folds_scale_into_b(denoiser: Any) -> None:
    picked = _pick(denoiser, 2)
    sd = _kohya_sd(picked, rank=4, alpha=2.0)
    stats = apply_branch_adapters(denoiser, [(sd, 0.5, "t/a")])
    assert stats["bucket"] == 16 and stats["covered"] == 2
    mods = quantized_modules(denoiser)
    for path, mod in mods.items():
        if path in picked:
            flat = "lora_unet_" + path.replace(".", "_")
            a = sd[f"{flat}.lora_down.weight"].to(mod.lora_a.dtype)
            b = sd[f"{flat}.lora_up.weight"].to(mod.lora_b.dtype)
            assert torch.equal(mod.lora_a[:4], a)
            # alpha/rank * user weight = 2/4 * 0.5 = 0.25, folded into B
            assert torch.allclose(mod.lora_b[:, :4], b * 0.25)
            assert mod.lora_a[4:].abs().sum() == 0
            assert mod.lora_b[:, 4:].abs().sum() == 0
        else:
            assert mod.lora_b.abs().sum() == 0  # canonical zeroed slot


def test_peft_and_scoped_key_forms_resolve(denoiser: Any) -> None:
    picked = _pick(denoiser, 1)
    path, mod = next(iter(picked.items()))
    for down, up in (
        (f"unet.{path}.lora_A.weight", f"unet.{path}.lora_B.weight"),
        (f"unet.{path}.lora_A.default.weight", f"unet.{path}.lora_B.default.weight"),
        (f"unet.{path}.lora.down.weight", f"unet.{path}.lora.up.weight"),
    ):
        sd = {
            down: torch.randn(8, mod.in_features),
            up: torch.randn(mod.out_features, 8),
        }
        mapped = map_adapter(sd, denoiser)
        assert set(mapped) == {path}
        a, b, alpha_scale = mapped[path]
        assert a.shape[0] == 8 and alpha_scale == 1.0


def test_rank_concat_addend_matches_sum_of_adapters(denoiser: Any) -> None:
    picked = _pick(denoiser, 1)
    path, mod = next(iter(picked.items()))
    sd1 = _kohya_sd({path: mod}, rank=4, alpha=4.0)
    sd2 = _kohya_sd({path: mod}, rank=8, alpha=2.0)
    apply_branch_adapters(denoiser, [(sd1, 1.0, "t/a"), (sd2, -0.5, "t/b")])
    assert branch_bucket(denoiser) == 16
    flat = "lora_unet_" + path.replace(".", "_")
    x = torch.randn(5, mod.in_features, dtype=mod.lora_a.dtype)

    def ref(sd: Dict[str, Any], w: float, alpha: float, r: int) -> Any:
        a = sd[f"{flat}.lora_down.weight"].to(x.dtype)
        b = sd[f"{flat}.lora_up.weight"].to(x.dtype)
        return (x @ a.t()) @ b.t() * (alpha / r * w)

    want = ref(sd1, 1.0, 4.0, 4) + ref(sd2, -0.5, 2.0, 8)
    got = mod._lora_addend(x)
    assert torch.allclose(got, want, rtol=0.05, atol=0.05)  # bf16 accumulation


def test_swap_is_copy_only_and_clear_zeroes(denoiser: Any) -> None:
    picked = _pick(denoiser, 2)
    apply_branch_adapters(denoiser, [(_kohya_sd(picked, 4, 4.0), 1.0, "t/a")])
    ids = [id(m.lora_a) for m in quantized_modules(denoiser).values()]
    # swap to a different adapter in the same bucket: buffers must be reused
    stats = apply_branch_adapters(denoiser, [(_kohya_sd(picked, 8, 8.0), 1.0, "t/b")])
    assert stats["resized"] is False
    assert ids == [id(m.lora_a) for m in quantized_modules(denoiser).values()]
    assert w8a8_lora.branches_active(denoiser)
    clear_branch_adapters(denoiser)
    assert not w8a8_lora.branches_active(denoiser)
    for mod in quantized_modules(denoiser).values():
        assert mod.lora_b.abs().sum() == 0
    assert branch_bucket(denoiser) == 16  # graph family stays


def test_bucket_growth_and_compiled_resize_refusal(denoiser: Any) -> None:
    picked = _pick(denoiser, 1)
    apply_branch_adapters(denoiser, [(_kohya_sd(picked, 4, 4.0), 1.0, "t/a")])
    big = _kohya_sd(picked, 24, 24.0)
    with pytest.raises(ValidationError):
        apply_branch_adapters(denoiser, [(big, 1.0, "t/big")], allow_resize=False)
    stats = apply_branch_adapters(denoiser, [(big, 1.0, "t/big")])
    assert stats["bucket"] == 32 and stats["resized"] is True
    # never shrink back — stay on the already-traced graph
    stats = apply_branch_adapters(denoiser, [(_kohya_sd(picked, 4, 4.0), 1.0, "t/a")])
    assert stats["bucket"] == 32 and stats["resized"] is False


def test_unresolved_and_misshaped_keys_fail_loud(denoiser: Any) -> None:
    with pytest.raises(RefCompatibilitySurprise, match="no w8a8-quantized module"):
        map_adapter({"lora_unet_no_such_block.lora_down.weight": torch.zeros(4, 8)},
                    denoiser, ref="t/x")
    picked = _pick(denoiser, 1)
    path, mod = next(iter(picked.items()))
    flat = "lora_unet_" + path.replace(".", "_")
    with pytest.raises(RefCompatibilitySurprise, match="do not match the base"):
        map_adapter({
            f"{flat}.lora_down.weight": torch.zeros(4, mod.in_features + 1),
            f"{flat}.lora_up.weight": torch.zeros(mod.out_features, 4),
        }, denoiser, ref="t/x")
    with pytest.raises(RefCompatibilitySurprise, match="down/up pair"):
        map_adapter({f"{flat}.lora_down.weight": torch.zeros(4, mod.in_features)},
                    denoiser, ref="t/x")


def test_split_state_dict_routes_te_to_peft() -> None:
    den, rest = split_state_dict({
        "lora_unet_x.lora_down.weight": 1,
        "unet.y.lora_A.weight": 2,
        "transformer.z.lora_A.weight": 3,
        "lora_te1_text_model.lora_down.weight": 4,
        "text_encoder.q.lora_A.weight": 5,
    })
    assert set(den) == {"lora_unet_x.lora_down.weight", "unet.y.lora_A.weight",
                        "transformer.z.lora_A.weight"}
    assert set(rest) == {"lora_te1_text_model.lora_down.weight",
                         "text_encoder.q.lora_A.weight"}


def test_disable_returns_to_branchless(denoiser: Any) -> None:
    enable_lora_branches(denoiser, 16)
    disable_lora_branches(denoiser)
    assert branch_bucket(denoiser) == 0
    for mod in quantized_modules(denoiser).values():
        assert mod.lora_a is None and mod.lora_b is None


def test_lane_stamp_and_compile_cell_parity(denoiser: Any) -> None:
    from gen_worker.compile_cache import artifact_metadata, flavor_label, lane_drift

    class _P:
        pass

    pipe = _P()
    pipe._cozy_weight_lane = "w8a8"
    enable_lora_branches(denoiser, 16)
    w8a8_lora.stamp_lane(pipe, denoiser)
    assert pipe._cozy_weight_lane == "w8a8-lora16"
    assert flavor_label("rtx-4090", "2.9.0+cu129", "w8a8-lora16") == (
        "inductor-rtx-4090-torch2.9-w8a8-lora16")
    # SYMMETRIC adopt guard: branchless cells never adopt onto a LoRA-bearing
    # pipeline and vice versa.
    assert "weight_lane" in lane_drift(
        artifact_metadata(family="f", weight_lane="w8a8"), pipe)
    lora_pipe_meta = artifact_metadata(family="f", weight_lane="w8a8-lora16")
    assert lane_drift(lora_pipe_meta, pipe) == ""
    disable_lora_branches(denoiser)
    w8a8_lora.stamp_lane(pipe, denoiser)
    assert pipe._cozy_weight_lane == "w8a8"
    assert "weight_lane" in lane_drift(lora_pipe_meta, pipe)


# ---------------------------------------------------------------------------
# Residency routing (integration through AdapterResidency, CPU)
# ---------------------------------------------------------------------------


class _RecordingPipe:
    """Real w8a8 denoiser + recording peft surface (the TE half)."""

    def __init__(self, denoiser: Any) -> None:
        self.unet = denoiser
        self.loaded: list = []
        self.set_calls: list = []
        self.disabled = 0

    def load_lora_weights(self, sd: Any, adapter_name: str = "") -> None:
        self.loaded.append((dict(sd), adapter_name))

    def set_adapters(self, names: Any, adapter_weights: Any = None) -> None:
        self.set_calls.append((list(names), list(adapter_weights or [])))

    def unload_lora_weights(self) -> None:
        self.loaded.clear()

    def disable_lora(self) -> None:
        self.disabled += 1

    def enable_lora(self) -> None:
        pass

    def delete_adapters(self, name: str) -> None:
        pass


def _prepared(sd: Dict[str, Any], ref: str, weight: float = 1.0) -> Any:
    key = f"{ref}@d"
    return lora_util.PreparedAdapter(
        slot="model", ref=ref, cache_key=key,
        name=lora_util.adapter_name(key), weight=weight, state_dict=sd)


def test_residency_routes_denoiser_to_branch_and_te_to_peft(denoiser: Any) -> None:
    pipe = _RecordingPipe(denoiser)
    assert branch_target(pipe) is denoiser
    picked = _pick(denoiser, 1)
    sd = _kohya_sd(picked, rank=4, alpha=4.0)
    te_key = "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight"
    sd[te_key] = torch.randn(4, 8)
    res = lora_util.AdapterResidency()
    res.activate("m", pipe, [_prepared(sd, "t/mixed", 0.5)], request_id="r1")
    # TE half went to peft, denoiser half did NOT
    assert len(pipe.loaded) == 1
    assert set(pipe.loaded[0][0]) == {te_key}
    assert w8a8_lora.branches_active(denoiser)
    assert pipe._cozy_weight_lane == "w8a8-lora16"
    assert res.needs_deactivation("m")
    res.deactivate("m", pipe, request_id="r1")
    assert not w8a8_lora.branches_active(denoiser)
    for mod in quantized_modules(denoiser).values():
        assert mod.lora_b.abs().sum() == 0


def test_residency_branch_only_pipe_needs_no_peft_surface(denoiser: Any) -> None:
    class _BarePipe:
        def __init__(self, d: Any) -> None:
            self.unet = d

    pipe = _BarePipe(denoiser)
    picked = _pick(denoiser, 1)
    res = lora_util.AdapterResidency()
    res.activate("m", pipe, [_prepared(_kohya_sd(picked, 4, 4.0), "t/a")])
    assert w8a8_lora.branches_active(denoiser)
    res.detach("m", pipe)  # demote: buffers drop, lane returns branchless
    assert branch_bucket(denoiser) == 0
    assert pipe._cozy_weight_lane == "w8a8"


def test_residency_refuses_resize_on_compiled_pipe(denoiser: Any) -> None:
    pipe = _RecordingPipe(denoiser)
    enable_lora_branches(denoiser, 16)
    pipe._cozy_compile = {"cells": 1}
    picked = _pick(denoiser, 1)
    res = lora_util.AdapterResidency()
    with pytest.raises(ValidationError, match="recompile at swap"):
        res.activate("m", pipe, [_prepared(_kohya_sd(picked, 24, 24.0), "t/big")])
    # rollback left nothing active
    assert not w8a8_lora.branches_active(denoiser)


def test_dequant_lane_keeps_peft_path(w8a8_tree: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """On hosts without scaled_mm the denoiser is plain bf16 Linears —
    branch_target must be None so the normal peft path applies."""
    from diffusers import DDPMPipeline

    monkeypatch.setattr(w8a8, "scaled_mm_supported", lambda: False)
    from gen_worker.models.loading import load_from_pretrained

    pipe = load_from_pretrained(DDPMPipeline, w8a8_tree)
    assert branch_target(pipe) is None


# ---------------------------------------------------------------------------
# GPU lane (sm_89+)
# ---------------------------------------------------------------------------

gpu = pytest.mark.skipif(not _cuda_sm89(), reason="needs CUDA sm_89+ (fp8 tensor cores)")


@gpu
def test_gpu_zero_branch_is_bit_exact_with_branchless() -> None:
    torch.manual_seed(0)
    lin_cls = fp8_scaled_linear_class()
    M, K, N = 64, 128, 96
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    scale = (w.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    wq = (w.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    mod = lin_cls(K, N, bias=False, compute_dtype=torch.bfloat16,
                  static_input_scale=False)
    mod.load_state_dict({"weight": wq, "weight_scale": scale}, assign=True)
    y0 = mod(x)
    w8a8_lora.alloc_branch_buffers(mod, 16)
    assert torch.equal(mod(x), y0)


@gpu
def test_gpu_forward_matches_dequant_reference_plus_addend(denoiser: Any) -> None:
    denoiser.to("cuda")
    picked = _pick(denoiser, 2)
    sd = _kohya_sd(picked, rank=8, alpha=8.0)
    apply_branch_adapters(denoiser, [(sd, 0.7, "t/a")])
    path, mod = next(iter(picked.items()))
    x = torch.randn(32, mod.in_features, device="cuda", dtype=torch.bfloat16)
    y = mod(x)
    wdq = (mod.weight.float() * mod.weight_scale).to(torch.bfloat16)
    flat = "lora_unet_" + path.replace(".", "_")
    a = sd[f"{flat}.lora_down.weight"].to("cuda", torch.bfloat16)
    b = sd[f"{flat}.lora_up.weight"].to("cuda", torch.bfloat16)
    ref = torch.nn.functional.linear(x, wdq, mod.bias) + ((x @ a.t()) @ b.t()) * 0.7
    rel = ((y - ref).abs().max() / ref.abs().max().clamp(min=1e-6)).item()
    assert rel < 0.08  # activation-quant error only; branch itself is exact bf16


@gpu
def test_gpu_full_pipeline_with_branch_runs(w8a8_tree: Path) -> None:
    from diffusers import DDPMPipeline

    w8a8.scaled_mm_supported.cache_clear()
    art = detect_w8a8_artifact(w8a8_tree)
    assert art is not None
    pipe = load_w8a8_pipeline(DDPMPipeline, w8a8_tree, art,
                              compute_dtype=torch.float16)
    pipe.to("cuda")
    denoiser = branch_target(pipe)
    assert denoiser is not None
    picked = _pick(denoiser, 2)
    apply_branch_adapters(denoiser, [(_kohya_sd(picked, 8, 8.0), 0.5, "t/a")])
    w8a8_lora.stamp_lane(pipe, denoiser)
    assert pipe._cozy_weight_lane == "w8a8-lora16"
    out = pipe(batch_size=1, num_inference_steps=2, output_type="np")
    assert out.images.shape[-3:] == (8, 8, 3)
    # swap without touching graphs, then clear back to the zeroed set
    apply_branch_adapters(denoiser, [(_kohya_sd(picked, 4, 4.0), 1.0, "t/b")])
    clear_branch_adapters(denoiser)
    out = pipe(batch_size=1, num_inference_steps=2, output_type="np")
    assert out.images.shape[-3:] == (8, 8, 3)


@gpu
def test_gpu_swap_never_recompiles() -> None:
    """Compile a branch-bearing module, then hot-swap adapters in the same
    bucket: dynamo must not re-trace (swap = buffer copy, the gw#547
    contract)."""
    import torch._dynamo as dynamo

    torch.manual_seed(0)
    lin_cls = fp8_scaled_linear_class()
    K, N = 128, 96
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    scale = (w.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    wq = (w.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    mod = lin_cls(K, N, bias=False, compute_dtype=torch.bfloat16,
                  static_input_scale=False)
    mod.load_state_dict({"weight": wq, "weight_scale": scale}, assign=True)
    w8a8_lora.alloc_branch_buffers(mod, 16)
    dynamo.reset()
    compiled = torch.compile(mod)
    x = torch.randn(32, K, device="cuda", dtype=torch.bfloat16)
    y0 = compiled(x)
    before = dynamo.utils.counters["stats"]["unique_graphs"]
    for _ in range(4):  # hot-swaps: same bucket, new tensors
        mod.lora_a.copy_(torch.randn_like(mod.lora_a))
        mod.lora_b.copy_(torch.randn_like(mod.lora_b))
        y = compiled(x)
    assert dynamo.utils.counters["stats"]["unique_graphs"] == before
    assert not torch.equal(y, y0)
