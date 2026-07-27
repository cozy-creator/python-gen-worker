"""pgw#727: the fp8-storage lane is module STRUCTURE, not a cast hook.

Every assertion here runs the REAL codepath (``loading.apply_fp8_storage`` on
a real, tiny ``UNet2DConditionModel``) on CPU — no mocks, no weights, no GPU.
The lane the SDK ships must be:

1. hook-free, where the diffusers hook lane demonstrably HAS hooks (the red
   half: the same tape proves the old shape and the new shape apart);
2. coverage-equivalent to diffusers' own rule — same leaf set, so residency
   and numerics cannot silently drift when upstream changes its rule;
3. numerically equivalent — bitwise-equal outputs vs the hook lane;
4. residency-equivalent — identical resident bytes per dtype (pgw#704 S12-c's
   "+11.6% VRAM" was the prototype swapping ``nn.Linear`` only);
5. exportable, where the hook lane is REFUSED by ``torch.export``;
6. a DIFFERENT traced graph, visible as a different
   ``compile_cache.execution_contract`` signature (new cell keys — intended)
   while the ``fp8-hooks`` wire lane value is unchanged;
7. still a LoRA branch target (gw#547/#627 branch machinery), and — pgw#726 —
   ``_Fp8ScaledLinear``'s ``lora_a``/``lora_b`` are DECLARED buffer slots that
   ``register_buffer`` accepts without popping ``__dict__`` first.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from gen_worker.models import fp8_storage, w8a8_lora  # noqa: E402
from gen_worker.models.loading import (  # noqa: E402
    apply_fp8_storage,
    pipeline_weight_lane,
)
from gen_worker.models.w8a8 import fp8_scaled_linear_class  # noqa: E402

STORAGE = torch.float8_e4m3fn
COMPUTE = torch.bfloat16


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
    ).to(COMPUTE).eval()


class _Pipe:
    """The minimal shape ``apply_fp8_storage`` consumes: an object with a
    denoiser component attribute."""

    def __init__(self, unet: Any) -> None:
        self.unet = unet


def _hook_lane(unet: Any) -> List[str]:
    """Arm diffusers' layerwise casting (the lane as it shipped) and return
    the leaf paths it hooked."""
    unet.enable_layerwise_casting(storage_dtype=STORAGE, compute_dtype=COMPUTE)
    hooked: List[str] = []
    for name, mod in unet.named_modules():
        registry = getattr(mod, "_diffusers_hook", None)
        if registry is not None and registry.get_hook("layerwise_casting") is not None:
            hooked.append(name)
    return hooked


def _inputs() -> tuple:
    torch.manual_seed(7)
    return (
        torch.randn(1, 4, 16, 16, dtype=COMPUTE),
        torch.tensor([1]),
        torch.randn(1, 4, 32, dtype=COMPUTE),
    )


def _resident_bytes(module: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in list(module.parameters()) + list(module.buffers()):
        if t is None:
            continue
        key = str(t.dtype)
        out[key] = out.get(key, 0) + t.numel() * t.element_size()
    return out


def _nn_hook_count(module: Any) -> int:
    total = 0
    for m in module.modules():
        for attr in ("_forward_pre_hooks", "_forward_hooks"):
            total += len(getattr(m, attr, {}) or {})
        if getattr(m, "_diffusers_hook", None) is not None:
            total += 1
    return total


@pytest.fixture(scope="module")
def lanes() -> Dict[str, Any]:
    """One hook-lane UNet and one restructured UNet from identical weights."""
    hooked = _tiny_unet()
    restructured = copy.deepcopy(hooked)
    hooked_paths = _hook_lane(hooked)
    pipe = _Pipe(restructured)
    applied = apply_fp8_storage(pipe, compute_dtype=COMPUTE)
    assert applied is True
    return {
        "hooked": hooked,
        "hooked_paths": hooked_paths,
        "restructured": restructured,
        "pipe": pipe,
    }


def test_hook_lane_has_hooks_restructured_lane_has_none(lanes: Dict[str, Any]) -> None:
    assert len(lanes["hooked_paths"]) > 50, "the hook lane must actually hook"
    assert _nn_hook_count(lanes["hooked"]) >= len(lanes["hooked_paths"])
    assert _nn_hook_count(lanes["restructured"]) == 0
    assert not any(
        "forward" in m.__dict__ for m in lanes["restructured"].modules()
    ), "no instance-forward shadowing: the punned class owns forward"


def test_coverage_matches_diffusers_own_rule(lanes: Dict[str, Any]) -> None:
    """Anti-drift: our mirrored selection rule and the installed diffusers'
    rule must name the SAME leaves. If upstream changes its supported-layer
    set or default skip patterns, this fails here instead of on a pod."""
    restructured = lanes["restructured"]
    covered = sorted(fp8_storage.fp8_storage_leaves(restructured))
    assert covered == sorted(lanes["hooked_paths"])
    assert covered == sorted(fp8_storage.castable_leaves(lanes["hooked"]))


def test_weights_reside_in_fp8_and_round_trip(lanes: Dict[str, Any]) -> None:
    """Storage really is fp8, upcast really is the compute dtype, and the
    quantized values are the same bytes the hook lane holds."""
    hooked, restructured = lanes["hooked"], lanes["restructured"]
    checked = 0
    for path in lanes["hooked_paths"]:
        a = hooked.get_submodule(path)
        b = restructured.get_submodule(path)
        assert b.weight.dtype is STORAGE
        assert b.compute_dtype is COMPUTE
        assert b.storage_dtype is STORAGE
        assert torch.equal(a.weight.to(COMPUTE), b.weight.to(COMPUTE))
        if getattr(b, "bias", None) is not None:
            assert b.bias.dtype is STORAGE
            assert torch.equal(a.bias.to(COMPUTE), b.bias.to(COMPUTE))
        checked += 1
    assert checked == len(lanes["hooked_paths"])


def test_outputs_are_bitwise_equal_to_the_hook_lane(lanes: Dict[str, Any]) -> None:
    x, t, enc = _inputs()
    with torch.no_grad():
        want = lanes["hooked"](x, t, enc).sample
        have = lanes["restructured"](x, t, enc).sample
    assert torch.equal(want, have)


def test_residency_is_identical(lanes: Dict[str, Any]) -> None:
    """pgw#704 S12-c measured +11.6% VRAM for the restructure. That was the
    prototype swapping ``nn.Linear`` ONLY, leaving convs bf16-resident; at
    upstream coverage the trade does not exist."""
    assert _resident_bytes(lanes["hooked"]) == _resident_bytes(lanes["restructured"])


def test_restructured_lane_exports_and_the_hook_lane_is_refused(
    lanes: Dict[str, Any],
) -> None:
    args = _inputs()
    exported = torch.export.export(lanes["restructured"], args, strict=False)
    nodes = list(exported.graph_module.graph.nodes)
    fp8_valued = sum(
        1 for n in nodes
        if str(getattr(n.meta.get("val", None), "dtype", "")) == str(STORAGE)
    )
    assert fp8_valued > 0, "the resident fp8 buffers must be visible in the graph"
    with pytest.raises(RuntimeError, match="Couldn't swap"):
        torch.export.export(lanes["hooked"], args, strict=False)


def test_lane_value_kept_and_graph_contract_changed(lanes: Dict[str, Any]) -> None:
    """The wire lane value is unchanged ("fp8-hooks" — tensorhub maps it to
    w8a16); the traced graph is NOT, and the execution contract says so."""
    from gen_worker.compile_cache import execution_contract

    pipe = lanes["pipe"]
    assert pipe.unet._cozy_fp8_storage_applied is True
    assert pipeline_weight_lane(pipe) == "fp8-hooks"

    hooked_pipe = _Pipe(lanes["hooked"])
    hooked_pipe.unet._cozy_fp8_storage_applied = True

    class _Cfg:
        targets = ("unet",)

    hook_sig, hook_contract = execution_contract(hooked_pipe, _Cfg())
    new_sig, new_contract = execution_contract(pipe, _Cfg())
    assert hook_contract["lane"] == new_contract["lane"] == "fp8-hooks"
    assert hook_sig != new_sig, "restructured graphs must not adopt hook cells"


def test_idempotent_and_refuses_to_compose_with_hooks() -> None:
    unet = _tiny_unet()
    first = fp8_storage.restructure_fp8_storage(
        unet, storage_dtype=STORAGE, compute_dtype=COMPUTE)
    again = fp8_storage.restructure_fp8_storage(
        unet, storage_dtype=STORAGE, compute_dtype=COMPUTE)
    assert first == again and first
    assert apply_fp8_storage(_Pipe(unet), compute_dtype=COMPUTE) is True

    hooked = _tiny_unet()
    _hook_lane(hooked)
    with pytest.raises(ValueError, match="already armed"):
        fp8_storage.restructure_fp8_storage(
            hooked, storage_dtype=STORAGE, compute_dtype=COMPUTE)


def test_conv_transpose_and_embedding_leaves_restructure() -> None:
    """Coverage kinds sdxl's UNet does not exercise but other denoisers do."""
    import torch.nn as nn

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.ConvTranspose2d(4, 4, 2, stride=2)
            self.emb = nn.Embedding(8, 4)
            self.lin = nn.Linear(4, 4)

        def forward(self, x: Any, ids: Any) -> Any:
            return self.lin(self.emb(ids)).sum() + self.up(x).sum()

    plain = _Tiny().to(COMPUTE).eval()
    ref = copy.deepcopy(plain)
    from diffusers.hooks.layerwise_casting import apply_layerwise_casting

    apply_layerwise_casting(ref, STORAGE, COMPUTE)
    covered = fp8_storage.restructure_fp8_storage(
        plain, storage_dtype=STORAGE, compute_dtype=COMPUTE)
    assert covered == ["emb", "lin", "up"]

    x = torch.randn(1, 4, 4, 4, dtype=COMPUTE)
    ids = torch.tensor([0, 3])
    with torch.no_grad():
        assert torch.equal(ref(x, ids), plain(x, ids))


def test_fp8_storage_leaf_is_a_lora_branch_target() -> None:
    """gw#547/#627: the branch machinery selects by exact class. A pgw#727
    leaf must be targeted as the plain class it was restructured from, and its
    branch must compute in the compute dtype (never fp8)."""
    unet = _tiny_unet()
    plain_targets = set(w8a8_lora.branch_modules(unet))
    fp8_storage.restructure_fp8_storage(
        unet, storage_dtype=STORAGE, compute_dtype=COMPUTE)
    assert set(w8a8_lora.branch_modules(unet)) == plain_targets

    w8a8_lora.enable_lora_branches(unet, 16)
    mod = unet.get_submodule("down_blocks.1.attentions.0.proj_in")
    target = next(m for n, m in unet.named_modules()
                  if fp8_storage.is_fp8_storage_leaf(m)
                  and getattr(m, "lora_a", None) is not None)
    assert target.lora_a.dtype is COMPUTE
    assert w8a8_lora.branch_bucket(unet) == 16
    assert mod is not None
    w8a8_lora.disable_lora_branches(unet)
    assert getattr(target, "lora_a", None) is None


def test_model_dtype_still_reports_the_compute_dtype(lanes: Dict[str, Any]) -> None:
    """The defect this lane found (measured, CPU): diffusers resolves
    ``ModelMixin.dtype`` from the first floating-point PARAMETER, special-casing
    an armed layerwise-cast hook to return its ``compute_dtype``. Drop the hook
    and leave fp8 PARAMETERS and ``model.dtype`` starts answering fp8 — which
    silently breaks every denoiser that casts to ``self.dtype`` inside forward.
    fp8 storage therefore lives in BUFFERS."""
    assert lanes["hooked"].dtype is COMPUTE  # upstream's answer
    assert lanes["restructured"].dtype is COMPUTE  # ours must match it
    for leaf in fp8_storage.fp8_storage_leaves(lanes["restructured"]).values():
        assert "weight" in leaf._buffers and "weight" not in leaf._parameters
        assert all(p.dtype is COMPUTE for p in leaf.parameters())


def test_state_dict_keys_and_module_identity_are_preserved(
    lanes: Dict[str, Any],
) -> None:
    """A class pun, not a module replacement: same FQNs, same state_dict keys,
    same objects — so the offload rung, LoRA branch and every held reference
    keep working."""
    plain = _tiny_unet()
    assert sorted(plain.state_dict()) == sorted(lanes["restructured"].state_dict())
    for path in lanes["hooked_paths"]:
        leaf = lanes["restructured"].get_submodule(path)
        assert isinstance(leaf, type(plain.get_submodule(path)))


def test_a_denoiser_that_casts_to_self_dtype_runs_and_matches() -> None:
    """UNet2DModel's ``get_time_embed`` does ``t_emb.to(dtype=self.dtype)`` —
    the exact shape that fails when ``self.dtype`` is fp8."""
    from diffusers import UNet2DModel

    torch.manual_seed(0)
    hooked = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"), norm_num_groups=8,
    ).to(COMPUTE).eval()
    restructured = copy.deepcopy(hooked)
    hooked_paths = _hook_lane(hooked)
    covered = fp8_storage.restructure_fp8_storage(
        restructured, storage_dtype=STORAGE, compute_dtype=COMPUTE)
    assert sorted(covered) == sorted(hooked_paths)

    torch.manual_seed(3)
    x = torch.randn(1, 3, 8, 8, dtype=COMPUTE)
    t = torch.tensor([1])
    with torch.no_grad():
        assert torch.equal(hooked(x, t).sample, restructured(x, t).sample)


def test_scaled_linear_lora_slots_are_declared_buffers() -> None:
    """pgw#726: declared slots, so ``register_buffer`` does not have to fight
    a plain attribute and the FQN is a structural fact from construction."""
    cls = fp8_scaled_linear_class()
    mod = cls(8, 4, bias=False, compute_dtype=COMPUTE,
              static_input_scale=False, gemm_mode="pertensor")
    assert "lora_a" in mod._buffers and mod._buffers["lora_a"] is None
    assert "lora_b" in mod._buffers and mod._buffers["lora_b"] is None
    assert "lora_a" not in mod.__dict__
    # The shape that used to raise KeyError("attribute 'lora_a' already exists").
    mod.register_buffer("lora_a", torch.zeros(16, 8), persistent=False)
    assert mod.lora_a.shape == (16, 8)
    # Declared None slots stay out of the state_dict.
    assert not [k for k in mod.state_dict() if k.startswith("lora_")]


def test_scaled_linear_slots_survive_a_branch_disable_cycle() -> None:
    import torch.nn as nn

    cls = fp8_scaled_linear_class()

    class _Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = cls(8, 8, bias=False, compute_dtype=COMPUTE,
                         static_input_scale=False, gemm_mode="pertensor")
            self.q.weight = torch.zeros(8, 8, dtype=STORAGE)
            self.q.weight_scale = torch.ones(8, 1)

    model = _Denoiser()
    w8a8_lora.enable_lora_branches(model, 32)
    assert model.q.lora_a.shape == (32, 8)
    w8a8_lora.disable_lora_branches(model)
    assert "lora_a" in model.q._buffers and model.q._buffers["lora_a"] is None
    assert "lora_a" not in model.q.__dict__
    w8a8_lora.enable_lora_branches(model, 16)
    assert model.q.lora_a.shape == (16, 8)
