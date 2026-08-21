from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from gen_worker.api.errors import ValidationError
from gen_worker.models.lora_lifted import (
    LIFTED_INPUT_NAMES,
    assert_no_baked_adapter,
    bind_views,
    build_plan,
    install_lifted_lora_forward,
    lifted_binding,
    lifted_input_names,
    lora_constant_fqns,
    remove_lifted_lora_forward,
    resolve_slots,
    unbind_views,
)
from gen_worker.models.w8a8 import fp8_scaled_linear_class
from gen_worker.models.w8a8_lora import (
    alloc_branch_buffers,
    apply_branch_adapters,
    branch_modules,
    clear_branch_adapters,
    enable_lora_branches,
)

DIM, DEPTH, BUCKET, TOK = 64, 4, 16, 8
DT = torch.float32


class _Stack(nn.Module):

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.layers = nn.ModuleList(
            [nn.Linear(DIM, DIM, dtype=DT) for _ in range(DEPTH)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(DIM, dtype=DT) for _ in range(DEPTH)])
        with torch.no_grad():
            for lin in self.layers:
                lin.weight.copy_(
                    torch.randn(DIM, DIM, generator=g, dtype=DT) / DIM ** 0.5)
                lin.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin, norm in zip(self.layers, self.norms):
            x = norm(lin(x))
        return x


def _adapter_sd(model: nn.Module, rank: int, seed: int,
                alpha: float | None = None) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    sd: Dict[str, Any] = {}
    for path, mod in branch_modules(model).items():
        sd[f"{path}.lora_A.weight"] = torch.randn(
            rank, mod.in_features, generator=g, dtype=DT) * 0.05
        sd[f"{path}.lora_B.weight"] = torch.randn(
            mod.out_features, rank, generator=g, dtype=DT) * 0.05
        if alpha is not None:
            sd[f"{path}.alpha"] = torch.tensor(float(alpha))
    return sd


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got - want).abs().max()
            / want.abs().max().clamp(min=1e-12)).item()


def _eager_reference(x: torch.Tensor, *adapters: Dict[str, Any]) -> List[torch.Tensor]:
    model = _Stack().eval()
    enable_lora_branches(model, BUCKET)
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for sd in adapters:
            apply_branch_adapters(model, [(sd, 1.0, "ref")], uniform=True)
            outs.append(model(x).clone())
        clear_branch_adapters(model)
        outs.append(model(x).clone())
    return outs


def _lifted(
    x: torch.Tensor, model: Optional[nn.Module] = None,
) -> Tuple[nn.Module, Any]:
    model = model if model is not None else _Stack().eval()
    enable_lora_branches(model, BUCKET)
    return model, install_lifted_lora_forward(model)


def test_g2_swap_matches_eager_and_differs_export() -> None:
    """Swap A->B on ONE exported artifact: out(B) matches eager-B, out(A) matches eager-A, and the two differ."""
    x = torch.randn(TOK, DIM, dtype=DT)
    sd_a = _adapter_sd(_Stack(), rank=8, seed=1, alpha=16.0)
    sd_b = _adapter_sd(_Stack(), rank=8, seed=2, alpha=4.0)
    ref_a, ref_b, ref_zero = _eager_reference(x, sd_a, sd_b)
    assert ref_a.abs().max().item() < 1e3
    assert _rel(ref_a, ref_b) > 1e-3, "the two adapters must be distinguishable"

    model, binding = _lifted(x)
    binding.swap([(sd_a, 1.0, "A")])
    with torch.no_grad():
        ep = torch.export.export(
            model, (x,), dict(zip(LIFTED_INPUT_NAMES, binding.tensors)),
            strict=False)
    graph = ep.module()

    with torch.no_grad():
        out_a = graph(x, **binding.call_kwargs()).clone()
        binding.swap([(sd_b, 1.0, "B")])
        out_b = graph(x, **binding.call_kwargs()).clone()
        binding.clear()
        out_zero = graph(x, **binding.call_kwargs()).clone()

    assert _rel(out_a, ref_a) < 1e-5, f"lifted A vs eager A: {_rel(out_a, ref_a)}"
    assert _rel(out_b, ref_b) < 1e-5, f"lifted B vs eager B: {_rel(out_b, ref_b)}"
    assert _rel(out_zero, ref_zero) < 1e-5
    assert _rel(out_a, out_b) > 1e-3


def test_g2_swap_under_dynamo() -> None:
    """The same swap through torch.compile — one call convention serves both the dynamo and the export paths."""
    x = torch.randn(TOK, DIM, dtype=DT)
    sd_a = _adapter_sd(_Stack(), rank=8, seed=3, alpha=16.0)
    sd_b = _adapter_sd(_Stack(), rank=8, seed=4, alpha=16.0)
    ref_a, ref_b, _ref_zero = _eager_reference(x, sd_a, sd_b)

    model, binding = _lifted(x)
    compiled = torch.compile(model, dynamic=False)
    with torch.no_grad():
        binding.swap([(sd_a, 1.0, "A")])
        out_a = compiled(x, **binding.call_kwargs()).clone()
        binding.swap([(sd_b, 1.0, "B")])
        out_b = compiled(x, **binding.call_kwargs()).clone()
    assert _rel(out_a, ref_a) < 1e-4
    assert _rel(out_b, ref_b) < 1e-4
    assert _rel(out_a, out_b) > 1e-3


def test_multi_adapter_rank_concat_and_weight_fold() -> None:
    """Two adapters at once, non-unit user weights: the lifted swap reproduces the shipped rank-concat + scale fold exactly."""
    x = torch.randn(TOK, DIM, dtype=DT)
    sd_a = _adapter_sd(_Stack(), rank=4, seed=5, alpha=8.0)
    sd_b = _adapter_sd(_Stack(), rank=8, seed=6)
    pair = [(sd_a, 0.7, "A"), (sd_b, 0.3, "B")]

    ref = _Stack().eval()
    enable_lora_branches(ref, BUCKET)
    with torch.no_grad():
        apply_branch_adapters(ref, pair, uniform=True)
        want = ref(x).clone()

    model, binding = _lifted(x)
    binding.swap(pair)
    with torch.no_grad():
        got = model(x, **binding.call_kwargs())
    assert _rel(got, want) < 1e-6


def test_flat_pair_holds_exactly_the_shipped_buffer_values() -> None:
    """The flat pair is a RE-LAYOUT of the shipped buffers, not a new computation: every slot equals what the buffer-copy path writes."""
    sd = _adapter_sd(_Stack(), rank=8, seed=7, alpha=32.0)

    ref = _Stack().eval()
    enable_lora_branches(ref, BUCKET)
    apply_branch_adapters(ref, [(sd, 0.6, "R")], uniform=True)

    model, binding = _lifted(torch.randn(TOK, DIM, dtype=DT))
    binding.swap([(sd, 0.6, "R")])
    ref_mods = branch_modules(ref)
    seen = 0
    for path, a_view, b_view in binding.plan.views(*binding.tensors):
        assert torch.equal(a_view, ref_mods[path].lora_a), path
        assert torch.equal(b_view, ref_mods[path].lora_b), path
        seen += 1
    assert seen == DEPTH


def test_g3_lifted_export_has_no_lora_constant_and_carries_the_pair() -> None:
    x = torch.randn(TOK, DIM, dtype=DT)
    model, binding = _lifted(x)
    binding.swap([(_adapter_sd(_Stack(), rank=8, seed=8), 1.0, "A")])
    with torch.no_grad():
        ep = torch.export.export(
            model, (x,), dict(zip(LIFTED_INPUT_NAMES, binding.tensors)),
            strict=False)
    assert lora_constant_fqns(ep) == ()
    names = lifted_input_names(ep)
    for want in LIFTED_INPUT_NAMES:
        assert want in names, f"{want} missing from user inputs {names}"
    assert_no_baked_adapter(ep, label="lifted")


def test_g3_catches_a_baked_adapter() -> None:
    """RED proof: the pre-lifting shape — adapter halves as registered buffers — lands in the constant table, and the gate refuses NAMING the tensors."""
    x = torch.randn(TOK, DIM, dtype=DT)
    model = _Stack().eval()
    for lin in model.layers:
        for name in ("lora_a", "lora_b"):
            lin.__dict__.pop(name, None)
            lin._buffers.pop(name, None)
        lin.register_buffer("lora_a", torch.randn(BUCKET, DIM, dtype=DT) * 0.02)
        lin.register_buffer("lora_b", torch.randn(DIM, BUCKET, dtype=DT) * 0.02)
    with torch.no_grad():
        ep = torch.export.export(model, (x,), strict=False)

    baked = lora_constant_fqns(ep)
    assert baked, "a buffer-held adapter must show up in the constant table"
    with pytest.raises(ValidationError) as excinfo:
        assert_no_baked_adapter(ep, label="baked")
    message = str(excinfo.value)
    assert "BAKED" in message
    assert baked[0] in message


def test_g3_catches_a_traced_away_branch() -> None:
    """A MISSING pair is the same defect wearing a different hat: exporting the branchless graph serves the base model for every request."""
    x = torch.randn(TOK, DIM, dtype=DT)
    model = _Stack().eval()
    with torch.no_grad():
        ep = torch.export.export(model, (x,), strict=False)
    assert lora_constant_fqns(ep) == ()
    with pytest.raises(ValidationError, match="traced away"):
        assert_no_baked_adapter(ep, label="branchless")


def test_bucket_zero_is_its_own_class_and_refuses_lifting() -> None:
    model = _Stack().eval()
    with pytest.raises(ValidationError, match="rank bucket 0"):
        build_plan(model)
    with pytest.raises(ValidationError, match="rank bucket 0"):
        install_lifted_lora_forward(model)
    assert lifted_binding(model) is None


def test_a_HALF_supplied_adapter_pair_refuses_by_name() -> None:
    """One operand without the other is a caller error and stays one."""
    x = torch.randn(TOK, DIM, dtype=DT)
    model, binding = _lifted(x)
    a, b = binding.tensors
    with pytest.raises(ValidationError, match="'lora_b' is missing"):
        model(x, lora_a=a)
    with pytest.raises(ValidationError, match="'lora_a' is missing"):
        model(x, lora_b=b)


def test_a_plain_call_TAKES_THE_PLAIN_BRANCH_at_serving_time() -> None:
    x = torch.randn(TOK, DIM, dtype=DT)
    plain = _Stack().eval()
    with torch.no_grad():
        before = plain(x).clone()

    model, _binding = _lifted(x, model=plain)
    with torch.no_grad():
        after = model(x)

    assert torch.equal(before, after)


def test_wrong_operand_layout_refuses() -> None:
    x = torch.randn(TOK, DIM, dtype=DT)
    model, binding = _lifted(x)
    a, b = binding.tensors
    with pytest.raises(ValidationError, match="want a flat"):
        model(x, lora_a=a.view(BUCKET, -1), lora_b=b)
    with pytest.raises(ValidationError, match="want a flat"):
        model(x, lora_a=a[:-1], lora_b=b)
    with pytest.raises(ValidationError, match="want torch.float32"):
        model(x, lora_a=a.to(torch.float64), lora_b=b)


def test_over_bucket_adapter_refuses_instead_of_resizing() -> None:
    """A wider set is a different graph specialization — a lifted unit never recompiles at swap time."""
    x = torch.randn(TOK, DIM, dtype=DT)
    model, binding = _lifted(x)
    wide = _adapter_sd(_Stack(), rank=BUCKET * 2, seed=9)
    with pytest.raises(ValidationError, match="recompile at swap time"):
        binding.swap([(wide, 1.0, "wide")])


def test_sparse_placement_is_refused() -> None:
    """Sparse (eager-only) placement is per-coverage and would be a graph per adapter set — lifting requires canonical placement."""
    model = _Stack().eval()
    enable_lora_branches(model, BUCKET)
    lin = model.layers[1]
    for name in ("lora_a", "lora_b"):
        lin._buffers.pop(name, None)
        lin.__dict__.pop(name, None)
    lin.lora_a = None
    lin.lora_b = None
    with pytest.raises(ValidationError, match="CANONICAL"):
        build_plan(model, BUCKET)


def test_plan_layout_matches_the_shipped_allocation() -> None:
    model = _Stack().eval()
    enable_lora_branches(model, BUCKET)
    plan = build_plan(model, BUCKET)
    mods = branch_modules(model)
    assert plan.a_numel == sum(m.lora_a.numel() for m in mods.values())
    assert plan.b_numel == sum(m.lora_b.numel() for m in mods.values())
    assert [s.path for s in plan.slots] == sorted(mods)
    assert plan.bucket == BUCKET and plan.dtype is DT


def test_conv_slots_keep_their_branch_shapes() -> None:
    """gw#627 conv branches: A [bucket, in, kh, kw], B [out, bucket, 1, 1]."""
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, dtype=DT)
    holder = nn.Module()
    holder.add_module("conv", conv)
    alloc_branch_buffers(conv, BUCKET)
    plan = build_plan(holder, BUCKET)
    (slot,) = plan.slots
    assert slot.a_shape == (BUCKET, 3, 3, 3)
    assert slot.b_shape == (8, BUCKET, 1, 1)
    assert plan.a_numel == BUCKET * 3 * 3 * 3
    assert plan.b_numel == 8 * BUCKET


def test_w8a8_slot_home_is_the_registered_buffer() -> None:
    """The w8a8 lane reads ``lora_a``/``lora_b`` as non-persistent BUFFERS from its own forward (pre-quant addend, scale folded into B)."""
    cls = fp8_scaled_linear_class()
    mod = cls(DIM, DIM, bias=False, compute_dtype=torch.bfloat16,
              static_input_scale=False, gemm_mode="pertensor")
    holder = nn.Module()
    holder.add_module("q", mod)
    alloc_branch_buffers(mod, BUCKET)
    assert "lora_a" in mod._buffers and "lora_b" in mod._buffers
    assert not [k for k in mod.state_dict() if "lora" in k]

    plan = build_plan(holder, BUCKET)
    assert plan.dtype is torch.bfloat16
    a, b = plan.alloc(torch.device("meta"))
    original = (mod._buffers["lora_a"], mod._buffers["lora_b"])
    resolved = resolve_slots(holder, plan)
    prior = bind_views(resolved, plan, a, b)
    assert prior == (original,)
    assert mod._buffers["lora_a"].shape == (BUCKET, DIM)
    assert mod._buffers["lora_b"].shape == (DIM, BUCKET)
    assert mod._buffers["lora_a"] is not original[0]
    unbind_views(resolved, prior)
    assert mod._buffers["lora_a"] is original[0]
    assert mod._buffers["lora_b"] is original[1]


def test_install_is_idempotent_and_removal_restores_the_forward() -> None:
    x = torch.randn(TOK, DIM, dtype=DT)
    model = _Stack().eval()
    with torch.no_grad():
        base = model(x).clone()
    enable_lora_branches(model, BUCKET)
    first = install_lifted_lora_forward(model)
    assert install_lifted_lora_forward(model) is first
    assert lifted_binding(model) is first
    remove_lifted_lora_forward(model)
    assert lifted_binding(model) is None
    clear_branch_adapters(model)
    with torch.no_grad():
        assert _rel(model(x), base) < 1e-6
