"""The fail-closed adapter-fidelity gate."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import activity as activity_mod  # noqa: E402
from gen_worker.api.errors import (  # noqa: E402
    AdapterFidelityRefused,
)
from gen_worker.models import adapter_fidelity as af  # noqa: E402
from gen_worker.models.w8a8 import fp8_scaled_linear_class  # noqa: E402
from gen_worker.models.w8a8_lora import (  # noqa: E402
    alloc_branch_buffers,
    apply_branch_adapters,
    branch_modules,
    enable_lora_branches,
    map_adapter,
)

def _attach(root: nn.Module, path: str, leaf: nn.Module) -> None:
    parts = path.split(".")
    cur = root
    for part in parts[:-1]:
        nxt = getattr(cur, part, None)
        if not isinstance(nxt, nn.Module):
            nxt = nn.Module()
            cur.add_module(part, nxt)
        cur = nxt
    cur.add_module(parts[-1], leaf)


def _to_fp8_rowwise(root: nn.Module) -> nn.Module:
    cls = fp8_scaled_linear_class()
    for name, mod in list(root.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        w = mod.weight.detach().float()
        scale = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
        q = (w / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        leaf = cls(int(w.shape[1]), int(w.shape[0]), bias=False,
                   compute_dtype=torch.bfloat16, static_input_scale=False,
                   gemm_mode="rowwise")
        leaf.to_empty(device="cpu")
        with torch.no_grad():
            leaf.weight.copy_(q)
            leaf.weight_scale.copy_(scale)
        parent, _, leafname = name.rpartition(".")
        target = root.get_submodule(parent) if parent else root
        target.add_module(leafname, leaf)
    return root


def _events(monkeypatch: Any) -> List[Any]:
    seen: List[Any] = []
    monkeypatch.setattr(activity_mod, "_sink", seen.append)
    return seen


def test_policy_thresholds_bracket_the_measured_evidence() -> None:
    """The floor is DERIVED, and the derivation is the test."""
    measured = [
        ("qwen lightning-8step, fp8 fuse (§3)", 0.074, True),
        ("sdxl lightning-4step, fp8 fuse (§3)", 0.254, True),
        ("sdxl dmd2-4step, fp8 fuse (§3)", 0.258, True),
        ("z-image fun-distill, fp8 fuse (§3, strongest we ship)", 0.689, True),
        ("qwen lightning-8step, bf16 fuse (§3)", 0.503, True),
        ("sdxl lightning-4step, bf16 fuse (whole adapter, this lane)", 0.900, False),
        ("sdxl lightning-4step, bf16 fuse (§3 53-module sample)", 0.918, False),
        ("z-image fun-distill, bf16 fuse (§3)", 1.000, False),
        ("any adapter, bf16 branch (this lane)", 1.000, False),
        ("z-image fun-distill, fp8 BRANCH (this lane)", 0.9998, False),
    ]
    for label, cosine, refused in measured:
        assert (cosine < af.FIDELITY_FLOOR) is refused, label
    best_refused = 0.689
    worst_accepted = 0.900
    assert best_refused < af.FIDELITY_FLOOR < worst_accepted
    assert af.FIDELITY_FLOOR == pytest.approx(
        math.sqrt(best_refused * worst_accepted), abs=0.02)
    assert af.FIDELITY_FLOOR < 0.900 < af.FIDELITY_WARN < 1.0


def test_verdicts_are_the_two_tier_shape() -> None:
    grid = af.TargetGrid(af.PATH_FUSE, "float8_e4m3fn", "per-out-channel")

    def verdict(cos: float) -> str:
        return af.AdapterSurvival("r", grid, (), cos, 1.0).verdict

    assert verdict(0.9999) == af.VERDICT_HEALTHY
    assert verdict(af.FIDELITY_WARN) == af.VERDICT_HEALTHY
    assert verdict(0.90) == af.VERDICT_DEGRADED
    assert verdict(af.FIDELITY_FLOOR) == af.VERDICT_DEGRADED
    assert verdict(af.FIDELITY_FLOOR - 1e-9) == af.VERDICT_DESTROYED
    assert verdict(0.074) == af.VERDICT_DESTROYED


def test_grid_is_read_from_the_module_not_from_the_source_dtype() -> None:
    plain = nn.Linear(8, 8, dtype=torch.bfloat16)
    assert af.grid_of_module(plain, path=af.PATH_FUSE) == af.TargetGrid(
        af.PATH_FUSE, "bfloat16", "none")

    cls = fp8_scaled_linear_class()
    scaled = cls(8, 8, bias=False, compute_dtype=torch.bfloat16,
                 static_input_scale=False, gemm_mode="rowwise")
    scaled.to_empty(device="cpu")
    with torch.no_grad():
        scaled.weight_scale.fill_(1.0)
    grid = af.grid_of_module(scaled, path=af.PATH_FUSE)
    assert grid == af.TargetGrid(af.PATH_FUSE, "float8_e4m3fn", "per-out-channel")
    assert af.grid_of_module(scaled, path=af.PATH_BRANCH).dtype == "bfloat16"


def test_a_future_fp8_branch_is_judged_as_fp8_without_an_edit_here() -> None:
    """Requirement 4's forward half: the branch grid is the LIVE buffer's dtype when one is armed, so an fp8-branch variant is gated the day it allocates fp8 A/B — nothing in this module has to learn abou..."""
    lin = nn.Linear(8, 8, dtype=torch.bfloat16)
    alloc_branch_buffers(lin, 16)
    assert af.grid_of_module(lin, path=af.PATH_BRANCH).dtype == "bfloat16"
    lin.lora_a = lin.lora_a.to(torch.float8_e4m3fn)
    lin.lora_b = lin.lora_b.to(torch.float8_e4m3fn)
    assert af.grid_of_module(lin, path=af.PATH_BRANCH).dtype == "float8_e4m3fn"


def test_quantizer_mirrors_the_producer_byte_for_byte() -> None:
    """The gate must judge the grid we SHIP."""
    w = torch.randn(64, 128) * 0.02
    scale = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    q = (w / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    expected = q.float() * scale
    grid = af.TargetGrid(af.PATH_FUSE, "float8_e4m3fn", "per-out-channel")
    assert torch.equal(af.quantizer_for(grid)(w), expected)


def _pair(model: nn.Module, rel: float, *, rank: int = 8) -> Dict[str, Any]:
    sd: Dict[str, Any] = {}
    g = torch.Generator().manual_seed(7)
    for name, mod in branch_modules(model).items():
        a = torch.randn(rank, mod.in_features, generator=g)
        b = torch.randn(mod.out_features, rank, generator=g)
        delta = b @ a
        w = af._base_weight(mod).detach().float()
        b = b * (rel * float(w.norm()) / float(delta.norm()))
        flat = "lora_unet_" + name.replace(".", "_")
        sd[flat + ".lora_down.weight"] = a.to(torch.bfloat16)
        sd[flat + ".lora_up.weight"] = b.to(torch.bfloat16)
        sd[flat + ".alpha"] = torch.tensor(float(rank))
    return sd


def _fp8_stack(width: int = 128, depth: int = 4) -> nn.Module:
    root = nn.Module()
    g = torch.Generator().manual_seed(3)
    for i in range(depth):
        lin = nn.Linear(width, width, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.copy_((torch.randn(width, width, generator=g)
                              / width ** 0.5).to(torch.bfloat16))
        _attach(root, f"layers.{i}", lin)
    return _to_fp8_rowwise(root)


def test_the_branch_attach_of_an_inert_adapter_refuses_before_a_buffer_moves(
    monkeypatch: Any,
) -> None:
    """The gate lives on ``_stage_for``'s PURE pass, so a refusal leaves the pipeline exactly as it was — the never-partially-attach rule."""
    model = _fp8_stack()
    enable_lora_branches(model, 16)
    sd = _pair(model, rel=1e-3, rank=8)
    for key in list(sd):
        if key.endswith("lora_down.weight"):
            sd[key] = (sd[key].float() * 1e-30).to(torch.bfloat16)

    before = {n: m.lora_b.clone() for n, m in branch_modules(model).items()}
    seen = _events(monkeypatch)
    with pytest.raises(AdapterFidelityRefused):
        apply_branch_adapters(model, [(sd, 1.0, "t/inert")], uniform=True,
                              allow_resize=False, request_id="req-1")
    for name, mod in branch_modules(model).items():
        assert torch.equal(mod.lora_b, before[name]), (
            f"{name}: a refused attach must not have touched a buffer")
    assert [u.phase for u in seen
            if u.kind == activity_mod.KIND_LORA_FIDELITY] == [af.PHASE_REFUSED]


def test_a_healthy_branch_attach_is_silent_and_lands(monkeypatch: Any) -> None:
    model = _fp8_stack()
    enable_lora_branches(model, 16)
    sd = _pair(model, rel=3e-3, rank=8)
    seen = _events(monkeypatch)
    stats = apply_branch_adapters(model, [(sd, 1.0, "t/healthy")], uniform=True,
                                  allow_resize=False, request_id="req-2")
    assert stats["covered"] == 4
    assert not [u for u in seen if u.kind == activity_mod.KIND_LORA_FIDELITY]
    assert any(float(m.lora_b.abs().sum()) > 0
               for m in branch_modules(model).values())


def test_the_same_adapter_refuses_on_fuse_and_passes_on_branch() -> None:
    """The requirement stated as one row: refuse ONLY on the path that destroys it."""
    model = _fp8_stack()
    sd = _pair(model, rel=3e-5, rank=8)
    mapped = map_adapter(sd, model, ref="t/tiny")
    mods = branch_modules(model)

    fuse = af.evaluate_fuse(mapped, mods, ref="t/tiny")
    branch = af.evaluate_branch(mapped, mods, ref="t/tiny")
    assert fuse is not None and branch is not None
    assert fuse.verdict == af.VERDICT_DESTROYED, fuse.evidence()
    assert branch.verdict == af.VERDICT_HEALTHY, branch.evidence()
    with pytest.raises(AdapterFidelityRefused):
        af.gate_fuse(mapped, mods, ref="t/tiny")
    passed = af.gate_branch(mapped, mods, ref="t/tiny")
    assert passed is not None and passed.verdict == af.VERDICT_HEALTHY
    assert passed.cosine == pytest.approx(branch.cosine)


def test_gray_band_serves_but_confesses_once(monkeypatch: Any) -> None:
    grid = af.TargetGrid(af.PATH_FUSE, "bfloat16")
    degraded = af.AdapterSurvival(
        "cozy/sdxl-lightning", grid,
        (af.ModuleSurvival("mid.attn.to_q", 1024, 3.2e-3, 1.11, 0.87, 0.45),),
        cosine=0.900, retention=1.111)
    seen = _events(monkeypatch)
    assert af.gate(degraded, request_id="r") is degraded
    assert af.gate(degraded, request_id="r", announce=False) is degraded
    rows = [u for u in seen if u.kind == activity_mod.KIND_LORA_FIDELITY]
    assert [u.phase for u in rows] == [af.PHASE_DEGRADED]
    assert "cozy/sdxl-lightning" in rows[0].detail
    assert "mid.attn.to_q" in rows[0].detail and "bfloat16" in rows[0].detail


def test_evidence_names_identity_grid_aggregate_and_worst_modules() -> None:
    grid = af.TargetGrid(af.PATH_FUSE, "float8_e4m3fn", "per-out-channel")
    surv = af.AdapterSurvival(
        "cozy/qwen-lightning-8step", grid,
        tuple(af.ModuleSurvival(f"blocks.{i}.attn.to_q", 4096, 3.7e-4,
                                15.29, 0.074 + i * 0.01, 0.005)
              for i in range(8)),
        cosine=0.074, retention=15.29)
    ev = surv.evidence()
    assert "cozy/qwen-lightning-8step" in ev
    assert "fuse:float8_e4m3fn, per-out-channel" in ev
    assert "cosine=0.0740" in ev and "retention=15.290" in ev
    assert f"floor={af.FIDELITY_FLOOR:g}" in ev
    assert "blocks.0.attn.to_q" in ev
    assert "blocks.7.attn.to_q" not in ev


def test_a_mixed_grid_set_is_judged_on_its_coarsest_rung() -> None:
    """The gate must never be softened by the friendliest layer in the set."""
    model = _fp8_stack(width=64, depth=2)
    plain = nn.Linear(64, 64, bias=False, dtype=torch.float32)
    _attach(model, "layers.2", plain)
    sd = _pair(model, rel=3e-4, rank=8)
    mapped = map_adapter(sd, model, ref="t/mixed")
    surv = af.evaluate_fuse(mapped, branch_modules(model), ref="t/mixed")
    assert surv is not None
    assert surv.grid.dtype == "float8_e4m3fn"


def test_evaluators_return_none_when_nothing_maps() -> None:
    model = _fp8_stack(width=32, depth=1)
    mods = branch_modules(model)
    assert af.evaluate_branch({}, mods, ref="t/empty") is None
    assert af.evaluate_fuse({}, mods, ref="t/empty") is None
    assert af.gate(None) is None
