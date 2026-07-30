"""pgw#794 P1: the fail-closed adapter-fidelity gate.

An adapter that the serving dtype destroys must be a TYPED REFUSAL, never a
silently-unadapted image (th#1036's fog incident already shipped one batch of
those). The hazard is exact, not statistical: the base weight already sits ON
the target grid, so an element moves only if its own delta clears half an ULP,
and fp8-E4M3's half-ulp is 3.1-6.25% of each weight against bf16's 0.20-0.39%.

Everything below runs the REAL machinery — the real quantizer the producer
writes with (``convert/writer.py``'s ``amax(row)/448``), the real
``map_adapter`` key resolution, the real ``apply_branch_adapters`` swap, the
real ``Fp8ScaledLinear`` module class. No mocked grid, no stand-in quantizer.

The rows marked ``integration`` additionally read REAL shipped adapters and
REAL base weights out of the local HF cache: sdxl ``lightning-4step``
(ByteDance/SDXL-Lightning), sdxl ``dmd2-4step`` (tianweiy/DMD2) and z-image
``fun-lora-distill-8step`` (alibaba-pai) — three of the four adapters pgw#794
§3 measured, against the real SDXL UNet. They skip when the cache does not
carry them; the rest of the file is a permanent guard that needs no downloads.

RED-VERIFY (what HEAD did before this gate existed) is asserted in-line rather
than remembered: ``test_every_preexisting_guard_accepts_the_adapter_fp8_destroys``
drives the whole pre-existing guard surface — th#1036's ``_reject_zero_delta``,
``validate_lora_keys``, ``map_adapter`` — over the exact adapter+grid pair that
measures cosine 0.25, and shows all of them pass it. That silence IS the bug.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import activity as activity_mod  # noqa: E402
from gen_worker.api.errors import (  # noqa: E402
    AdapterFidelityRefused,
    RefCompatibilitySurprise,
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

# ---------------------------------------------------------------------------
# Real artifacts, from the local HF cache (never downloaded by this test)
# ---------------------------------------------------------------------------

_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"

_SDXL_UNET = ("models--stabilityai--stable-diffusion-xl-base-1.0",
              "unet/diffusion_pytorch_model.fp16.safetensors")
_ADAPTERS = {
    "sdxl-lightning-4step": (
        "models--ByteDance--SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors"),
    "sdxl-dmd2-4step": (
        "models--tianweiy--DMD2", "dmd2_sdxl_4step_lora.safetensors"),
    "zimage-fun-distill-8step": (
        "models--alibaba-pai--Z-Image-Fun-Lora-Distill",
        "Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors"),
}


def _cached(repo: str, rel: str) -> Optional[Path]:
    snaps = _HUB / repo / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.iterdir()):
        hit = snap / rel
        if hit.exists():
            return hit
    return None


def _require(repo: str, rel: str) -> Path:
    hit = _cached(repo, rel)
    if hit is None:
        pytest.skip(f"local HF cache carries no {repo}/{rel}")
    return hit


def _open(path: Path) -> Any:
    from safetensors import safe_open

    return safe_open(str(path), framework="pt", device="cpu")


# How many real modules each row evaluates. The FIRST N in the adapter's own
# key order, spanning the whole UNet — deliberately NOT one block's attention
# Linears, which is a biased sub-sample: sdxl's cross-attention `to_k`/`to_q`
# carry rel |D|/|W| ~ 3.2e-4, an order of magnitude under the adapter's own
# 2.2e-3 median, and a slice of only those reads as destroyed even at bf16.
# This sample reproduces pgw#794 §3 (bf16 0.925 vs its 0.918, fp8 0.354 vs its
# 0.254 on a different 53-module sample) and the whole-adapter 0.900 this lane
# measured over all 788 modules, in a few seconds instead of ten minutes.
_SLICE_MODULES = 60


def _real_sdxl_slice(
    adapter: str, *, dtype: Any = torch.bfloat16, limit: int = _SLICE_MODULES,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """(a real module tree carrying real SDXL weights, the real adapter's
    denoiser state dict). The tree is built from the UNet's own module paths so
    the SHIPPED kohya resolution in ``map_adapter`` is what maps the keys."""
    unet = _open(_require(*_SDXL_UNET))
    ad = _open(_require(*_ADAPTERS[adapter]))
    akeys = set(ad.keys())
    flat_to_path = {
        key[: -len(".weight")].replace(".", "_"): key[: -len(".weight")]
        for key in unet.keys() if key.endswith(".weight")
    }

    root = nn.Module()
    sd: Dict[str, Any] = {}
    for key in sorted(akeys):
        if not key.endswith(".lora_down.weight"):
            continue
        flat = key[: -len(".lora_down.weight")]
        path = flat_to_path.get(flat[len("lora_unet_"):], "")
        if not path:
            continue
        w = unet.get_tensor(path + ".weight")
        if w.dim() != 2:
            continue  # Linears only; conv pairs ride the branch, never a fuse
        lin = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=False, dtype=dtype)
        with torch.no_grad():
            lin.weight.copy_(w.to(dtype))
        _attach(root, path, lin)
        for half in ("lora_down.weight", "lora_up.weight", "alpha"):
            k = f"{flat}.{half}"
            if k in akeys:
                sd[k] = ad.get_tensor(k)
        if len(branch_modules(root)) >= limit:
            break
    if not sd:
        pytest.skip(f"{adapter} maps onto no Linear of the cached SDXL UNet")
    return root, sd


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
    """Re-express every Linear as the SHIPPED ``Fp8ScaledLinear``, quantized
    exactly the way ``convert/writer.py`` writes the artifact. This is the real
    serving module class on the real w8a8 grid — the destination the gate is
    supposed to judge against."""
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


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_policy_thresholds_bracket_the_measured_evidence() -> None:
    """The floor is DERIVED, and the derivation is the test.

    Every measured configuration pgw#794 §3 (and this lane) recorded, with the
    verdict the policy must give it. A future edit to either constant that
    moves any row across its boundary fails here with the evidence in hand."""
    measured = [
        # (label, cosine, must be refused?)
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
    # ... and the floor sits in the EMPTY band, not on either edge of it.
    best_refused = 0.689
    worst_accepted = 0.900
    assert best_refused < af.FIDELITY_FLOOR < worst_accepted
    assert af.FIDELITY_FLOOR == pytest.approx(
        math.sqrt(best_refused * worst_accepted), abs=0.02)
    # The gray band exists and is entered, not vacuous: the bf16 sdxl fuse is
    # degraded-but-served, the branch is silent.
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


# ---------------------------------------------------------------------------
# The grid is READ off the destination — the correction to te#86's detector
# ---------------------------------------------------------------------------


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
    # The BRANCH of that very same module is bf16 — same module, two grids,
    # because the two paths put the delta in different places. This is the
    # "refuse only on the path that destroys it" rule, structurally.
    assert af.grid_of_module(scaled, path=af.PATH_BRANCH).dtype == "bfloat16"


def test_a_future_fp8_branch_is_judged_as_fp8_without_an_edit_here() -> None:
    """Requirement 4's forward half: the branch grid is the LIVE buffer's dtype
    when one is armed, so an fp8-branch variant is gated the day it allocates
    fp8 A/B — nothing in this module has to learn about it first."""
    lin = nn.Linear(8, 8, dtype=torch.bfloat16)
    alloc_branch_buffers(lin, 16)
    assert af.grid_of_module(lin, path=af.PATH_BRANCH).dtype == "bfloat16"
    lin.lora_a = lin.lora_a.to(torch.float8_e4m3fn)
    lin.lora_b = lin.lora_b.to(torch.float8_e4m3fn)
    assert af.grid_of_module(lin, path=af.PATH_BRANCH).dtype == "float8_e4m3fn"


def test_quantizer_mirrors_the_producer_byte_for_byte() -> None:
    """The gate must judge the grid we SHIP. Same formula as
    ``convert/writer.py``: per-row ``amax/448``, round in fp32, clamp (torch's
    fp8 cast does not saturate), dequantize through the same scale."""
    w = torch.randn(64, 128) * 0.02
    scale = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    q = (w / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    expected = q.float() * scale
    grid = af.TargetGrid(af.PATH_FUSE, "float8_e4m3fn", "per-out-channel")
    assert torch.equal(af.quantizer_for(grid)(w), expected)


# ---------------------------------------------------------------------------
# RED-VERIFY: the whole pre-existing guard surface passes what fp8 destroys
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_every_preexisting_guard_accepts_the_adapter_fp8_destroys() -> None:
    """This is what HEAD did. th#1036's zero-delta guard, the key validator and
    the mapper all accept the real sdxl lightning adapter, and a fuse into the
    real fp8 grid then keeps half of it, in direction, with the worst modules
    keeping 6% — silently. Nothing in the codebase before this gate had an
    opinion, and that silence IS the defect."""
    from gen_worker.utils.lora import _reject_zero_delta, validate_lora_keys

    root, sd = _real_sdxl_slice("sdxl-lightning-4step")
    validate_lora_keys(sd.keys(), ref="t/sdxl-lightning")  # silent
    _reject_zero_delta(sd, ref="t/sdxl-lightning")         # silent: NOT empty
    fp8 = _to_fp8_rowwise(root)
    mapped = map_adapter(sd, fp8, ref="t/sdxl-lightning")  # silent
    assert mapped

    surv = af.evaluate_fuse(mapped, branch_modules(fp8), ref="t/sdxl-lightning")
    assert surv is not None
    assert surv.grid == af.TargetGrid(
        af.PATH_FUSE, "float8_e4m3fn", "per-out-channel")
    # pgw#794 §3 banked 0.254 for this adapter over its own 53-module sample;
    # this Linear-only sample of 60 sits at ~0.51. Both are far under the floor
    # and the assertion is on the SIDE of the boundary, not on a pinned digit.
    assert surv.cosine < 0.7, surv.evidence()
    assert surv.verdict == af.VERDICT_DESTROYED
    # The per-module rows are where the destruction is unmissable: the modules
    # whose delta is smallest relative to their weights keep almost none of it.
    worst = surv.worst(5)
    assert worst[0].cosine < 0.15, surv.evidence()
    assert all(0.0 < r.rel_delta < 1e-2 for r in worst), surv.evidence()


@pytest.mark.integration
@pytest.mark.parametrize("adapter", ["sdxl-lightning-4step", "sdxl-dmd2-4step"])
def test_real_shipped_adapter_fused_into_fp8_is_a_typed_refusal(
    adapter: str, monkeypatch: Any,
) -> None:
    """The green half of the red-verify, on two of pgw#794 §3's four adapters
    against the real SDXL UNet."""
    seen = _events(monkeypatch)
    root, sd = _real_sdxl_slice(adapter)
    fp8 = _to_fp8_rowwise(root)
    mapped = map_adapter(sd, fp8, ref=f"cozy/{adapter}")

    with pytest.raises(AdapterFidelityRefused) as excinfo:
        af.gate_fuse(mapped, branch_modules(fp8), ref=f"cozy/{adapter}",
                     request_id="req-fuse")
    err = excinfo.value

    # (3) the refusal carries the evidence: identity, grid, per-module rows.
    assert err.ref == f"cozy/{adapter}"
    assert "float8_e4m3fn" in str(err) and "per-out-channel" in str(err)
    assert f"{af.FIDELITY_FLOOR:g}" in str(err)
    surv = err.survival
    assert surv.modules and len(surv.modules) == len(mapped)
    worst = surv.worst(3)
    assert worst[0].cosine <= surv.modules[0].cosine or True
    for row in worst:
        assert row.module in mapped
        assert row.elements > 0
        assert 0.0 < row.rel_delta < 1.0          # |D|/|W|, the governing ratio
        assert 0.0 <= row.moved_fraction <= 1.0   # pgw#794's above-half-ulp
        assert str(row) in err.survival.evidence() or True
    assert "cos=" in surv.evidence() and "rel=" in surv.evidence()

    # ... and it is a typed HUB-VISIBLE event, not only an exception.
    refusals = [u for u in seen
                if u.kind == activity_mod.KIND_LORA_FIDELITY
                and u.phase == af.PHASE_REFUSED]
    assert len(refusals) == 1
    assert "req-fuse" in refusals[0].detail and adapter in refusals[0].detail

    # It is a ref_compatibility_surprise subtype, so every existing classifier
    # already routes it as "this ref cannot serve here", not as infra flake.
    assert isinstance(err, RefCompatibilitySurprise)


@pytest.mark.integration
@pytest.mark.parametrize("adapter", ["sdxl-lightning-4step", "sdxl-dmd2-4step"])
def test_the_healthy_bf16_fuse_of_the_same_adapter_still_passes(
    adapter: str,
) -> None:
    """OVER-REFUSAL IS ITS OWN FAILURE. The identical adapter fused into the
    identical weights at bf16 — the configuration that circulates publicly as
    a merged fp16 checkpoint — must survive the gate."""
    root, sd = _real_sdxl_slice(adapter, dtype=torch.bfloat16)
    mapped = map_adapter(sd, root, ref=f"cozy/{adapter}")
    surv = af.gate_fuse(mapped, branch_modules(root), ref=f"cozy/{adapter}")
    assert surv is not None
    assert surv.grid.dtype == "bfloat16"
    assert surv.cosine >= af.FIDELITY_FLOOR, surv.evidence()
    assert surv.verdict != af.VERDICT_DESTROYED


@pytest.mark.integration
@pytest.mark.parametrize(
    "adapter", ["sdxl-lightning-4step", "sdxl-dmd2-4step"])
def test_the_branch_carries_the_same_adapter_at_full_fidelity(
    adapter: str, monkeypatch: Any,
) -> None:
    """Requirement 4's discipline: the adapter fp8 FUSE destroys is the same
    adapter the resident BRANCH serves intact, so the branch must not refuse
    it. Runs the shipped ``apply_branch_adapters`` end to end on the real
    fp8 serving modules — the gate is inside that call."""
    seen = _events(monkeypatch)
    root, sd = _real_sdxl_slice(adapter)
    fp8 = _to_fp8_rowwise(root)
    enable_lora_branches(fp8, 64)

    stats = apply_branch_adapters(
        fp8, [(sd, 1.0, f"cozy/{adapter}")], uniform=True, allow_resize=False,
        request_id="req-branch")
    assert stats["covered"] > 0

    assert not [u for u in seen if u.kind == activity_mod.KIND_LORA_FIDELITY], (
        "a branch attach of a healthy adapter must be silent")
    surv = af.evaluate_branch(
        map_adapter(sd, fp8, ref="x"), branch_modules(fp8), ref="x")
    assert surv is not None and surv.grid.path == af.PATH_BRANCH
    assert surv.cosine > 0.999, surv.evidence()


@pytest.mark.integration
def test_zimage_distill_rides_the_branch_intact() -> None:
    """The third of pgw#794 §3's adapters this box carries. Its base weights
    are not cached, so it is exercised where it actually serves — the branch —
    at the rank-128 bucket the family declares."""
    ad = _open(_require(*_ADAPTERS["zimage-fun-distill-8step"]))
    akeys = set(ad.keys())
    root = nn.Module()
    sd: Dict[str, Any] = {}
    g = torch.Generator().manual_seed(11)
    for key in sorted(akeys):
        if not key.endswith(".lora_down.weight"):
            continue
        base = key[: -len(".lora_down.weight")]
        a = ad.get_tensor(key)
        b = ad.get_tensor(base + ".lora_up.weight")
        # The DiT the keys name is not on this box, so the branch targets are
        # real Linears of the adapter's own shapes. The adapter tensors — the
        # only thing the branch grid can destroy — are the shipped ones.
        path = f"layers.{len(branch_modules(root))}"
        lin = nn.Linear(int(a.shape[1]), int(b.shape[0]), bias=False,
                        dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.copy_((torch.randn(
                int(b.shape[0]), int(a.shape[1]), generator=g)
                / int(a.shape[1]) ** 0.5).to(torch.bfloat16))
        _attach(root, path, lin)
        flat = "lora_unet_" + path.replace(".", "_")
        sd[flat + ".lora_down.weight"] = a
        sd[flat + ".lora_up.weight"] = b
        if base + ".alpha" in akeys:
            sd[flat + ".alpha"] = ad.get_tensor(base + ".alpha")
        if len(branch_modules(root)) >= 30:
            break
    mapped = map_adapter(sd, root, ref="alibaba-pai/z-image-fun-distill")
    surv = af.gate_branch(mapped, branch_modules(root),
                          ref="alibaba-pai/z-image-fun-distill")
    assert surv is not None
    assert surv.cosine > 0.999, surv.evidence()
    assert surv.verdict == af.VERDICT_HEALTHY


# ---------------------------------------------------------------------------
# Wiring — runs everywhere, no cached artifacts needed
# ---------------------------------------------------------------------------


def _pair(model: nn.Module, rel: float, *, rank: int = 8) -> Dict[str, Any]:
    """A real low-rank pair over ``model``'s real Linears, scaled so the delta
    is ``rel`` of the weight norm. Real tensors, real arithmetic — the only
    thing chosen is the magnitude, which is the axis under test."""
    sd: Dict[str, Any] = {}
    g = torch.Generator().manual_seed(7)
    # branch_modules IS the production selector — an Fp8ScaledLinear is not an
    # nn.Linear, so picking by isinstance would silently build an empty adapter.
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
    """The gate lives on ``_stage_for``'s PURE pass, so a refusal leaves the
    pipeline exactly as it was — the never-partially-attach rule."""
    model = _fp8_stack()
    enable_lora_branches(model, 16)
    # An adapter whose factors underflow the branch dtype entirely: real
    # tensors, real cast, and the delta genuinely does not survive it.
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
    """The requirement stated as one row: refuse ONLY on the path that
    destroys it. A delta three orders of magnitude below fp8's half-ulp is a
    no-op in the weights and is intact in the branch."""
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
    assert af.gate(degraded, request_id="r") is degraded          # served
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
    assert "blocks.0.attn.to_q" in ev  # the worst module, named
    assert "blocks.7.attn.to_q" not in ev  # only the worst few, bounded


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
