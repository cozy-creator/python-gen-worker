"""pgw#860 — native-kernel dispatch: env gating, probe fallback, extension
probe, and the fused-lane swap seam. CPU-only; kernel numerics run on the
pgw#865 GPU harness."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gen_worker.models import native_kernels as nk  # noqa: E402
from gen_worker.models import svdq_fused  # noqa: E402
from gen_worker.models import svdq_native as native  # noqa: E402
from gen_worker.models.nvfp4_quant import BLOCK, pack_e2m1  # noqa: E402
from gen_worker.models.svdq_layout import DecodedLinear  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_arming(monkeypatch: pytest.MonkeyPatch):
    nk.reset_native_kernels_arming()
    yield
    nk.reset_native_kernels_arming()


# --- env gating ------------------------------------------------------------


def test_unset_env_stays_baseline_dormant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(nk.NATIVE_ENV, raising=False)
    assert nk.svdq_linear_execution_lane() == "baseline"
    assert "env-gated" in nk.svdq_linear_execution_lane_reason()
    assert nk.svdq_modulation_execution_lane() == "dense"
    assert "env-gated" in nk.svdq_modulation_execution_lane_reason()


def test_kill_switch_forces_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(nk.NATIVE_ENV, "0")
    assert nk.svdq_linear_execution_lane() == "baseline"
    assert "kill-switch" in nk.svdq_linear_execution_lane_reason()
    assert nk.svdq_modulation_execution_lane() == "dense"
    assert "kill-switch" in nk.svdq_modulation_execution_lane_reason()


def test_format_one_has_one_deterministic_native_kernel_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TCG's closed metadata cannot hide a second worker-local verdict."""
    monkeypatch.setenv(nk.NATIVE_ENV, "1")
    assert nk.svdq_linear_execution_lane() == "baseline"
    assert nk.svdq_modulation_execution_lane() == "dense"
    assert "format 1" in nk.svdq_linear_execution_lane_reason()
    assert "format 1" in nk.svdq_modulation_execution_lane_reason()


def test_no_sm_allowlist_survives() -> None:
    """No hardware tuple silently chooses a worker-local artifact policy."""
    assert not hasattr(nk, "FUSED_SMS")
    assert not hasattr(nk, "FUSED_LINEAR_SMS")
    assert not hasattr(nk, "PACKED_MODULATION_SMS")
    assert not hasattr(nk, "BLACKWELL_SMS")


# --- extension probe -------------------------------------------------------


def test_extension_absent_is_typed_not_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    monkeypatch.delenv(nk.NATIVE_LIB_ENV, raising=False)
    monkeypatch.setattr(nk, "_EXT_DEFAULT", str(tmp_path / "nope.so"))
    reason = nk.load_extension()
    assert reason is not None and "no extension library" in reason
    assert nk.extension_available() is False


def test_extension_env_override_bad_path_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    bad = tmp_path / "missing.so"
    monkeypatch.setenv(nk.NATIVE_LIB_ENV, str(bad))
    reason = nk.load_extension()
    assert reason is not None and str(bad) in reason


def test_extension_garbage_so_fails_load_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    junk = tmp_path / "junk.so"
    junk.write_bytes(b"\x7fELF not really")
    monkeypatch.setenv(nk.NATIVE_LIB_ENV, str(junk))
    reason = nk.load_extension()
    assert reason is not None and "failed to load" in reason
    assert nk.extension_available() is False


# --- fused module contract -------------------------------------------------


def _tiny_decoded(out_f: int = 256, in_f: int = 256, rank: int = 32,
                  per_channel: bool = False) -> DecodedLinear:
    gen = torch.Generator().manual_seed(3)
    codes = torch.randint(0, 16, (out_f, in_f), dtype=torch.uint8,
                          generator=gen)
    scales = (torch.rand(out_f, in_f // BLOCK, generator=gen) * 4 + 0.05).to(
        torch.float8_e4m3fn)
    second = (torch.rand(out_f if per_channel else 1, generator=gen) + 0.1)
    return DecodedLinear(
        out_features=out_f, in_features=in_f, codes=codes, scales=scales,
        second=second, second_kind="per_channel" if per_channel else
        "per_tensor", rank=rank,
        proj_down=torch.randn(in_f, rank, generator=gen).to(torch.bfloat16),
        proj_up=torch.randn(out_f, rank, generator=gen).to(torch.bfloat16),
        smooth_factor=(torch.rand(in_f, generator=gen) + 0.5).to(
            torch.bfloat16),
        bias=torch.randn(out_f, generator=gen).to(torch.bfloat16),
    )


def test_build_fused_linear_resident_swizzle() -> None:
    if svdq_fused.fused_ops() is None:
        pytest.skip("triton unavailable")
    from gen_worker.models.nvfp4_quant import to_blocked_scales

    dec = _tiny_decoded()
    mod = svdq_fused.build_svdq_fused_linear(dec)
    assert mod.weight.dtype == torch.uint8
    assert tuple(mod.weight.shape) == (dec.out_features,
                                       dec.in_features // 2)
    assert torch.equal(mod.weight, pack_e2m1(dec.codes))
    assert torch.equal(mod.weight_scale.view(torch.uint8),
                       to_blocked_scales(dec.scales).view(torch.uint8))
    assert mod.second.dtype == torch.float32
    assert mod.proj_down.shape == (dec.in_features, dec.rank)
    assert mod.proj_up.shape == (dec.out_features, dec.rank)


def test_fused_shape_contract_refuses_typed() -> None:
    assert svdq_fused.fused_shape_supported(256, 256, 32)
    assert not svdq_fused.fused_shape_supported(256, 200, 32)  # K % 128
    assert not svdq_fused.fused_shape_supported(250, 256, 32)  # N % 16
    assert not svdq_fused.fused_shape_supported(256, 256, 0)   # no low-rank
    assert not svdq_fused.fused_shape_supported(256, 256, 5)   # rank % 16
    if svdq_fused.fused_ops() is None:
        pytest.skip("triton unavailable")
    cls = svdq_fused.svdq_fused_linear_class()
    with pytest.raises(svdq_fused.SvdqFusedError):
        cls(200, 256, rank=32, bias=False, compute_dtype=torch.bfloat16,
            per_channel_scale=False, smooth=True)


# --- swap seam -------------------------------------------------------------


class _Model(torch.nn.Module):
    def __init__(self, out_f: int, in_f: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(in_f, out_f, bias=True)


def test_swap_picks_fused_execution_lane_when_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if svdq_fused.fused_ops() is None:
        pytest.skip("triton unavailable")
    dec = _tiny_decoded()
    model = _Model(dec.out_features, dec.in_features)
    monkeypatch.setattr(nk, "svdq_linear_execution_lane", lambda: "fused")
    monkeypatch.setattr(native, "svdq_linear_execution_lane", lambda: "fused")
    monkeypatch.setattr(nk, "svdq_modulation_execution_lane", lambda: "packed")
    monkeypatch.setattr(native, "svdq_modulation_execution_lane", lambda: "packed")
    counts = native.swap_svdq_linears(model, {"proj": dec}, mode="blockwise")
    assert counts == {"blockwise": 0, "dense": 0, "fused": 1, "prefixes": 1,
                      "linears": 1}
    assert getattr(model.proj, "_cozy_svdq_fused", False)


def test_swap_baseline_when_execution_lane_off(monkeypatch: pytest.MonkeyPatch) -> None:
    dec = _tiny_decoded()
    model = _Model(dec.out_features, dec.in_features)
    monkeypatch.setattr(nk, "svdq_linear_execution_lane", lambda: "baseline")
    monkeypatch.setattr(native, "svdq_linear_execution_lane", lambda: "baseline")
    monkeypatch.setattr(nk, "svdq_modulation_execution_lane", lambda: "dense")
    monkeypatch.setattr(native, "svdq_modulation_execution_lane", lambda: "dense")
    counts = native.swap_svdq_linears(model, {"proj": dec}, mode="blockwise")
    assert counts["fused"] == 0
    assert counts["blockwise"] == 1
    assert not getattr(model.proj, "_cozy_svdq_fused", False)


def test_swap_unsupported_shape_degrades_to_blockwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Armed lane, but a rank-less unit: falls to the baseline SvdqLinear —
    degrade, not refuse."""
    dec = _tiny_decoded()
    dec = DecodedLinear(
        out_features=dec.out_features, in_features=dec.in_features,
        codes=dec.codes, scales=dec.scales, second=dec.second,
        second_kind=dec.second_kind, rank=0, proj_down=None, proj_up=None,
        smooth_factor=dec.smooth_factor, bias=dec.bias)
    model = _Model(dec.out_features, dec.in_features)
    monkeypatch.setattr(nk, "svdq_linear_execution_lane", lambda: "fused")
    monkeypatch.setattr(native, "svdq_linear_execution_lane", lambda: "fused")
    monkeypatch.setattr(nk, "svdq_modulation_execution_lane", lambda: "packed")
    monkeypatch.setattr(native, "svdq_modulation_execution_lane", lambda: "packed")
    counts = native.swap_svdq_linears(model, {"proj": dec}, mode="blockwise")
    assert counts["fused"] == 0
    assert counts["blockwise"] == 1
