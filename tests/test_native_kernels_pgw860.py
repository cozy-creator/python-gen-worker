"""pgw#860 — native-kernel dispatch: env gating, probe fallback, extension
probe, and the fused-lane swap seam. CPU-only; kernel numerics run on the
pgw#865 GPU harness."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gen_worker import kernel_path  # noqa: E402
from gen_worker.models import native_kernels as nk  # noqa: E402
from gen_worker.models import svdq_fused  # noqa: E402
from gen_worker.models import svdq_native as native  # noqa: E402
from gen_worker.models.nvfp4_quant import BLOCK, pack_e2m1  # noqa: E402
from gen_worker.models.svdq_layout import DecodedLinear  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_arming(monkeypatch: pytest.MonkeyPatch):
    nk.reset_native_kernels_arming()
    kernel_path.clear()
    yield
    nk.reset_native_kernels_arming()
    kernel_path.clear()


# --- env gating ------------------------------------------------------------


def test_unset_env_stays_baseline_dormant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(nk.NATIVE_ENV, raising=False)
    assert nk.svdq_linear_execution_lane() == "baseline"
    assert "env-gated" in nk.svdq_linear_execution_lane_reason()
    assert nk.svdq_modulation_execution_lane() == "dense"
    assert "env-gated" in nk.svdq_modulation_execution_lane_reason()


def test_kill_switch_forces_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(nk.NATIVE_ENV, "0")
    # Even an armed verdict + passing self-checks must not arm past the
    # kill-switch, on either axis.
    kernel_path.pin("fused+packed", "test")
    monkeypatch.setattr(nk, "_fused_linear_self_check", lambda: None)
    monkeypatch.setattr(nk, "_packed_modulation_self_check", lambda: None)
    assert nk.svdq_linear_execution_lane() == "baseline"
    assert "kill-switch" in nk.svdq_linear_execution_lane_reason()
    assert nk.svdq_modulation_execution_lane() == "dense"
    assert "kill-switch" in nk.svdq_modulation_execution_lane_reason()


def test_armed_verdict_with_passing_self_checks_arms_both_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(nk.NATIVE_ENV, "1")
    kernel_path.pin("fused+packed", "compiled_graph verdict")
    monkeypatch.setattr(nk, "_fused_linear_self_check", lambda: None)
    monkeypatch.setattr(nk, "_packed_modulation_self_check", lambda: None)
    assert nk.svdq_linear_execution_lane() == "fused"
    assert nk.svdq_modulation_execution_lane() == "packed"
    assert "compiled_graph verdict" in nk.svdq_linear_execution_lane_reason()
    assert "compiled_graph verdict" in nk.svdq_modulation_execution_lane_reason()


def test_the_axes_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """pgw#863: a card whose fused LINEAR loses must still get the packed
    modulation. Binding them to one switch cost sm_100 either the residency
    win or 19% of its step time, with no way to take both — so the verdict
    vocabulary has to be able to say `baseline+packed`, and B200's does."""
    monkeypatch.setenv(nk.NATIVE_ENV, "1")
    kernel_path.pin("baseline+packed", "compiled_graph verdict: B200 228 vs 350 ms")
    monkeypatch.setattr(nk, "_packed_modulation_self_check", lambda: None)
    assert nk.svdq_linear_execution_lane() == "baseline"
    assert nk.svdq_modulation_execution_lane() == "packed"


def test_a_baseline_axis_never_runs_its_self_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An axis a compiled graph puts on its degraded value is obeyed verbatim: no
    probe, no SM read, no compile."""
    def boom() -> str:
        raise AssertionError("the self-check must not run for a degraded axis")

    monkeypatch.setenv(nk.NATIVE_ENV, "1")
    kernel_path.pin("baseline+dense", "compiled_graph verdict: B200 measured 228 vs 350")
    monkeypatch.setattr(nk, "_fused_linear_self_check", boom)
    monkeypatch.setattr(nk, "_packed_modulation_self_check", boom)
    assert nk.svdq_linear_execution_lane() == "baseline"
    assert nk.svdq_modulation_execution_lane() == "dense"
    assert "228 vs 350" in nk.svdq_linear_execution_lane_reason()


def test_a_failing_self_check_degrades_only_its_own_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(nk.NATIVE_ENV, "1")
    kernel_path.pin("fused+packed", "compiled_graph verdict")
    monkeypatch.setattr(nk, "_fused_linear_self_check", lambda: "no fp4 here")
    monkeypatch.setattr(nk, "_packed_modulation_self_check", lambda: None)
    assert nk.svdq_linear_execution_lane() == "baseline"
    assert "no fp4 here" in nk.svdq_linear_execution_lane_reason()
    assert nk.svdq_modulation_execution_lane() == "packed"


def test_self_check_raise_degrades_not_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> str:
        raise RuntimeError("driver exploded")

    monkeypatch.setenv(nk.NATIVE_ENV, "1")
    kernel_path.pin("fused+packed", "compiled_graph verdict")
    monkeypatch.setattr(nk, "_fused_linear_self_check", boom)
    assert nk.svdq_linear_execution_lane() == "baseline"
    assert "driver exploded" in nk.svdq_linear_execution_lane_reason()


def test_no_verdict_is_the_declared_default_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#947 (d): nothing pinned a lane => the conservative default on BOTH
    axes, with a TYPED reason. Never a silent fall-through to a hand tuple."""
    monkeypatch.setenv(nk.NATIVE_ENV, "1")
    kernel_path.clear()
    assert nk.svdq_linear_execution_lane() == kernel_path.linear_of(
        kernel_path.DEFAULT_EXECUTION_LANE)
    assert nk.svdq_modulation_execution_lane() == kernel_path.modulation_of(
        kernel_path.DEFAULT_EXECUTION_LANE)
    assert kernel_path.REASON_ABSENT in nk.svdq_linear_execution_lane_reason()
    assert kernel_path.REASON_ABSENT in nk.svdq_modulation_execution_lane_reason()


def test_no_sm_allowlist_survives() -> None:
    """pgw#863 + pgw#947: BOTH hand-maintained tuples are DELETED, not
    renamed and not split in two. The only SM fact left in the mechanism is
    the fused linear's capability floor, and it lives in kernel_lane where
    the candidate enumeration is — the packed modulation has no SM term at
    all, because its kernel never needed Blackwell silicon."""
    assert not hasattr(nk, "FUSED_SMS")
    assert not hasattr(nk, "FUSED_LINEAR_SMS")
    assert not hasattr(nk, "PACKED_MODULATION_SMS")
    assert not hasattr(nk, "BLACKWELL_SMS")
    assert kernel_path.FUSED_MIN_SM == 100


def test_real_self_checks_on_this_host_degrade_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a box without the kernels the REAL self-checks must return a
    reason, never raise (on a Blackwell runner they instead arm — also
    fine)."""
    monkeypatch.setenv(nk.NATIVE_ENV, "1")
    kernel_path.pin("fused+packed", "compiled_graph verdict")
    assert nk.svdq_linear_execution_lane() in ("baseline", "fused")
    assert nk.svdq_linear_execution_lane_reason()
    assert nk.svdq_modulation_execution_lane() in ("dense", "packed")
    assert nk.svdq_modulation_execution_lane_reason()


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
