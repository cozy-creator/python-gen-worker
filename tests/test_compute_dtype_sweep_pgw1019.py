"""Every quantized leaf records its `compute_dtype`, and the consumer refuses to default one."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from gen_worker.models import adapter_fidelity as af  # noqa: E402
from gen_worker.models.fp8_storage import (  # noqa: E402
    restructure_fp8_storage,
)
from gen_worker.models.svdq_awq_packed import awq_packed_linear_class  # noqa: E402
from gen_worker.models.svdq_fused import svdq_fused_linear_class  # noqa: E402
from gen_worker.models.svdq_native import svdq_linear_class  # noqa: E402
from gen_worker.models.w4a4 import w4a4_linear_class  # noqa: E402
from gen_worker.models.w8a8 import fp8_scaled_linear_class  # noqa: E402
from gen_worker.models.w8a8_lora import alloc_branch_buffers  # noqa: E402

LANE = torch.float32

LEAVES = {
    "_Fp8ScaledLinear": (fp8_scaled_linear_class, dict(
        in_features=128, out_features=128, bias=False, compute_dtype=LANE,
        static_input_scale=False, gemm_mode="pertensor")),
    "_W4A4Linear": (w4a4_linear_class, dict(
        in_features=128, out_features=128, bias=False, compute_dtype=LANE,
        static_input_scale=False, pre_quant_scale=False)),
    "_SvdqLinear": (svdq_linear_class, dict(
        in_features=128, out_features=128, rank=0, bias=False,
        compute_dtype=LANE, per_channel_scale=False, smooth=False)),
    "_SvdqFusedLinear": (svdq_fused_linear_class, dict(
        in_features=128, out_features=128, rank=16, bias=False,
        compute_dtype=LANE, per_channel_scale=False, smooth=False)),
    "_AwqPackedLinear": (awq_packed_linear_class, dict(
        in_features=128, out_features=128, bias=False, compute_dtype=LANE)),
}


def test_a_leaf_that_states_no_compute_dtype_is_refused_by_name() -> None:
    cls, kwargs = LEAVES["_Fp8ScaledLinear"]
    mod = cls()(**kwargs)
    del mod.compute_dtype

    with pytest.raises(af.UnknownComputeDtypeError) as excinfo:
        af.branch_compute_dtype(mod)

    msg = str(excinfo.value)
    assert "_Fp8ScaledLinear" in msg
    assert "float8_e4m3fn" in msg
    assert "self.compute_dtype = compute_dtype" in msg
    assert "pgw#1015" in msg


def test_the_allocator_refuses_too_because_it_shares_the_definition() -> None:
    cls, kwargs = LEAVES["_Fp8ScaledLinear"]
    mod = cls()(**kwargs)
    del mod.compute_dtype
    with pytest.raises(af.UnknownComputeDtypeError):
        alloc_branch_buffers(mod, 16)
    with pytest.raises(af.UnknownComputeDtypeError):
        af.grid_of_module(mod, path=af.PATH_BRANCH)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_every_branch_capable_kind_answers_without_the_fallback(dtype) -> None:
    """The closed universe `w8a8_lora.branch_modules` admits, on every compute dtype — this is the evidence the refusal is safe."""
    plain = nn.Linear(8, 8, dtype=dtype)
    assert af.branch_compute_dtype(plain) is dtype

    conv = nn.Conv2d(4, 4, 3, dtype=dtype)
    assert af.branch_compute_dtype(conv) is dtype

    leaf = nn.Linear(8, 8, dtype=dtype)
    holder = nn.Module()
    holder.inner = leaf
    restructure_fp8_storage(holder, storage_dtype=torch.float8_e4m3fn,
                            compute_dtype=dtype)
    assert leaf.weight.dtype is torch.float8_e4m3fn
    assert leaf.bias.dtype is torch.float8_e4m3fn
    assert af.branch_compute_dtype(leaf) is dtype

    scaled = fp8_scaled_linear_class()(
        128, 128, bias=False, compute_dtype=dtype,
        static_input_scale=False, gemm_mode="pertensor")
    assert af.branch_compute_dtype(scaled) is dtype


def test_a_live_branch_buffer_still_outranks_the_rule() -> None:
    """`branch_grid_dtype` reads the destination first; only the unarmed case falls through to the (now refusing) rule."""
    lin = nn.Linear(8, 8, dtype=torch.bfloat16)
    alloc_branch_buffers(lin, 16)
    lin.lora_a = lin.lora_a.to(torch.float8_e4m3fn)
    lin.lora_b = lin.lora_b.to(torch.float8_e4m3fn)
    assert af.branch_grid_dtype(lin) is torch.float8_e4m3fn


@pytest.mark.parametrize("name", sorted(LEAVES))
def test_every_quantized_leaf_records_its_compute_dtype(name: str) -> None:
    cls, kwargs = LEAVES[name]
    mod = cls()(**kwargs)
    assert type(mod).__name__ == name
    assert mod.bias is None, "the census shape is bias-free on purpose"
    assert mod.compute_dtype is LANE
    assert af.branch_compute_dtype(mod) is LANE


def test_the_census_covers_every_leaf_accessor_in_models() -> None:
    """Discovery, so a NEW quantized leaf cannot be added without a row."""
    import gen_worker.models as models_pkg

    found: dict[str, str] = {}
    for info in pkgutil.iter_modules(models_pkg.__path__):
        mod = importlib.import_module(f"{models_pkg.__name__}.{info.name}")
        for fname, fn in vars(mod).items():
            if not (fname.endswith("_class") and inspect.isfunction(fn)):
                continue
            if inspect.signature(fn).parameters:
                continue
            try:
                cls = fn()
            except Exception:
                continue
            if not (inspect.isclass(cls) and issubclass(cls, nn.Module)):
                continue
            if "compute_dtype" in inspect.signature(cls.__init__).parameters:
                found[cls.__name__] = f"{info.name}.{fname}"

    assert set(found) == set(LEAVES), (
        f"census drift: {sorted(set(found) - set(LEAVES))} accept "
        f"compute_dtype but are not in LEAVES ({found}); "
        f"{sorted(set(LEAVES) - set(found))} are listed but no longer exist"
    )
