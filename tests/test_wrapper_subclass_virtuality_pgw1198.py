"""Virtuality is a question about STORAGE, never about TYPE."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

import torch.nn as nn  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))

from gen_worker import meta_instantiation as mi  # noqa: E402
from gen_worker.models import structure_only as so  # noqa: E402
from gen_worker.models.memory import device_mismatches  # noqa: E402


class Quantized(torch.Tensor):
    """torchao ``Float8Tensor``'s contract, minus torchao."""

    @staticmethod
    def __new__(cls, qdata: Any, scale: Any, dtype: Any) -> "Quantized":
        return torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, qdata.shape, dtype=dtype, device=qdata.device,
            requires_grad=False)

    def __init__(self, qdata: Any, scale: Any, dtype: Any) -> None:
        self.qdata = qdata
        self.scale = scale

    def __tensor_flatten__(self) -> Tuple[list, Dict[str, Any]]:
        return ["qdata", "scale"], {"dtype": self.dtype}

    @classmethod
    def __tensor_unflatten__(
        cls, inner: Dict[str, Any], meta: Dict[str, Any],
        outer_size: Any, outer_stride: Any,
    ) -> "Quantized":
        return cls(inner["qdata"], inner["scale"], meta["dtype"])

    @classmethod
    def __torch_dispatch__(cls, func: Any, types: Any, args: Any = (),
                           kwargs: Any = None) -> Any:
        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            held = args[0]
            return cls(func(held.qdata), func(held.scale), held.dtype)
        raise NotImplementedError(func)  # pragma: no cover


def quantize_like_setup(module: Any) -> int:
    """What ``wan-2.2``'s ``setup()`` does to the module the forge just built."""
    mode = so.fake_mode_of(module)
    swapped = 0
    for sub in module.modules():
        if not isinstance(sub, nn.Linear):
            continue
        weight = sub.weight
        with mode, torch.device(str(weight.device)):
            qdata = torch.empty(tuple(weight.shape),
                                dtype=torch.float8_e4m3fn)
            scale = torch.empty((weight.shape[0], 1), dtype=torch.float32)
        sub._parameters["weight"] = nn.Parameter(
            Quantized(qdata, scale, weight.dtype), requires_grad=False)
        swapped += 1
    return swapped


def real_quantized(shape: Tuple[int, ...], *, scale_real: bool = True,
                   qdata_real: bool = True) -> Any:
    """The same subclass over REAL storage — the case that must still breach."""
    def _make(size: Tuple[int, ...], dtype: Any, real: bool) -> Any:
        if real:
            return torch.empty(size, dtype=dtype)
        with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
            return torch.empty(size, dtype=dtype)

    qdata = _make(shape, torch.float8_e4m3fn, qdata_real)
    scale = _make((shape[0], 1), torch.float32, scale_real)
    return Quantized(qdata, scale, torch.bfloat16)


class _Composed:

    def __init__(self, **parts: Any) -> None:
        self._parts = dict(parts)
        for name, part in parts.items():
            setattr(self, name, part)

    @property
    def components(self) -> Dict[str, Any]:
        return dict(self._parts)


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A config-only tree."""
    from harness.structure_tree import build_config_only_tree

    return build_config_only_tree(tmp_path_factory.mktemp("structure-tree"))


@pytest.fixture()
def quantized_target(tree: Path) -> Any:
    module, _facts = so.build_component(tree, "transformer", device="cpu")
    assert quantize_like_setup(module) > 0, (
        "the denoiser must carry Linears for this to test anything")
    return module


def test_a_subclass_over_fake_data_is_VIRTUAL() -> None:
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    with FakeTensorMode():
        qdata = torch.empty((64, 32), dtype=torch.float8_e4m3fn)
        scale = torch.empty((64, 1), dtype=torch.float32)
    wrapped = Quantized(qdata, scale, torch.bfloat16)

    assert not isinstance(wrapped, FakeTensor), (
        "the premise of this whole issue: the object is NOT a FakeTensor")
    assert wrapped.dtype is torch.bfloat16, "the outer view is high precision"
    assert mi.is_virtual(wrapped), "it allocated nothing and must say so"


def test_a_subclass_over_REAL_data_is_NOT_virtual() -> None:
    assert not mi.is_virtual(real_quantized((64, 32)))


@pytest.mark.parametrize("qdata_real,scale_real", [(True, False), (False, True)])
def test_a_MIXED_subclass_is_NOT_virtual(qdata_real: bool,
                                         scale_real: bool) -> None:
    """All-of, not any-of."""
    assert not mi.is_virtual(
        real_quantized((64, 32), qdata_real=qdata_real, scale_real=scale_real))


def test_a_plain_real_tensor_is_still_NOT_virtual() -> None:
    assert not mi.is_virtual(torch.empty((8, 8), dtype=torch.bfloat16))


def test_the_weight_free_fence_passes_a_setup_QUANTIZED_structure(
    quantized_target: Any,
) -> None:
    pipe = _Composed(transformer=quantized_target)

    assert so.weight_free_breaches(pipe, ("transformer",)) == ()
    so.assert_weight_free(pipe, ("transformer",), what="the pgw#1198 boundary")


def test_the_fence_still_FIRES_when_the_quantizer_left_real_weights(
    quantized_target: Any,
) -> None:
    """The other direction, in the same walk."""
    victim = next(
        m for _n, m in quantized_target.named_modules()
        if isinstance(m, nn.Linear)
    )
    victim.weight = nn.Parameter(
        real_quantized(tuple(victim.weight.shape)), requires_grad=False)
    pipe = _Composed(transformer=quantized_target)

    breaches = so.weight_free_breaches(pipe, ("transformer",))
    assert len(breaches) == 1
    assert breaches[0].reason == "real_parameters"
    assert breaches[0].real_param_bytes > 0
    with pytest.raises(so.StructureNotHonored):
        so.assert_weight_free(pipe, ("transformer",))


def test_the_placement_walk_does_not_read_a_quantized_structure_as_misplaced() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(), torch.device("cuda"):
        qdata = torch.empty((32, 16), dtype=torch.float8_e4m3fn)
        scale = torch.empty((32, 1), dtype=torch.float32)
    module = nn.Module()
    module._parameters["weight"] = nn.Parameter(  # type: ignore[assignment]
        Quantized(qdata, scale, torch.bfloat16), requires_grad=False)

    assert str(module.weight.device).startswith("cuda")
    assert device_mismatches(_Composed(transformer=module), "cpu") == []


def test_a_META_tensor_is_still_reported_by_the_placement_walk() -> None:
    """The one virtual thing that walk must keep reporting: outside a structure-only component a meta tensor is an unmaterialized load, and ``meta_tensors`` reads it out of here."""
    module = nn.Linear(4, 4, device="meta")
    found = device_mismatches(_Composed(transformer=module), "cpu")
    assert [name for _c, name, _d in found] == ["weight", "bias"]


def test_revirtualizing_keeps_the_quantized_topology(
    quantized_target: Any,
) -> None:
    """`virtualize` rebuilt every parameter as a plain tensor of its OUTER dtype, which turns a quantized weight into a bf16 one — so the export would trace bf16 Linears for a pod that serves fp8: a compi..."""
    quantized = {name for name, p in quantized_target.named_parameters()
                 if isinstance(p.data, Quantized) or isinstance(p, Quantized)}
    assert quantized, "the fixture must have swapped something"

    so.virtualize(quantized_target, device="cpu")

    after = dict(quantized_target.named_parameters())
    for name in quantized:
        held = after[name]
        inner = held.data if not isinstance(held, Quantized) else held
        assert isinstance(inner, Quantized), (
            f"{name} lost its quantization to the re-virtualize")
        assert inner.qdata.dtype is torch.float8_e4m3fn
        assert mi.is_virtual(inner), "and it must still allocate nothing"
