"""pgw#1198: virtuality is a question about STORAGE, never about TYPE.

THE BOUNDARY THIS FILE FIXES IN PLACE
-------------------------------------
§4.33 rests on "the compile is weight-free", and pod ``729431an6ugbvq``
(H100-80, wan-2.2, 0.113.0) refused it:

    structure-only build of component 'transformer,transformer_2' (WanPipeline)
    is not possible: 2 of 2 declared compile target(s) hold REAL parameters
    totalling 56_203_673_600 bytes

The forge had NOT failed. Both experts were built from code + config, stamped,
injected through ``components=`` and carried by the pipeline. What ran next was
the ENDPOINT's own ``setup()`` — which ``run_setup`` calls after composition —
and ``wan_2_2.main._quantize_fp8`` swapped every in-scope ``nn.Linear.weight``
for a torchao ``Float8Tensor``: a traceable wrapper subclass whose inner
``qdata``/``scale`` are the original FAKE tensors and whose OUTER dtype stays
bf16. Every virtuality test in the tree asked ``isinstance(t, FakeTensor)``,
which that object is not, and then priced ``numel * element_size`` at the outer
dtype — the ENTIRE bf16 checkpoint, for storage that did not exist.

So the boundary of the structure-only forge is NOT a list of pipeline classes,
and a survey of families would have gone stale the day one of them added a
quantizer. It is two properties, and this file tests the second one because the
first was already fenced:

1. the declared compile target must be buildable from code + config
   (``load_config`` + ``from_config``, named in ``model_index.json``) — refused
   by name in ``StructureOnlyUnsupported``, covered by
   ``test_structure_only_pgw1080``;
2. **whatever the endpoint's own ``setup()``/``warmup()`` then does to that
   module must remain visible AS virtual.** A quantizer that re-wraps a fake
   parameter in a tensor subclass is the shape the fleet actually ships
   (``wan-2.2``, ``minimax-h3``), and nothing fenced it.

Torchao is not in this image, so the subclass here is written to torch's own
wrapper-subclass contract — ``__tensor_flatten__`` / ``__tensor_unflatten__``,
outer dtype bf16 over an fp8 payload and an fp32 scale — which is precisely the
contract ``Float8Tensor`` implements and precisely what the fix consults. The
test drives the REAL seams: ``structure_only.build_component`` on the micro
family's real tree, then the real fences.

NOTHING HERE WEAKENS A FENCE
----------------------------
``StructureNotHonored`` behaved exactly as designed on that pod — it refused
rather than mint garbage. The rows below pin both directions: a subclass over
FAKE data is weight-free, and a subclass over REAL data (or over a MIX) is a
breach, still, by the same walk.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

import torch.nn as nn  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"
if str(MICRO_SRC) not in sys.path:
    sys.path.insert(0, str(MICRO_SRC))

from gen_worker import meta_instantiation as mi  # noqa: E402
from gen_worker.models import structure_only as so  # noqa: E402
from gen_worker.models.memory import device_mismatches  # noqa: E402


# ---------------------------------------------------------------------------
# The shape the fleet actually produces: a quantizer's wrapper subclass
# ---------------------------------------------------------------------------


class Quantized(torch.Tensor):
    """torchao ``Float8Tensor``'s contract, minus torchao.

    Outer dtype is the HIGH-PRECISION one (that is what makes the outer
    ``numel * element_size`` overstate by 2x on an fp8 payload); the storage
    lives in the inner tensors, which are whatever the weight was made of.
    """

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
        # `nn.Parameter(...)` detaches its data, so the subclass has to survive
        # that much — the same op torchao's own table implements first. Nothing
        # else is claimed: an op this test never exercises must say so rather
        # than silently produce a plain tensor.
        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            held = args[0]
            return cls(func(held.qdata), func(held.scale), held.dtype)
        raise NotImplementedError(func)  # pragma: no cover


def quantize_like_setup(module: Any) -> int:
    """What ``wan-2.2``'s ``setup()`` does to the module the forge just built.

    Runs inside the module's OWN fake mode, exactly as torchao's ``quantize_``
    does when it is handed a structure-only expert: the inner tensors it
    derives from a fake weight are fake, and it hands back a subclass.
    """
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
    """The same subclass over REAL storage — the case that must still breach.

    ``scale_real=False`` / ``qdata_real=False`` build the MIXED tensor: part
    fake, part real. Every inner tensor has to be virtual for the whole to be,
    so a mix is real.
    """
    def _make(size: Tuple[int, ...], dtype: Any, real: bool) -> Any:
        if real:
            return torch.empty(size, dtype=dtype)
        with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
            return torch.empty(size, dtype=dtype)

    qdata = _make(shape, torch.float8_e4m3fn, qdata_real)
    scale = _make((shape[0], 1), torch.float32, scale_real)
    return Quantized(qdata, scale, torch.bfloat16)


class _Composed:
    """A diffusers-shaped composition: ``.components`` plus plain attributes,
    the same fixture shape ``test_weight_free_premise_pgw1173`` uses."""

    def __init__(self, **parts: Any) -> None:
        self._parts = dict(parts)
        for name, part in parts.items():
            setattr(self, name, part)

    @property
    def components(self) -> Dict[str, Any]:
        return dict(self._parts)


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from micro_diffusion.weights import SEED, materialize

    root = tmp_path_factory.mktemp("micro-tree")
    return materialize(root, seed=SEED)


@pytest.fixture()
def quantized_target(tree: Path) -> Any:
    module, _facts = so.build_component(tree, "transformer", device="cpu")
    assert quantize_like_setup(module) > 0, (
        "the micro denoiser must carry Linears for this to test anything")
    return module


# ---------------------------------------------------------------------------
# 1. The primitive: virtuality answers about STORAGE
# ---------------------------------------------------------------------------


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
    """All-of, not any-of. A quantizer that leaves a real scale beside a fake
    payload has allocated, and the fence must charge for it."""
    assert not mi.is_virtual(
        real_quantized((64, 32), qdata_real=qdata_real, scale_real=scale_real))


def test_a_plain_real_tensor_is_still_NOT_virtual() -> None:
    assert not mi.is_virtual(torch.empty((8, 8), dtype=torch.bfloat16))


# ---------------------------------------------------------------------------
# 2. The fence pod 729431an6ugbvq tripped
# ---------------------------------------------------------------------------


def test_the_weight_free_fence_passes_a_setup_QUANTIZED_structure(
    quantized_target: Any,
) -> None:
    """RED before pgw#1198: this is the wan-2.2 refusal, reproduced.

    Without the fix every quantized Linear is priced at its OUTER bf16 dtype,
    so the breach total is the whole checkpoint of a module holding nothing.
    """
    pipe = _Composed(transformer=quantized_target)

    assert so.weight_free_breaches(pipe, ("transformer",)) == ()
    so.assert_weight_free(pipe, ("transformer",), what="the pgw#1198 boundary")


def test_the_fence_still_FIRES_when_the_quantizer_left_real_weights(
    quantized_target: Any,
) -> None:
    """The other direction, in the same walk. A fence that cannot fire is
    worse than no fence, and widening what counts as virtual must not have
    bought that."""
    quantized_target.proj_out.weight = nn.Parameter(
        real_quantized(tuple(quantized_target.proj_out.weight.shape)),
        requires_grad=False)
    pipe = _Composed(transformer=quantized_target)

    breaches = so.weight_free_breaches(pipe, ("transformer",))
    assert len(breaches) == 1
    assert breaches[0].reason == "real_parameters"
    assert breaches[0].real_param_bytes > 0
    with pytest.raises(so.StructureNotHonored):
        so.assert_weight_free(pipe, ("transformer",))


def test_the_placement_walk_does_not_read_a_quantized_structure_as_misplaced() -> None:
    """``device_mismatches`` had the same ``isinstance`` and the same blindness
    (pgw#1124 defect 2, arriving by the other door): a structure that allocates
    nothing cannot be moved, and counting it makes a CPU rollback unsatisfiable.

    The parameter must CLAIM THE CARD for this to test anything — that is the
    whole shape on the pod, and a fake tensor can claim one on a cardless box.
    """
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
    """The one virtual thing that walk must keep reporting: outside a
    structure-only component a meta tensor is an unmaterialized load, and
    ``meta_tensors`` reads it out of here."""
    module = nn.Linear(4, 4, device="meta")
    found = device_mismatches(_Composed(transformer=module), "cpu")
    assert [name for _c, name, _d in found] == ["weight", "bias"]


# ---------------------------------------------------------------------------
# 3. Re-virtualizing must preserve the QUANTIZED topology
# ---------------------------------------------------------------------------


def test_revirtualizing_keeps_the_quantized_topology(
    quantized_target: Any,
) -> None:
    """`virtualize` rebuilt every parameter as a plain tensor of its OUTER
    dtype, which turns a quantized weight into a bf16 one — so the export would
    trace bf16 Linears for a pod that serves fp8: a compiled graph for a graph the pod
    never executes, which is exactly what `_refuse_artifact_lanes` exists to
    prevent, arriving by the other door.

    (The rows that used to sit here drove `materialize_random` — the pgw#984
    warm proof's real-value materialisation. pgw#1199 deleted it: the proof
    runs on the RESIDENT parent now, so nothing in this module can put values
    on a structure. `test_structure_only_pgw1080` asserts that absence.)
    """
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
