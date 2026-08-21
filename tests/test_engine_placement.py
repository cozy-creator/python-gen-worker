"""I4 PLACEMENT: the sweep and the fence walk the MODULE (pgw#1644/pgw#1647).

The sweep that stood here read `pipeline.components` and fell back to the
pipeline object when that yielded no `nn.Module`. Both arms miss a MODULAR
pipeline — `MiniMaxH3StreamingPipeline` is a `ModularPipeline`/`ConfigMixin`
and is NEITHER an `nn.Module` NOR a carrier of `components` — so root discovery
returned `[]` and the sweep was a silent no-op for every component. That is how
three RoPE `inv_freq` buffers reached an H200 on the CPU under a model whose
every weight was on CUDA, and surfaced as `mat1 is on cpu` inside `diffusers`
eight milliseconds into a forward.

`census.place` takes a MODULE, so there is no root discovery to get wrong, and
`census.verify_placement` asserts the result instead of hoping for it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker.serving.streaming import census  # noqa: E402

from harness.nvml import nvml_is_healthy  # noqa: E402


class _Tiny(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        self.register_buffer("position_ids", torch.arange(2), persistent=False)


def test_a_meta_tensor_is_left_alone() -> None:
    """A tensor still on `meta` was never installed.

    Moving it would invent storage the container never delivered and would hide
    the refusal that names the real defect.
    """
    module = _Tiny()
    with torch.device("meta"):
        stranded = torch.arange(2)
    module._buffers["position_ids"] = stranded
    assert census.place(module, "cpu") == 0
    stayed = module._buffers["position_ids"]
    assert stayed is not None and stayed.device.type == "meta"


def test_a_tensor_left_on_meta_is_a_typed_I4_refusal() -> None:
    """...and it is reported, by name, as I4 rather than left to a matmul."""
    module = _Tiny()
    with torch.device("meta"):
        module._buffers["position_ids"] = torch.arange(2)
    with pytest.raises(census.CensusMismatch) as caught:
        census.verify_placement("tiny", module, "cpu")
    assert caught.value.invariant == census.I4_PLACEMENT
    assert caught.value.tensor == "position_ids"


def test_the_sweep_is_a_no_op_when_everything_is_already_placed() -> None:
    """It must not re-copy what the engine already streamed."""
    module = _Tiny()
    held = module._buffers["position_ids"]
    assert held is not None
    before = (module.weight.data_ptr(), held.data_ptr())
    assert census.place(module, "cpu") == 0
    after_buf = module._buffers["position_ids"]
    assert after_buf is not None
    assert (module.weight.data_ptr(), after_buf.data_ptr()) == before


def test_the_sweep_keeps_a_TIE_a_tie() -> None:
    """The move must not split an alias into a private copy.

    A rebind of `_parameters[leaf]` would place both names correctly and leave
    two tensors where the module had one — double the resident bytes, serving
    whichever half was written last. That is pgw#1626's failure wearing a
    placement fix's clothes, so the sweep moves `tensor.data` and the OBJECT
    survives.
    """
    module = nn.Module()
    source = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    module.register_parameter("source", source)
    module.register_parameter("alias", source)
    assert module.get_parameter("alias") is module.get_parameter("source")

    census.place(module, "cpu")
    assert module.get_parameter("alias") is module.get_parameter("source")


def test_off_target_is_INDEX_TOLERANT_so_it_cannot_refuse_a_healthy_load() -> None:
    """`torch.device("cuda") != torch.device("cuda", 0)` is True.

    The stream lands tensors on `cuda:0` while callers pass a bare `"cuda"`. A
    `!=` comparison would make this fence refuse every healthy load on the
    commonest spelling of the device it checks, and a fence that fires on
    correct input is worse than the defect it guards.
    """
    assert census._same_device(torch.device("cuda", 0), torch.device("cuda"))
    assert census._same_device(torch.device("cuda"), torch.device("cuda", 0))
    assert census._same_device(torch.device("cuda", 1), torch.device("cuda", 1))
    assert not census._same_device(torch.device("cuda", 1), torch.device("cuda", 0))
    assert not census._same_device(torch.device("cpu"), torch.device("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.skipif(
    not nvml_is_healthy(),
    reason="this host's NVML is version-mismatched; torch's P2P check raises inside pipeline.to(cuda)",
)
def test_a_buffer_the_container_never_carried_still_lands_on_the_device() -> None:
    module = _Tiny().to("cuda")
    module._buffers["position_ids"] = torch.arange(2)
    stranded = module._buffers["position_ids"]
    assert stranded is not None and stranded.device.type == "cpu"

    assert census.place(module, "cuda") == 1

    placed = module._buffers["position_ids"]
    assert placed is not None and placed.device.type == "cuda", (
        "a buffer absent from the container was left on the host; the first "
        "op reading it dies with 'index is on cpu, different from other "
        "tensors on cuda:0'"
    )
    census.verify_placement("tiny", module, "cuda")
