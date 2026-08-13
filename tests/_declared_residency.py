"""Declare a multi-gigabyte CUDA-resident component on a box with no card.

The residency arithmetic is about GIGABYTES that are already on the card, and it
has to be exercised on a cardless CI runner. A FAKE tensor cannot stand in:
``memory._sum_tensor_bytes`` exempts fake tensors, correctly.

WHAT IS DECLARED, AND WHAT IS REAL. Two facts cannot be allocated on a cardless
box, and exactly those two are declared:

* **size**, through the tensor's SHAPE — a one-element storage ``expand``\\ ed
  to the weight count. Already this test's technique for the host-side
  component; it is now used for both sides.
* **device**, through a ``torch.Tensor`` SUBCLASS whose ``device`` reads
  ``cuda``. A cardless box can allocate no CUDA storage at all, so the device a
  15.5 GiB resident parameter reports is the one fact that has to be stated
  rather than produced.

Everything else is REAL, and that is the point: a real ``torch.Tensor``, with a
real and DISTINCT ``data_ptr`` — so the storage dedupe behaves, the
element-size arithmetic runs for real, and the walk's fake-tensor exemption
correctly does NOT fire, exactly as it would not fire on a genuine CUDA
parameter. The production path is not stubbed and no assertion is special-cased:
``_sum_tensor_bytes`` sees what it would see on a card.

The virtual counterpart — a component that declares bytes it will never
occupy — is built by production's own ``structure_only.virtualize``
(see ``virtual_component``), never by hand, so the thing under test is the
object production composes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_GIB = 1024 ** 3


class DeclaredCudaTensor(torch.Tensor):
    """A REAL tensor (not a FakeTensor) that reports a CUDA device.

    Its storage is one element and its shape is the weight count, so it costs
    two bytes to state fifteen gigabytes. Only ``device`` is overridden — every
    other property the byte walk reads (``numel``, ``element_size``,
    ``data_ptr``, ``isinstance``) is the tensor's own.
    """

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        return torch.device("cuda", 0)


def declared_bytes(gb: float, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """A host tensor whose SHAPE declares ``gb`` gigabytes."""
    element = torch.empty(1, dtype=dtype)
    count = int(gb * _GIB / element.element_size())
    return element.expand(count)


def declared_cuda_bytes(
    gb: float, *, dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """A tensor declaring ``gb`` gigabytes RESIDENT on a CUDA device."""
    return declared_bytes(gb, dtype=dtype).as_subclass(DeclaredCudaTensor)


def resident_component(gb: float) -> nn.Module:
    """A component whose weights are already on the card."""
    module = nn.Module()
    module._parameters["weight"] = declared_cuda_bytes(gb)  # type: ignore[assignment]
    return module


def host_component(gb: float) -> nn.Module:
    """A component still on the host, waiting to be placed."""
    module = nn.Module()
    module._parameters["weight"] = declared_bytes(gb)  # type: ignore[assignment]
    return module


def virtual_component(gb: float, *, device: str = "cuda") -> nn.Module:
    """A pgw#1080 STRUCTURE-ONLY component: fake parameters on the compute
    device, allocating nothing, built by production's own ``virtualize``.

    Construct-on-meta + ``virtualize`` is the sequence
    ``structure_only.build_component`` ends in, minus the config read — so the
    parameters here are fake in exactly the way a real boot-trace child's
    compile target is, rather than in a way a hand-written module imitates.
    (``accelerate.init_empty_weights`` is ``init_on_device(meta)``; for a
    DIRECT construction like this one the two are the same context, and this
    module then owes no dependency the SDK's own extras decide.)
    """
    from gen_worker.models import structure_only

    count = int(gb * _GIB / 2)
    with torch.device("meta"):
        module = nn.Linear(1, count, bias=False)
    structure_only.virtualize(module, device=device, dtype=torch.bfloat16)
    return module
