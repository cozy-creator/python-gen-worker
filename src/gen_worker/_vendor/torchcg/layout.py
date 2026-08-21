"""Layout morphisms -- the byte layout an artifact DECLARES for its inputs.

tcg#83, the torchcg half of ``research/layout-morphisms/0-DESIGN.md``. The
compiler decides a byte layout for every tensor it touches; today the platform
delivers row-major bytes and inductor re-derives its preferred layout on every
forward call (sdxl UNet: 40 Triton permute launches, 1.19 GiB/step of DRAM
traffic, measured in tcg#82). The fix is a CONTRACT: the artifact declares the
layout it was compiled against, the key carries that declaration, and the
loader delivers bytes in it.

**A morphism is DATA, not code by convention.** Every entry here is one of
torch's own memory formats, so the permutation is torch's specification and the
verifier is ``Tensor.is_contiguous(memory_format=...)``. Nothing in this module
copies a stride order or a permutation out of torch: orders are computed from a
meta-device probe through ``torch._inductor.ir.get_stride_order``, which is the
same function inductor uses to make the decision this catalog names.

**The catalog is CLOSED, and this is only its INDUCTOR class.** The design
gives layout morphisms two classes: the inductor class (small, known upfront,
the entries below) and the endpoint-declared / kernel-library class
(``bfl.nvfp4-preswizzled`` vs ``cozy.nvfp4-flat`` -- quant-rule identity, where
the real wins are). The second class is catalog DATA owned by tensorfs
([[tensorfs#155]]) and reaches torchcg as a vendored document; this module is
the first class and the loader for the second. A layout nobody ratified is
never invented here: it is a CANDIDATE, emitted for ratification, and the mint
falls back to the stored layout (:mod:`torchcg.compiler`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

from .lane import _HANDLE_RE


class LayoutError(ValueError):
    """A layout declaration or wish is not a ratified catalog morphism."""


@dataclass(frozen=True, slots=True)
class LayoutMorphism:
    """One named, versioned byte-layout map over a tensor's storage.

    ``memory_format`` is the name of the ``torch.memory_format`` this morphism
    IS -- the permutation lives in torch, and this entry only names it. ``rank``
    is the tensor rank the format is defined for (``None`` = every rank, which
    is true only of row-major). A morphism that does not apply to a tensor's
    rank leaves it row-major: a 1-D bias under a ``channels_last`` declaration
    is contiguous, not an error, because the permutation is the identity there.
    """

    handle: str
    memory_format: str
    rank: int | None

    def __post_init__(self) -> None:
        if _HANDLE_RE.fullmatch(self.handle) is None:
            raise LayoutError(
                f"a layout morphism handle is '<producer>.<format>@<version>'; "
                f"got {self.handle!r}"
            )

    def format(self) -> Any:
        import torch

        return getattr(torch, self.memory_format)

    def applies_to(self, rank: int) -> bool:
        return self.rank is None or self.rank == rank

    def matches(self, tensor: Any) -> bool:
        """Whether these bytes are ALREADY in this layout -- no copy needed.

        This is the predicate the whole design turns on: inductor's
        ``require_strides`` emits no copy when the strides already match
        (torch ``_inductor/ir.py``), so a tree delivered in the declared layout
        costs zero permute kernels. Read as a question about the tensor, never
        as an instruction to transform it.
        """

        import torch

        rank = tensor.dim()
        if not self.applies_to(rank):
            return bool(tensor.is_contiguous(memory_format=torch.contiguous_format))
        return bool(tensor.is_contiguous(memory_format=self.format()))

    def describe(self, tensor: Any) -> str:
        return (
            f"shape {tuple(tensor.shape)} strides {tuple(tensor.stride())} "
            f"(declared {self.handle})"
        )

    def stride_order(self, rank: int) -> tuple[int, ...] | None:
        """This morphism's stride order at one rank, computed FROM torch."""

        if not self.applies_to(rank):
            return None
        return _probe_stride_order(self.memory_format, rank)


#: Row-major. The identity morphism, the default for every tree and every
#: artifact that existed before this axis, and the PERMANENT fallback the
#: design forbids removing: a wish outside the catalog compiles against this.
CONTIGUOUS = "torch.contiguous@1"

#: The closed inductor class. `transposed` and `stride-padding` from the design
#: sketch are deliberately absent: neither is a torch memory format, so neither
#: is machine-verifiable by the auto-ratification rule (apply O then O-inverse
#: and byte-compare). They arrive as catalog DATA or not at all.
CATALOG: dict[str, LayoutMorphism] = {
    morphism.handle: morphism
    for morphism in (
        LayoutMorphism(CONTIGUOUS, "contiguous_format", None),
        LayoutMorphism("torch.channels-last@1", "channels_last", 4),
        LayoutMorphism("torch.channels-last-3d@1", "channels_last_3d", 5),
    )
}


def require_morphism(handle: object) -> LayoutMorphism:
    """Resolve one declared layout, or refuse naming the whole catalog.

    An unclassified layout value is a refusal, never a coercion to row-major:
    a declaration nobody can verify would let an artifact claim a layout its
    loader silently ignores, which is the defect this axis exists to remove.
    """

    if not isinstance(handle, str) or handle not in CATALOG:
        raise LayoutError(
            f"declared input layout {handle!r} is not a ratified layout morphism; "
            f"the catalog is {sorted(CATALOG)!r}. Machines derive along ratified "
            f"morphisms and never invent one: an unknown layout is a CANDIDATE "
            f"for ratification, and the mint falls back to {CONTIGUOUS!r}"
        )
    return CATALOG[handle]


def is_catalog_handle(value: object) -> bool:
    return isinstance(value, str) and value in CATALOG


@cache
def _probe_stride_order(memory_format: str, rank: int) -> tuple[int, ...]:
    """The stride order one memory format produces at one rank, per torch.

    Read rather than written: a meta-device probe with pairwise-distinct
    dimensions is materialized in the format, and its strides are converted by
    inductor's OWN ``get_stride_order``. That is the same conversion inductor
    performs when it decides what layout it wants, so the two cannot drift.
    """

    import torch
    from torch._inductor.ir import get_stride_order

    size = tuple(range(2, 2 + rank))
    probe = torch.empty(size, device="meta").to(
        memory_format=getattr(torch, memory_format)
    )
    return tuple(int(value) for value in get_stride_order(probe.stride()))


def morphism_for_stride_order(order: Sequence[int]) -> LayoutMorphism | None:
    """The catalog morphism one stride order names, or None (a CANDIDATE).

    ``None`` is the design's permanent fallback path, not a failure: the mint
    compiles against the stored layout and emits the order as an unratified
    candidate. Never invent a name for it.
    """

    wanted = tuple(int(value) for value in order)
    for morphism in CATALOG.values():
        if morphism.stride_order(len(wanted)) == wanted:
            return morphism
    return None


__all__ = [
    "CATALOG",
    "CONTIGUOUS",
    "LayoutError",
    "LayoutMorphism",
    "is_catalog_handle",
    "morphism_for_stride_order",
    "require_morphism",
]
