"""pgw#1128: a VIRTUAL tensor is not resident bytes, and is never data_ptr-ed.

``memory._sum_tensor_bytes`` walked every parameter and buffer as if it held
storage. A pgw#1080 structure-only component's parameters are FAKE on the
compute device by construction — they declare a shape and a device and allocate
nothing — and the walk booked them three ways wrong:

1. **as resident VRAM.** ``estimate_cuda_resident_gb`` is the "what occupies
   the card right now" question, and it answered with the full declared weight
   of a structure that occupies none of it. Downstream that is
   ``select_auto_mode``'s pgw#1025 net requirement falling to ZERO, so a
   structure-only pipeline reads as one whose weights the card has already paid
   for: a 40 GB tree "fits" a 24 GB card and takes a fully RESIDENT rung.
2. **collapsed by the storage dedupe.** Every FakeTensor answers ``data_ptr()``
   with ``0``, so a tree of them deduped to its FIRST tensor and the whole rest
   of the requirement vanished.
3. **on a call torch is removing.** *"Accessing the data pointer of FakeTensor
   is deprecated and will error"* — and this walk's callers wrap it in a bare
   ``except: return 0.0``, so the day it errors, every estimate in the placement
   ladder silently becomes zero.

The counterpart the fix must NOT break is the pgw#1025 measurement technique:
that test stands in for a 15.5 GiB CUDA-resident component on a cardless box,
and its stand-in WAS a fake tensor. ``tests/_declared_residency`` replaces it
with a real tensor that declares its size through shape and its device through
a subclass — so the arithmetic runs on a real tensor with a real storage, and
the fake-tensor exemption proved here correctly does not fire on it.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")

from gen_worker.models import memory  # noqa: E402

from _declared_residency import (  # noqa: E402
    host_component, resident_component, virtual_component,
)


class _Pipeline:
    """A diffusers-shaped composition: ``.components`` is the walk's own entry
    point."""

    def __init__(self, **parts: Any) -> None:
        self._parts: Dict[str, Any] = dict(parts)
        for name, part in parts.items():
            setattr(self, name, part)

    @property
    def components(self) -> Dict[str, Any]:
        return dict(self._parts)


# ---------------------------------------------------------------------------
# 1. a virtual structure occupies no card
# ---------------------------------------------------------------------------


def test_a_structure_only_component_is_not_resident_vram() -> None:
    """The boot-trace / mint child's composition. Its transformer's parameters
    are fake ON the compute device — that placement is part of the graph's
    identity (pgw#1080) and is precisely why the naive walk read them as 40 GB
    of occupied card."""
    pipe = _Pipeline(transformer=virtual_component(40.0, device="cuda"))
    assert memory.estimate_cuda_resident_gb(pipe) == 0.0


def test_the_requirement_still_counts_what_the_virtual_structure_declares() -> None:
    """The other half, and the reason this is not a blanket "ignore fakes": the
    shape and dtype a fake parameter declares are the bytes a real load — or
    ``materialize_random`` in the mint child — goes on to allocate."""
    pipe = _Pipeline(transformer=virtual_component(40.0, device="cuda"))
    assert memory.estimate_pipeline_size_gb(pipe) == pytest.approx(40.0)


def test_a_virtual_pipeline_does_not_read_as_already_paid_for() -> None:
    """The production consequence, through pgw#1025's own arithmetic.

    Pre-fix: requirement 40.0 AND resident 40.0, so the net fit requirement was
    0.0 and a 40 GB structure "fit" a 24 GB card with 6 GB free — ``vae_only``,
    a fully RESIDENT rung. Post-fix the net requirement is the whole 40 GB, it
    does not fit, and 6 GB free is the aggressive rung's own threshold.
    """
    pipe = _Pipeline(transformer=virtual_component(40.0, device="cuda"))
    mode = memory.select_auto_mode(
        pipeline=pipe, available_vram_gb=6.0, total_vram_gb=24.0)
    assert mode == "group_offload", (
        f"selected {mode!r}: a structure-only pipeline was booked as already "
        "resident, so its whole weight was subtracted from its own "
        "requirement and a 40 GB tree read as fitting a 24 GB card"
    )


# ---------------------------------------------------------------------------
# 2. the storage dedupe does not collapse a tree of virtual tensors
# ---------------------------------------------------------------------------


def test_two_virtual_components_are_not_deduped_into_one() -> None:
    """``data_ptr()`` is ``0`` for EVERY fake tensor, so the shared-storage
    dedupe treated a whole structure-only tree as one tensor. Two 8 GB
    components must weigh 16 GB, not 8."""
    pipe = _Pipeline(
        transformer=virtual_component(8.0, device="cuda"),
        text_encoder=virtual_component(8.0, device="cuda"),
    )
    assert memory.estimate_pipeline_size_gb(pipe) == pytest.approx(16.0)


def test_a_shared_real_storage_is_still_counted_once() -> None:
    """The dedupe the fake tensors were colliding with is real and stays: two
    components VIEWING one storage (the gw#479 shared-component shape) weigh it
    once."""
    shared = host_component(10.0)
    pipe = _Pipeline(text_encoder=shared, text_encoder_2=shared)
    assert memory.estimate_pipeline_size_gb(pipe) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 3. the deprecated call is never made
# ---------------------------------------------------------------------------


def test_a_fake_tensors_data_pointer_is_never_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch: *"Accessing the data pointer of FakeTensor is deprecated and will
    error"*. Both estimates walk the same function, so both are asserted.

    The call is RECORDED rather than sniffed through torch's own warning: that
    warning is emitted once per process, so a suite that had already tripped it
    elsewhere would leave this row unable to go red — a check that cannot fail.
    """
    from torch._subclasses.fake_tensor import FakeTensor

    reads: list[tuple[int, ...]] = []

    def _record(self: Any) -> int:
        reads.append(tuple(self.shape))
        return 0

    monkeypatch.setattr(FakeTensor, "data_ptr", _record)
    pipe = _Pipeline(
        transformer=virtual_component(8.0, device="cuda"),
        vae=virtual_component(1.0, device="cuda"),
    )
    memory.estimate_cuda_resident_gb(pipe)
    memory.estimate_pipeline_size_gb(pipe)
    assert reads == [], (
        f"the byte walk read {len(reads)} fake tensor data pointer(s): {reads}")


def test_the_walk_survives_a_data_pointer_that_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The future torch, today: with the deprecation landed as an exception, a
    walk that still called it would take the callers' bare ``except`` and
    return 0.0 for EVERY estimate — a placement ladder measuring nothing, with
    no failure anywhere to read."""
    from torch._subclasses.fake_tensor import FakeTensor

    def _refuse(self: Any) -> int:
        raise RuntimeError(
            "Accessing the data pointer of FakeTensor is not allowed")

    monkeypatch.setattr(FakeTensor, "data_ptr", _refuse)
    pipe = _Pipeline(
        transformer=virtual_component(8.0, device="cuda"),
        text_encoder=host_component(4.0),
    )
    assert memory.estimate_pipeline_size_gb(pipe) == pytest.approx(12.0)
    assert memory.estimate_cuda_resident_gb(pipe) == 0.0


# ---------------------------------------------------------------------------
# 4. the pgw#1025 technique still measures what it was measuring
# ---------------------------------------------------------------------------


def test_a_declared_resident_component_is_real_to_the_walk() -> None:
    """The stand-in that replaces the fake tensor: not virtual, so the pgw#1128
    exemption must NOT fire on it, and its declared 15.5 GB is resident."""
    from torch._subclasses.fake_tensor import FakeTensor

    weight = next(resident_component(15.5).parameters())
    assert isinstance(weight, torch.Tensor)
    assert not isinstance(weight, FakeTensor)
    assert weight.device.type == "cuda"
    assert weight.data_ptr() != 0
    pipe = _Pipeline(text_encoder=resident_component(15.5))
    assert memory.estimate_cuda_resident_gb(pipe) == pytest.approx(15.5)
    assert memory.estimate_pipeline_size_gb(pipe) == pytest.approx(15.5)


def test_two_declared_resident_components_are_not_deduped() -> None:
    """The reason the stand-in keeps a REAL storage: two of them are two
    distinct ``data_ptr``\\ s, so a shared-component pipeline can declare more
    than one resident component. The fake-tensor stand-in could not — every
    fake tensor shares the address ``0``."""
    pipe = _Pipeline(
        text_encoder=resident_component(8.0), vae=resident_component(2.0))
    assert memory.estimate_cuda_resident_gb(pipe) == pytest.approx(10.0)
