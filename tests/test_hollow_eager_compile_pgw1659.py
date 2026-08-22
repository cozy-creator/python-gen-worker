"""A hollow drive must not run compiled kernels, and must not blame constants.

pgw#1659. minimax-h3's `load()` arms `torch.compile` on its VAE decoder. Inside
a hollow session that compiled callable is handed FAKE tensors, so inductor's
generated wrapper reads a fake data pointer and launches a real kernel on it:

    Warning: Accessing the data pointer of FakeTensor ...
    minimax-h3: compiled vae.decoder failed (AcceleratorError: CUDA error: an
    illegal memory access was encountered); serving eager for the life of this
    process

The author caught it and carried on, so the drive finished against a dead CUDA
context and the derive died hundreds of frames later in torchcg's literal
digest:

    target 'transformer': observed call cannot state its identity: literal
    constant 'lifted_tensor_0' could not be digested: AcceleratorError: CUDA
    error: an illegal memory access was encountered

Nothing was wrong with `lifted_tensor_0`. Two properties keep that
misdiagnosis from happening again, and both live in the DERIVE rather than in
the vendored snapshot -- the derive owns the drive, and a vendored snapshot is
fixed upstream and re-vendored, never patched here.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from gen_worker.release.drive_hygiene import (  # noqa: E402
    accelerator_is_alive,
    dead_accelerator_sentence,
    eager_only_compile,
)


class Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_an_authors_torch_compile_is_identity_under_the_drive() -> None:
    """The whole prevention in one assertion, in every spelling authors use."""

    outside = torch.compile
    module = Decoder()
    eager = module.forward  # a bound method is a fresh object per attribute read

    with eager_only_compile():
        assert torch.compile is not outside
        # h3's spelling: a bound method, with kwargs.
        assert torch.compile(eager, dynamic=False) is eager
        # The module spelling.
        assert torch.compile(module) is module
        # `@torch.compile(mode=...)` -- the decorator FACTORY, no model given.
        assert torch.compile(mode="max-autotune")(eager) is eager
        # `Module.compile()` mutates in place -- and must not.
        module.compile()
        assert getattr(module.forward, "__self__", None) is module

    assert torch.compile is outside


def test_torch_compile_is_restored_even_when_the_body_raises() -> None:
    outside = torch.compile
    with pytest.raises(ZeroDivisionError):
        with eager_only_compile():
            assert torch.compile is not outside
            raise ZeroDivisionError
    assert torch.compile is outside


def test_a_compiled_module_still_gives_the_drive_its_structure() -> None:
    """Identity is not just a type check: the observation still has to work."""

    with eager_only_compile():
        module = Decoder()
        module.forward = torch.compile(  # type: ignore[method-assign]
            module.forward, dynamic=False
        )
        observed: list[tuple[int, ...]] = []
        module.register_forward_pre_hook(
            lambda _m, args: observed.append(tuple(args[0].shape))
        )
        out = module(torch.zeros(2, 4))

    assert observed == [(2, 4)]
    assert tuple(out.shape) == (2, 4)


def test_a_device_that_cannot_round_trip_a_byte_is_not_alive() -> None:
    """The RED arm of the probe, on any host: no backend is registered here.

    `torch.cuda.is_available()` keeps answering True on a context an illegal
    access killed, which is exactly why the probe is a real allocation and a
    real copy to host rather than an availability check.
    """

    assert accelerator_is_alive("cpu") is True
    assert accelerator_is_alive("privateuseone") is False


def test_the_dead_accelerator_refusal_names_the_drive_and_the_known_producer() -> None:
    said = dead_accelerator_sentence("cuda")

    assert "the drive left this process's cuda context DEAD" in said
    assert "The graphs are NOT the cause" in said
    # It must point at the known producer, not leave the reader guessing.
    assert "`torch.compile`d module handed FAKE tensors" in said
    assert "CUDA_LAUNCH_BLOCKING=1" in said


class _Session:
    """The two attributes the derive's probe reads, and nothing else."""

    def __init__(self, device: str) -> None:
        self.drive_device = device

    @property
    def drive_device_type(self) -> str:
        return self.drive_device.split(":", 1)[0]


def test_the_derive_refuses_a_dead_accelerator_instead_of_relaying_the_victim() -> None:
    """The misdiagnosis, reproduced and then refused correctly.

    `cause` is the sentence pgw#1659 actually shipped — a constant's name. The
    derive must NOT relay it as the diagnosis, and must keep it chained.
    """

    from gen_worker.release.derive import DeriveError, _refuse_a_dead_accelerator

    victim = RuntimeError(
        "literal constant 'lifted_tensor_0' could not be digested: "
        "AcceleratorError: CUDA error: an illegal memory access was encountered"
    )

    with pytest.raises(DeriveError) as refusal:
        _refuse_a_dead_accelerator("derive: class H3Model", _Session("privateuseone"), victim)

    said = str(refusal.value)
    assert "derive: class H3Model" in said
    assert "context DEAD" in said
    assert "lifted_tensor_0" not in said
    assert refusal.value.__cause__ is victim


def test_a_live_drive_device_is_not_refused() -> None:
    from gen_worker.release.derive import _refuse_a_dead_accelerator

    _refuse_a_dead_accelerator("derive", _Session("cpu"))
    _refuse_a_dead_accelerator("derive", None)
