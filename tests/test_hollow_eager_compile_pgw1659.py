"""A hollow drive must not run compiled kernels, and must not blame constants.

pgw#1659. minimax-h3's `load()` arms `torch.compile` on its VAE decoder. Inside
a hollow session that compiled callable is handed FAKE tensors, inductor's
generated wrapper reads a fake data pointer and launches a real kernel on it:

    Warning: Accessing the data pointer of FakeTensor ...
    minimax-h3: compiled vae.decoder failed (AcceleratorError: CUDA error: an
    illegal memory access was encountered); serving eager for the life of this
    process

The author caught it and carried on, so the drive finished against a dead CUDA
context and the derive died hundreds of frames later in the literal digest:

    target 'transformer': observed call cannot state its identity: literal
    constant 'lifted_tensor_0' could not be digested: AcceleratorError: CUDA
    error: an illegal memory access was encountered

Nothing was wrong with `lifted_tensor_0`. Three properties keep that
misdiagnosis from happening again: compile is IDENTITY in a session, a drive
that kills the accelerator is refused AT THE DRIVE, and a constant that cannot
be read refuses BY NAME with its device stated.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from gen_worker._vendor.torchcg.declaration import (  # noqa: E402
    DeclarationError,
    _literal_digest_for,
)
from gen_worker._vendor.torchcg.discovery import (  # noqa: E402
    DiscoveryError,
    _assert_the_drive_left_the_accelerator_alive,
)
from gen_worker._vendor.torchcg.hollow import (  # noqa: E402
    accelerator_is_alive,
    hollow_session,
)


class Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_an_authors_torch_compile_is_identity_inside_a_session() -> None:
    """The whole prevention in one assertion, in every spelling authors use."""

    outside = torch.compile
    module = Decoder()

    eager = module.forward  # a bound method is a fresh object per attribute read
    with hollow_session("cpu"):
        assert torch.compile is not outside
        # h3's spelling: a bound method, with kwargs.
        assert torch.compile(eager, dynamic=False) is eager
        # The module spelling.
        assert torch.compile(module) is module
        # `@torch.compile(mode=...)` -- the decorator FACTORY, no model given.
        assert torch.compile(mode="max-autotune")(eager) is eager
        # `Module.compile()` mutates in place and returns None either way.
        assert module.compile() is None
        assert type(module.forward.__self__) is Decoder

    assert torch.compile is outside


def test_the_session_restores_torch_compile_even_when_the_body_raises() -> None:
    outside = torch.compile
    with pytest.raises(ZeroDivisionError):
        with hollow_session("cpu"):
            assert torch.compile is not outside
            raise ZeroDivisionError
    assert torch.compile is outside


def test_a_compiled_module_in_a_hollow_drive_returns_the_eager_answer() -> None:
    """Identity is not just a type check: the drive still gets its structure.

    A no-op compile has to leave the drive able to observe the call, which is
    the only thing the drive is for.
    """

    with hollow_session("cpu"):
        module = Decoder()
        module.forward = torch.compile(module.forward, dynamic=False)
        observed: list[tuple[int, ...]] = []
        module.register_forward_pre_hook(
            lambda _m, args: observed.append(tuple(args[0].shape))
        )
        out = module(torch.zeros(2, 4))

    assert observed == [(2, 4)]
    assert tuple(out.shape) == (2, 4)


class _Session:
    """The two attributes the drive probe reads, and nothing else."""

    def __init__(self, device: str) -> None:
        self.drive_device = device

    @property
    def drive_device_type(self) -> str:
        return self.drive_device.split(":", 1)[0]


def test_a_device_that_cannot_round_trip_a_byte_is_not_alive() -> None:
    """The RED arm of the probe, on any host: no backend is registered here.

    `torch.cuda.is_available()` keeps answering True on a context an illegal
    access killed, which is exactly why the probe is a real allocation and a
    real copy to host rather than an availability check.
    """

    assert accelerator_is_alive("cpu") is True
    assert accelerator_is_alive("privateuseone") is False


def test_a_drive_that_killed_the_accelerator_is_refused_AT_THE_DRIVE() -> None:
    with pytest.raises(DiscoveryError) as refusal:
        _assert_the_drive_left_the_accelerator_alive(
            "minimax-h3.diffusers@1+plain.bf16@1", _Session("privateuseone")
        )

    said = str(refusal.value)
    assert "the drive left this process's privateuseone context DEAD" in said
    assert "The graphs are NOT the cause" in said
    # It must point at the known producer, not leave the reader guessing.
    assert "`torch.compile`d module handed FAKE tensors" in said
    assert "CUDA_LAUNCH_BLOCKING=1" in said


def test_a_live_drive_device_is_not_refused() -> None:
    _assert_the_drive_left_the_accelerator_alive("lane", _Session("cpu"))
    _assert_the_drive_left_the_accelerator_alive("lane", None)


class _Program:
    def __init__(self, constants: dict[str, Any]) -> None:
        self.constants = constants


def test_an_unreadable_literal_refuses_BY_NAME_and_states_its_device() -> None:
    """A meta constant has no storage, so this is undigestable on every host."""

    program = _Program({"lifted_tensor_0": torch.zeros(2, 2, device="meta")})

    with pytest.raises(DeclarationError) as refusal:
        _literal_digest_for(program, ["lifted_tensor_0"])

    said = str(refusal.value)
    assert "literal constant 'lifted_tensor_0' (on meta) could not be digested" in said
    # And it says the copy-to-host is where a fault SURFACES, not where it is.
    assert "a fault raised EARLIER in this process surfaces here" in said


def test_a_readable_cpu_literal_carries_no_accelerator_sentence() -> None:
    """The hint is for off-host constants only -- noise on a cpu one."""

    program = _Program({"table": torch.arange(4, dtype=torch.float32)})
    assert _literal_digest_for(program, ["table"])

    broken = _Program({"table": torch.zeros(2, 2).to_sparse()})
    with pytest.raises(DeclarationError) as refusal:
        _literal_digest_for(broken, ["table"])
    said = str(refusal.value)
    assert "literal constant 'table' (on cpu) could not be digested" in said
    assert "surfaces here" not in said
