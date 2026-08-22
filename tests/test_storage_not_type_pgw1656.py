"""Virtuality is a question about STORAGE, never about TYPE (pgw#1661/#1662).

The three sites below were `isinstance(..., FakeTensor)` in the vendored torchcg
tree, where they could not be fixed in place. They land here with the absorb, so
they are fixed first-party — and each one is asked with a WRAPPER SUBCLASS OVER
FAKE, which is exactly the shape an isinstance check reads as real: torchao's
`Float8Tensor` today, and every quantized wrapper after it.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from gen_worker.meta_instantiation import is_virtual


class _Wrapper(torch.Tensor):
    """A wrapper subclass holding a fake tensor — NOT a FakeTensor itself."""

    _inner: Any

    @staticmethod
    def __new__(cls, inner: Any) -> "_Wrapper":
        shape = tuple(int(d) for d in inner.shape)
        made = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, shape, dtype=inner.dtype, device=inner.device, requires_grad=False
        )
        made._inner = inner
        return made  # type: ignore[no-any-return]

    def __tensor_flatten__(self) -> tuple[list[str], None]:
        return ["_inner"], None

    @staticmethod
    def __tensor_unflatten__(
        inner: Any, _ctx: Any, _sizes: Any = None, _strides: Any = None
    ) -> "_Wrapper":
        # Required for `is_traceable_wrapper_subclass`, which is what makes
        # `is_virtual` unwrap at all — torchao's Float8Tensor defines both.
        return _Wrapper(inner["_inner"])

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]
        cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        unwrapped = tuple(
            a._inner if isinstance(a, _Wrapper) else a for a in args
        )
        return func(*unwrapped, **(kwargs or {}))


@pytest.fixture
def wrapped_fake() -> Any:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        inner = torch.zeros(4, 4)
    wrapper = _Wrapper(inner)
    from torch._subclasses.fake_tensor import FakeTensor

    assert not isinstance(wrapper, FakeTensor), (
        "the fixture must NOT be a FakeTensor, or it proves nothing"
    )
    return wrapper


def test_the_wrapper_is_virtual_even_though_it_is_not_a_FakeTensor(
    wrapped_fake: Any,
) -> None:
    """The premise. `is_virtual` unwraps; `isinstance` does not."""

    assert is_virtual(wrapped_fake)


def test_is_hollow_reads_a_wrapped_parameter_as_HOLLOW(wrapped_fake: Any) -> None:
    """`hollow.py`: a quantized hollow module answering NOT hollow lets author
    `.to("cuda")` take torch's real path on parameters that allocated nothing."""

    import inspect

    from gen_worker.graphs import hollow

    source = inspect.getsource(hollow._hollow_module_moves)
    assert "is_virtual(p) for p in module.parameters()" in source, (
        "is_hollow asks isinstance again — a wrapper over fake reads as real"
    )


def test_the_host_egress_shim_asks_storage(wrapped_fake: Any) -> None:
    """`hollow.py`: a wrapper over fake has no bytes to copy, so the shim must
    not try to `.cpu()` it — that is a no-op that fails one frame later."""

    import inspect

    from gen_worker.graphs import hollow

    source = inspect.getsource(hollow.observation_shims)
    assert "not is_virtual(host)" in source
    assert "isinstance(host, FakeTensor)" not in source


def test_the_literal_values_assertion_catches_a_wrapped_fake(
    wrapped_fake: Any,
) -> None:
    """`discovery.py`: a wrapper over fake IS a table of zeros — the whole
    hazard — so the assertion that exists to catch it must not wave it past."""

    from gen_worker.graphs.discovery import DiscoveryError, _assert_literal_values_are_real

    class _Program:
        constants = {"w.weight": wrapped_fake}

    with pytest.raises(DiscoveryError, match="FAKE"):
        _assert_literal_values_are_real("lane@1", "unet", _Program())


def test_a_REAL_constant_still_passes() -> None:
    """The green half: without it the arm above only proves something raises."""

    from gen_worker.graphs.discovery import _assert_literal_values_are_real

    class _Program:
        constants = {"w.weight": torch.zeros(4, 4)}

    _assert_literal_values_are_real("lane@1", "unet", _Program())
