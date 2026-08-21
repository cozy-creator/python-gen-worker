from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import meta_instantiation as mi  # noqa: E402


class LazyPinnedRope:
    """RED — upstream z-image's `RopeEmbedder` shape (transformer_z_image.py)."""

    def __init__(self, dim: int = 8, end: int = 16) -> None:
        self.dim = dim
        self.end = end
        self.freqs_cis = None

    def __call__(self) -> torch.Tensor:
        if self.freqs_cis is None:
            with torch.device("cpu"):
                self.freqs_cis = torch.arange(self.end, dtype=torch.float32)
        return self.freqs_cis


class BoundRope(nn.Module):

    def __init__(self, dim: int = 8, end: int = 16) -> None:
        super().__init__()
        self.register_buffer(
            "freqs", torch.arange(end, dtype=torch.float32), persistent=False)

    def forward(self) -> torch.Tensor:
        return self.freqs


def test_the_bound_rope_instantiates_entirely_on_meta() -> None:
    """GREEN control: correctly-authored code allocates nothing real."""
    with mi.guard("instantiation") as census:
        with mi.meta_device():
            module = BoundRope()
    assert census.clean
    assert module.freqs.device.type == "meta"


def test_a_device_pin_at_CALL_time_is_caught_even_though_init_was_clean() -> None:
    with mi.guard("instantiation") as census:
        with mi.meta_device():
            rope = LazyPinnedRope()
    assert census.clean, "the lazy build does nothing at __init__, by design"

    with pytest.raises(mi.MetaMaterializationError) as caught:
        with mi.guard("trace"):
            with mi.meta_device():
                rope()

    error = caught.value
    assert error.phase == "trace"
    assert error.device.startswith("cpu")
    assert "register_buffer" in str(error)
    assert "torch.device" in str(error)


def test_the_refusal_names_the_ENDPOINT_site_not_an_sdk_frame() -> None:
    """A refusal whose `file:line` points into the SDK names the observer instead of the cause, and an author cannot act on it."""
    with pytest.raises(mi.MetaMaterializationError) as caught:
        with mi.guard("trace"):
            with mi.meta_device():
                LazyPinnedRope()()
    site = caught.value.site
    assert site, "the refusal carried no site at all"
    assert "/gen_worker/" not in site.replace("\\", "/")
    assert "test_meta_instantiation_pgw1080.py" in site


def test_an_explicit_real_device_kwarg_is_caught_too() -> None:
    """The pin is not the only way to escape a default."""
    with pytest.raises(mi.MetaMaterializationError):
        with mi.guard("instantiation"):
            with mi.meta_device():
                torch.zeros(4, device="cpu")


def test_fake_tensors_are_VIRTUAL_and_never_refused() -> None:
    """`FakeTensorMode` reports a real device by design — that is what makes it faithful to trace against."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with mi.guard("trace") as census:
        with FakeTensorMode():
            made = torch.zeros(8, 8, device="cpu")
            assert mi.is_virtual(made)
            assert str(made.device) == "cpu"
    assert census.clean


def test_observe_reports_without_raising() -> None:
    """The census is a measurement; only `guard` refuses."""
    with mi.observe("instantiation") as census:
        torch.zeros(2, device="cpu")
        torch.ones(3, device="cpu")
    assert not census.clean
    assert len(census.events) == 2
    assert {e.op for e in census.events} == {"zeros", "ones"}
    assert all(e.device.startswith("cpu") for e in census.events)
