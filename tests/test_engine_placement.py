"""The streaming engine places tensors the CONTAINER did not carry."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker.serving.streaming.engine import StreamingLoader  # noqa: E402

from harness.nvml import nvml_is_healthy  # noqa: E402


class _Tiny(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        self.register_buffer("position_ids", torch.arange(2), persistent=False)


class _Pipe:

    def __init__(self, module: nn.Module) -> None:
        self.module = module

    @property
    def components(self) -> dict:
        return {"module": self.module}


def _sweep(pipeline: object, device: str) -> None:
    StreamingLoader._place_uninstalled(None, pipeline, device)  # type: ignore[arg-type]


def test_a_meta_tensor_is_left_alone() -> None:
    """A tensor still on `meta` was never installed."""
    module = _Tiny()
    with torch.device("meta"):
        stranded = torch.arange(2)
    module._buffers["position_ids"] = stranded
    _sweep(_Pipe(module), "cpu")
    stayed = module._buffers["position_ids"]
    assert stayed is not None and stayed.device.type == "meta", (
        "the sweep moved a meta tensor — that invents storage the container "
        "never delivered and hides the survivors check's own refusal"
    )


def test_the_sweep_is_a_no_op_when_everything_is_already_placed() -> None:
    """It must not re-copy what the engine already streamed."""
    module = _Tiny()
    held = module._buffers["position_ids"]
    assert held is not None
    before = (module.weight.data_ptr(), held.data_ptr())
    _sweep(_Pipe(module), "cpu")
    after_buf = module._buffers["position_ids"]
    assert after_buf is not None
    assert (module.weight.data_ptr(), after_buf.data_ptr()) == before


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

    _sweep(_Pipe(module), "cuda")

    placed = module._buffers["position_ids"]
    assert placed is not None and placed.device.type == "cuda", (
        "a buffer absent from the container was left on the host; the first "
        "op reading it dies with 'index is on cpu, different from other "
        "tensors on cuda:0'"
    )
