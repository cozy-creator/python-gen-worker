"""The streaming engine places tensors the CONTAINER did not carry.

pgw#1454, isolated to a single tensor on a real 4070: after a clean engine load
every parameter of sd1.5's text encoder was on `cuda:0` and exactly one thing
was not —

    BUFFER embeddings.position_ids -> cpu

— so the first `nn.Embedding` forward died with "index is on cpu, different
from other tensors on cuda:0". The `index` in that message is a BUFFER the
engine never placed, not a request tensor.

The engine builds the skeleton on `meta` and installs each tensor the container
NAMES. A tensor absent from the container is never visited, so it materialises
where `__init__` put it — the host — and nothing moves it.
`from_pretrained` cannot have this bug: it builds on CPU and then `.to(device)`
moves parameters and buffers together, whatever their provenance.

And conversion is what EXPOSES it: `position_ids` is a non-persistent buffer, so
a modern `state_dict()` omits it by design (sd1.5's raw mirror carries 197 keys,
the reconverted tree 196). A conversion step that is individually correct
produced a tree this loader could not serve.

The safety property is the one that runs everywhere, and it is the one worth
guarding hardest: a tensor still on `meta` was NEVER INSTALLED, and moving it
would fabricate uninitialised memory and hide the mismatch the engine's own
survivors check exists to raise. That arm needs no GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker.serving.streaming.engine import StreamingLoader  # noqa: E402

from harness.nvml import nvml_is_healthy  # noqa: E402


class _Tiny(nn.Module):
    """One installed parameter, one buffer the container would not carry."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        # `persistent=False` is exactly CLIP's `position_ids`: real at runtime,
        # absent from `state_dict()`, therefore absent from any container built
        # out of one.
        self.register_buffer("position_ids", torch.arange(2), persistent=False)


class _Pipe:
    """The `components` mapping the engine walks."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module

    @property
    def components(self) -> dict:
        return {"module": self.module}


def _sweep(pipeline: object, device: str) -> None:
    StreamingLoader._place_uninstalled(None, pipeline, device)  # type: ignore[arg-type]


def test_a_meta_tensor_is_left_alone() -> None:
    """A tensor still on `meta` was never installed. Moving it would fabricate
    uninitialised memory and turn a loud, correct refusal into silent garbage —
    so the sweep must not touch it, and this arm needs no GPU to prove it."""
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
    """It must not re-copy what the engine already streamed. Same device in,
    same tensor objects out — the whole point of streaming is not paying for a
    second pass over the weights."""
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
    """RED before pgw#1454: the buffer stayed on the host and the first forward
    that read it raised a device mismatch."""
    module = _Tiny().to("cuda")
    # What the engine leaves behind: parameters installed on the device, a
    # non-persistent buffer still on the host because no container named it.
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
