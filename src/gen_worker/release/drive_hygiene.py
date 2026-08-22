"""What a hollow drive is allowed to do to the machine it runs on (pgw#1659).

A hollow derive runs the AUTHOR's `load()` and the author's pipeline against
FAKE weights. Two things follow, and neither is torchcg's to decide -- the
derive owns the drive, so the derive owns its hygiene.

**No compiled kernels.** Authors arm compiled modules in `load()`: minimax-h3
does, on its VAE decoder (`engine.py::_compile_with_eager_fallback`, a real
1.68x serve win). Inside a hollow session that compiled callable is handed FAKE
tensors, so dynamo/inductor compile against them and the generated wrapper
reads a fake data pointer and launches a real kernel on it:

    Warning: Accessing the data pointer of FakeTensor ...
    minimax-h3: compiled vae.decoder failed (AcceleratorError: CUDA error: an
    illegal memory access was encountered); serving eager for the life of this
    process

Nothing is lost by making `torch.compile` identity here. What the derive
produces is a `torch.export` of the MARKED module; what author code compiles at
load reaches neither the graph nor its key. What it cost was minutes of
inductor work per session and, on a real card, the accelerator.

**A dead accelerator is the drive's fault, and must say so.** A CUDA fault is
STICKY and process-wide: afterwards every touch of the device raises the same
error, wherever it is. Author code catches broadly -- h3's compiled-decoder
fallback and diffusers' modular block runner both swallow any `Exception` and
carry on -- so a fault raised mid-drive can be logged as a degrade while the
drive finishes green against a corpse. The next thing to touch the card then
wears the blame. In pgw#1659 that was torchcg's literal digest, and a P1 spent
its life named after `lifted_tensor_0`, a constant with nothing wrong with it.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

__all__ = [
    "accelerator_is_alive",
    "dead_accelerator_sentence",
    "eager_only_compile",
]


@contextlib.contextmanager
def eager_only_compile() -> Iterator[None]:
    """`torch.compile` and `Module.compile` are IDENTITY for this block."""

    import torch

    original_compile = torch.compile
    original_module_compile = torch.nn.Module.compile

    def eager_compile(model: Any = None, *args: Any, **kwargs: Any) -> Any:
        if model is None:
            # `@torch.compile(mode=...)` -- the decorator-factory spelling.
            return lambda inner: inner
        return model

    def eager_module_compile(module: Any, *args: Any, **kwargs: Any) -> None:
        return None

    torch.compile = eager_compile
    torch.nn.Module.compile = eager_module_compile  # type: ignore[method-assign, assignment]
    try:
        yield
    finally:
        torch.compile = original_compile
        torch.nn.Module.compile = original_module_compile  # type: ignore[method-assign]


def accelerator_is_alive(device: Any) -> bool:
    """Can this process still round-trip ONE real byte through ``device``?

    Not an availability check: `torch.cuda.is_available()` keeps answering True
    on a context an illegal access killed. This is a real allocation and a real
    copy to host -- exactly what every later victim does.
    """

    import torch

    if str(device).split(":", 1)[0] == "cpu":
        return True
    try:
        # Outside the hollow session's own redirection: its `OneTraceDevice`
        # shim answers a `cpu` ask with the trace device, which would make the
        # probe a no-op that proves nothing.
        with torch._C.DisableTorchFunction():
            torch.empty(1, device=device).detach().to("cpu")
    except Exception:
        return False
    return True


def dead_accelerator_sentence(device_type: str) -> str:
    """The refusal that names the DRIVE instead of the drive's next victim."""

    return (
        f"the drive left this process's {device_type} context DEAD — a real "
        f"kernel faulted while the pipeline was being driven and author code "
        f"swallowed it, so nothing after this point can touch the device. The "
        f"graphs are NOT the cause and re-running the export will not help: "
        f"find what launched a real kernel during a hollow drive (a "
        f"`torch.compile`d module handed FAKE tensors is the known one, "
        f"pgw#1659) or drive on cpu. Re-run with CUDA_LAUNCH_BLOCKING=1 to "
        f"place the fault."
    )
