"""The ONE meta-instantiation seam, owned and proven in-process."""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from ..api.errors import WorkerError

CAPABILITY = "meta-instantiation (parameters on meta, buffers real)"


class MetaInitUnavailable(WorkerError):
    """This PROCESS cannot meta-instantiate — a broken image, never normal."""

    def __init__(self, *, capability: str, lacks: str) -> None:
        self.capability = str(capability or CAPABILITY)
        self.lacks = str(lacks or "")
        super().__init__(
            f"this process cannot provide {self.capability}: {self.lacks}. "
            f"Weight-free instantiation is how a compile target is built from "
            f"CODE + CONFIG, so without it no boot key can be derived, no "
            f"compiled graph can be asked for, and this pod will self-mint on every "
            f"boot")


@contextlib.contextmanager
def init_empty_weights() -> Iterator[None]:
    """Instantiate modules with their PARAMETERS on ``meta`` and no storage."""
    import torch
    from torch import nn

    original = nn.Module.register_parameter
    meta = torch.device("meta")

    def register_empty_parameter(
        module: Any, name: str, param: Any,
    ) -> None:
        original(module, name, param)
        held = module._parameters.get(name)
        if held is None:
            return
        cls = type(held)
        kwargs = dict(held.__dict__)
        kwargs["requires_grad"] = held.requires_grad
        module._parameters[name] = cls(held.to(meta), **kwargs)

    setattr(nn.Module, "register_parameter", register_empty_parameter)
    try:
        yield
    finally:
        setattr(nn.Module, "register_parameter", original)


def require_meta_init() -> None:
    """Prove the capability on THIS process, or refuse naming it."""
    try:
        import torch
        from torch import nn
    except Exception as exc:  # noqa: BLE001
        raise MetaInitUnavailable(
            capability=CAPABILITY,
            lacks=f"`torch` is not importable ({exc})") from exc

    class _Probe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(2, 2))
            self.register_buffer("table", torch.zeros(2))

    try:
        with init_empty_weights():
            probe = _Probe()
    except Exception as exc:  # noqa: BLE001
        raise MetaInitUnavailable(
            capability=CAPABILITY,
            lacks=f"building a probe module refused ({exc!r})") from exc

    if probe.weight.device.type != "meta":
        raise MetaInitUnavailable(
            capability=CAPABILITY,
            lacks=(f"a parameter built under the context manager landed on "
                   f"{probe.weight.device!r}, not on meta — this build would "
                   f"hold real weights"))
    buffer = probe.get_buffer("table")
    if buffer.device.type == "meta":
        raise MetaInitUnavailable(
            capability=CAPABILITY,
            lacks=("a BUFFER built under the context manager landed on meta; "
                   "buffers are config-derived values a literal-bearing compiled graph "
                   "packs, and a fake one makes the package unbuildable"))


__all__ = [
    "CAPABILITY", "MetaInitUnavailable", "init_empty_weights",
    "require_meta_init",
]
